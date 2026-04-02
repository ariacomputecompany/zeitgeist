use crate::types::*;
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};
use tokio::process::Command;

#[async_trait]
pub trait BackendAdapter: Send + Sync {
    fn descriptor(&self) -> BackendDescriptor;
    fn name(&self) -> &str;
    async fn execute(&self, request: &JobRequest, plan: &JobPlan) -> Result<JobResult>;
}

pub type SharedBackend = Arc<dyn BackendAdapter>;

pub struct SyntheticBackend {
    descriptor: BackendDescriptor,
    failures_remaining: AtomicU32,
}

impl SyntheticBackend {
    pub fn new(descriptor: BackendDescriptor) -> Self {
        let failures_remaining = match descriptor.metadata.get("force_fail").map(String::as_str) {
            Some("once") => 1,
            _ => 0,
        };
        Self {
            descriptor,
            failures_remaining: AtomicU32::new(failures_remaining),
        }
    }
}

#[async_trait]
impl BackendAdapter for SyntheticBackend {
    fn descriptor(&self) -> BackendDescriptor {
        self.descriptor.clone()
    }

    fn name(&self) -> &str {
        &self.descriptor.name
    }

    async fn execute(&self, request: &JobRequest, plan: &JobPlan) -> Result<JobResult> {
        match self.descriptor.metadata.get("force_fail").map(String::as_str) {
            Some("always") => anyhow::bail!("synthetic backend {} forced to fail", self.name()),
            Some("once") => {
                if self
                    .failures_remaining
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                        if current > 0 {
                            Some(current - 1)
                        } else {
                            None
                        }
                    })
                    .is_ok()
                {
                    anyhow::bail!("synthetic backend {} forced transient failure", self.name());
                }
            }
            _ => {}
        }
        let token_count = request.max_tokens.min(64);
        let seed = request
            .prompt
            .bytes()
            .fold(0u64, |acc, byte| acc + u64::from(byte))
            + token_count as u64;
        let text = match request.job_type {
            JobType::Embedding => format!("embedding:{}:{}:{}", self.name(), plan.mode_string(), seed),
            JobType::ChatCompletion | JobType::TextCompletion | JobType::DistributedShardExecution => {
                format!(
                    "[{}:{}] {}",
                    self.name(),
                    plan.mode_string(),
                    request.prompt.chars().take(120).collect::<String>()
                )
            }
            _ => format!("[{}] executed {}", self.name(), serde_json::to_string(&request.job_type)?),
        };
        let embeddings = if request.job_type == JobType::Embedding {
            Some(
                (0..8)
                    .map(|index| (((seed + index as u64) % 100) as f32) / 100.0)
                    .collect(),
            )
        } else {
            None
        };

        Ok(JobResult {
            text,
            tokens: token_count,
            latency_ms: if request.determinism.low_latency { 8 } else { 16 },
            backend: self.name().to_string(),
            embeddings,
        })
    }
}

pub struct MlxBackend {
    descriptor: BackendDescriptor,
    python: String,
    model: String,
}

impl MlxBackend {
    pub fn new(python: String, model: String) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("execution_surface".into(), "mlx_lm.load + generate/stream_generate".into());
        metadata.insert("interchange_hint".into(), "buffer_protocol, dlpack".into());
        Self {
            descriptor: BackendDescriptor {
                name: "mlx".into(),
                version: "proxy".into(),
                trust_level: TrustLevel::TrustedExecutor,
                topology: BackendTopologyHints {
                    locality: "local".into(),
                    zone: "apple_silicon".into(),
                    hop_count: 0,
                    base_latency_ms: 2,
                },
                memory_budget_mb: 12 * 1024,
                attestation: Some(BackendAttestation {
                    format: "artifact_hash".into(),
                    signer: "zeitgeist-local".into(),
                    artifact_hash: "sha256:mlx-backend-attestation".into(),
                    verified: true,
                }),
                execution_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing, ExecutionMode::PipelineParallel],
                model_families: vec!["llama".into(), "mistral".into()],
                quantization: vec![QuantFormat::None, QuantFormat::Q4, QuantFormat::Q8],
                dtypes: vec![DType::F16, DType::BF16, DType::F32],
                attention: vec![AttentionVariant::Standard, AttentionVariant::Flash],
                cache: vec![CacheDescriptor {
                    version: "zgc-1".into(),
                    dtype: DType::F16,
                    layout: TensorLayout::RowMajorContiguous,
                    head_grouping: "grouped-query".into(),
                    rope_state: PositionEncoding::Rope,
                    sequence_indexing: "absolute".into(),
                    eviction: "lru".into(),
                    compression: None,
                    transferable: true,
                }],
                tensor_layouts: vec![TensorLayout::RowMajorContiguous],
                parallelism: vec![
                    ExecutionMode::Solo,
                    ExecutionMode::PipelineParallel,
                    ExecutionMode::TensorParallel,
                    ExecutionMode::ExpertParallel,
                    ExecutionMode::Hybrid,
                ],
                streaming: true,
                batching: false,
                extensions: vec!["metal".into(), "dlpack".into()],
                metadata,
            },
            python,
            model,
        }
    }
}

#[async_trait]
impl BackendAdapter for MlxBackend {
    fn descriptor(&self) -> BackendDescriptor {
        self.descriptor.clone()
    }

    fn name(&self) -> &str {
        &self.descriptor.name
    }

    async fn execute(&self, request: &JobRequest, _plan: &JobPlan) -> Result<JobResult> {
        let script = format!(
            "from mlx_lm import load, generate\nmodel, tokenizer = load({model:?})\ntext = generate(model, tokenizer, prompt={prompt:?}, max_tokens={max_tokens})\nprint(text)",
            model = self.model,
            prompt = request.prompt,
            max_tokens = request.max_tokens.min(64),
        );
        let output = Command::new(&self.python)
            .arg("-c")
            .arg(script)
            .output()
            .await
            .context("failed to launch mlx backend")?;
        if !output.status.success() {
            anyhow::bail!(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
        Ok(JobResult {
            text: String::from_utf8_lossy(&output.stdout).trim().to_string(),
            tokens: request.max_tokens.min(64),
            latency_ms: 12,
            backend: self.name().into(),
            embeddings: None,
        })
    }
}

pub struct VllmBackend {
    descriptor: BackendDescriptor,
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl VllmBackend {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("execution_surface".into(), "OpenAI-compatible HTTP server".into());
        metadata.insert("kv_cache_dtype".into(), "auto, fp8, fp8_e4m3, fp8_e5m2".into());
        Self {
            descriptor: BackendDescriptor {
                name: "vllm".into(),
                version: "proxy".into(),
                trust_level: TrustLevel::TrustedExecutor,
                topology: BackendTopologyHints {
                    locality: "remote".into(),
                    zone: "linux_gpu".into(),
                    hop_count: 1,
                    base_latency_ms: 8,
                },
                memory_budget_mb: 24 * 1024,
                attestation: Some(BackendAttestation {
                    format: "artifact_hash".into(),
                    signer: "zeitgeist-linux".into(),
                    artifact_hash: "sha256:vllm-backend-attestation".into(),
                    verified: true,
                }),
                execution_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing, ExecutionMode::PipelineParallel],
                model_families: vec!["llama".into(), "mistral".into(), "qwen".into()],
                quantization: vec![QuantFormat::None, QuantFormat::Fp8, QuantFormat::Q4, QuantFormat::Q8],
                dtypes: vec![DType::Fp8E4m3, DType::Fp8E5m2, DType::F16, DType::BF16],
                attention: vec![AttentionVariant::Standard, AttentionVariant::Flash],
                cache: vec![CacheDescriptor {
                    version: "zgc-1".into(),
                    dtype: DType::Fp8E4m3,
                    layout: TensorLayout::RowMajorContiguous,
                    head_grouping: "grouped-query".into(),
                    rope_state: PositionEncoding::Rope,
                    sequence_indexing: "absolute".into(),
                    eviction: "paged".into(),
                    compression: Some("fp8".into()),
                    transferable: true,
                }],
                tensor_layouts: vec![TensorLayout::RowMajorContiguous, TensorLayout::BackendBlocked],
                parallelism: vec![
                    ExecutionMode::Solo,
                    ExecutionMode::RoutedServing,
                    ExecutionMode::PipelineParallel,
                    ExecutionMode::TensorParallel,
                    ExecutionMode::ExpertParallel,
                    ExecutionMode::Hybrid,
                ],
                streaming: true,
                batching: true,
                extensions: vec!["openai_compatible".into()],
                metadata,
            },
            base_url,
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl BackendAdapter for VllmBackend {
    fn descriptor(&self) -> BackendDescriptor {
        self.descriptor.clone()
    }

    fn name(&self) -> &str {
        &self.descriptor.name
    }

    async fn execute(&self, request: &JobRequest, _plan: &JobPlan) -> Result<JobResult> {
        let endpoint = match request.job_type {
            JobType::Embedding => "/v1/embeddings",
            _ => "/v1/chat/completions",
        };
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(api_key) = &self.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}"))?;
            headers.insert(AUTHORIZATION, value);
        }
        let body = match request.job_type {
            JobType::Embedding => json!({
                "model": request.model_id,
                "input": request.prompt,
            }),
            _ => json!({
                "model": request.model_id,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }),
        };
        let response = self
            .client
            .post(format!("{}{}", self.base_url.trim_end_matches('/'), endpoint))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .context("failed to reach vllm backend")?;
        let status = response.status();
        let payload: serde_json::Value = response.json().await?;
        if !status.is_success() {
            anyhow::bail!(payload.to_string());
        }
        let text = payload
            .pointer("/choices/0/message/content")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let embeddings = payload
            .pointer("/data/0/embedding")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_f64().map(|number| number as f32))
                    .collect::<Vec<_>>()
            });
        Ok(JobResult {
            text,
            tokens: request.max_tokens,
            latency_ms: 10,
            backend: self.name().into(),
            embeddings,
        })
    }
}

pub fn default_backends() -> Vec<SharedBackend> {
    let mlx = SyntheticBackend::new(MlxBackend::new("python3".into(), "mlx-community/Llama-3.2-1B-Instruct-4bit".into()).descriptor());
    let vllm = SyntheticBackend::new(VllmBackend::new("http://127.0.0.1:8000".into(), None).descriptor());
    vec![Arc::new(mlx), Arc::new(vllm)]
}

trait PlanModeString {
    fn mode_string(&self) -> &'static str;
}

impl PlanModeString for JobPlan {
    fn mode_string(&self) -> &'static str {
        match self.mode {
            ExecutionMode::Solo => "solo",
            ExecutionMode::RoutedServing => "routed",
            ExecutionMode::TensorParallel => "tensor_parallel",
            ExecutionMode::PipelineParallel => "pipeline_parallel",
            ExecutionMode::ExpertParallel => "expert_parallel",
            ExecutionMode::Hybrid => "hybrid",
            ExecutionMode::ClientOnly => "client_only",
        }
    }
}
