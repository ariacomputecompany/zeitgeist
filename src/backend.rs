use crate::{peer, types::*};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};
use tokio::process::Command;
use tokio::sync::RwLock;

#[async_trait]
pub trait BackendAdapter: Send + Sync {
    fn descriptor(&self) -> BackendDescriptor;
    fn name(&self) -> &str;
    async fn execute(&self, request: &JobRequest, plan: &JobPlan) -> Result<JobResult>;
}

pub type SharedBackend = Arc<dyn BackendAdapter>;

pub struct RemotePeerBackend {
    descriptor: BackendDescriptor,
    node_id: String,
    peer_tcp_addr: String,
    management_url: Option<String>,
    auth_token: Option<String>,
    protocol_version: String,
    upstream_backend_name: String,
}

impl RemotePeerBackend {
    pub fn new(
        node_id: String,
        peer_tcp_addr: String,
        management_url: Option<String>,
        auth_token: Option<String>,
        protocol_version: String,
        mut descriptor: BackendDescriptor,
    ) -> Self {
        let upstream_backend_name = descriptor.name.clone();
        descriptor.name = format!("{}/{}", node_id, descriptor.name);
        descriptor.topology.locality = "mesh_remote".into();
        descriptor.topology.hop_count = descriptor.topology.hop_count.saturating_add(1);
        descriptor
            .metadata
            .insert("mesh_node_id".into(), node_id.clone());
        descriptor
            .metadata
            .insert("mesh_peer_tcp_addr".into(), peer_tcp_addr.clone());
        if let Some(management_url) = &management_url {
            descriptor
                .metadata
                .insert("mesh_management_url".into(), management_url.clone());
        }
        Self {
            descriptor,
            node_id,
            peer_tcp_addr,
            management_url,
            auth_token,
            protocol_version,
            upstream_backend_name,
        }
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }
}

#[async_trait]
impl BackendAdapter for RemotePeerBackend {
    fn descriptor(&self) -> BackendDescriptor {
        self.descriptor.clone()
    }

    fn name(&self) -> &str {
        &self.descriptor.name
    }

    async fn execute(&self, request: &JobRequest, _plan: &JobPlan) -> Result<JobResult> {
        let mut forwarded = request.clone();
        forwarded.preferred_backends = vec![self.upstream_backend_name.clone()];
        let peer_request = peer::PeerRequest::ExecuteJob {
            protocol_version: self.protocol_version.clone(),
            auth_token: self.auth_token.clone(),
            request: forwarded,
        };
        let response = match peer::send(&self.peer_tcp_addr, &peer_request).await {
            Ok(response) => response,
            Err(tcp_error) => {
                let Some(management_url) = &self.management_url else {
                    return Err(tcp_error);
                };
                peer::send_http(management_url, &peer_request)
                    .await
                    .with_context(|| {
                        format!(
                            "mesh peer {} TCP failed ({tcp_error}); HTTP fallback also failed",
                            self.node_id
                        )
                    })?
            }
        };
        match response {
            peer::PeerResponse::ExecuteJob { record } => record
                .result
                .ok_or_else(|| anyhow!("remote peer {} returned no job result", self.node_id)),
            peer::PeerResponse::Error { message, .. } => {
                anyhow::bail!("remote peer {} execution failed: {}", self.node_id, message)
            }
            other => anyhow::bail!("unexpected mesh execute response: {:?}", other),
        }
    }
}

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
        match self
            .descriptor
            .metadata
            .get("force_fail")
            .map(String::as_str)
        {
            Some("always") => anyhow::bail!("synthetic backend {} forced to fail", self.name()),
            Some("once") => {
                if self
                    .failures_remaining
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                        if current > 0 { Some(current - 1) } else { None }
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
            JobType::Embedding => {
                format!("embedding:{}:{}:{}", self.name(), plan.mode_string(), seed)
            }
            JobType::ChatCompletion
            | JobType::TextCompletion
            | JobType::DistributedShardExecution => {
                format!(
                    "[{}:{}] {}",
                    self.name(),
                    plan.mode_string(),
                    request.prompt.chars().take(120).collect::<String>()
                )
            }
            _ => format!(
                "[{}] executed {}",
                self.name(),
                serde_json::to_string(&request.job_type)?
            ),
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
            latency_ms: if request.determinism.low_latency {
                8
            } else {
                16
            },
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
        metadata.insert(
            "execution_surface".into(),
            "mlx_lm.load + generate/stream_generate".into(),
        );
        metadata.insert("interchange_hint".into(), "buffer_protocol, dlpack".into());
        let mut model_families = vec!["llama".to_string(), "mistral".to_string()];
        if model.to_ascii_lowercase().contains("qwen") {
            model_families.push("qwen".into());
        }
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
                    signature: Some("sig:verified".into()),
                }),
                execution_modes: vec![
                    ExecutionMode::Solo,
                    ExecutionMode::RoutedServing,
                    ExecutionMode::PipelineParallel,
                ],
                model_families,
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
    model_aliases: RwLock<BTreeMap<String, String>>,
}

impl VllmBackend {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "execution_surface".into(),
            "OpenAI-compatible HTTP server".into(),
        );
        metadata.insert(
            "kv_cache_dtype".into(),
            "auto, fp8, fp8_e4m3, fp8_e5m2".into(),
        );
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
                    signature: Some("sig:verified".into()),
                }),
                execution_modes: vec![
                    ExecutionMode::Solo,
                    ExecutionMode::RoutedServing,
                    ExecutionMode::PipelineParallel,
                ],
                model_families: vec!["llama".into(), "mistral".into(), "qwen".into()],
                quantization: vec![
                    QuantFormat::None,
                    QuantFormat::Fp8,
                    QuantFormat::Q4,
                    QuantFormat::Q8,
                ],
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
                tensor_layouts: vec![
                    TensorLayout::RowMajorContiguous,
                    TensorLayout::BackendBlocked,
                ],
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
            model_aliases: RwLock::new(BTreeMap::new()),
        }
    }

    async fn resolve_model_id(&self, requested_model_id: &str) -> String {
        if let Some(alias) = self
            .model_aliases
            .read()
            .await
            .get(requested_model_id)
            .cloned()
        {
            return alias;
        }

        let mut headers = HeaderMap::new();
        if let Some(api_key) = &self.api_key {
            match HeaderValue::from_str(&format!("Bearer {api_key}")) {
                Ok(value) => {
                    headers.insert(AUTHORIZATION, value);
                }
                Err(_) => return requested_model_id.to_string(),
            }
        }

        let response = match self
            .client
            .get(format!("{}/v1/models", self.base_url.trim_end_matches('/')))
            .headers(headers)
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => return requested_model_id.to_string(),
        };
        if !response.status().is_success() {
            return requested_model_id.to_string();
        }

        let payload: VllmModelsResponse = match response.json().await {
            Ok(payload) => payload,
            Err(_) => return requested_model_id.to_string(),
        };

        let mut aliases = self.model_aliases.write().await;
        for model in payload.data {
            aliases.insert(model.id.clone(), model.id.clone());
            if let Some(root) = model.root {
                aliases.insert(root, model.id.clone());
            }
        }

        aliases
            .get(requested_model_id)
            .cloned()
            .unwrap_or_else(|| requested_model_id.to_string())
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
        let upstream_model_id = self.resolve_model_id(&request.model_id).await;
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
                "model": upstream_model_id,
                "input": request.prompt,
            }),
            _ => json!({
                "model": upstream_model_id,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }),
        };
        let response = self
            .client
            .post(format!(
                "{}{}",
                self.base_url.trim_end_matches('/'),
                endpoint
            ))
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
    let mlx_python = std::env::var("ZEITGEIST_MLX_PYTHON")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "python3".into());
    let mlx_model = std::env::var("ZEITGEIST_MLX_MODEL")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "mlx-community/Llama-3.2-1B-Instruct-4bit".into());
    let vllm_base_url = std::env::var("ZEITGEIST_VLLM_BASE_URL")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".into());
    let vllm_api_key = std::env::var("ZEITGEIST_VLLM_API_KEY")
        .ok()
        .filter(|value| !value.is_empty());

    vec![
        Arc::new(MlxBackend::new(mlx_python, mlx_model)) as SharedBackend,
        Arc::new(VllmBackend::new(vllm_base_url, vllm_api_key)) as SharedBackend,
    ]
}

pub fn synthetic_backends() -> Vec<SharedBackend> {
    let mlx = SyntheticBackend::new(
        MlxBackend::new(
            "python3".into(),
            "mlx-community/Llama-3.2-1B-Instruct-4bit".into(),
        )
        .descriptor(),
    );
    let vllm =
        SyntheticBackend::new(VllmBackend::new("http://127.0.0.1:8000".into(), None).descriptor());
    vec![Arc::new(mlx), Arc::new(vllm)]
}

#[derive(Debug, Deserialize)]
struct VllmModelsResponse {
    data: Vec<VllmModelEntry>,
}

#[derive(Debug, Deserialize)]
struct VllmModelEntry {
    id: String,
    root: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::State, routing::get, routing::post};
    use std::{net::SocketAddr, sync::Arc as StdArc};
    use tokio::net::TcpListener;

    #[derive(Clone, Default)]
    struct MockState {
        seen_models: StdArc<RwLock<Vec<String>>>,
    }

    #[tokio::test]
    async fn vllm_backend_resolves_root_model_to_served_model_id() {
        async fn models() -> Json<serde_json::Value> {
            Json(serde_json::json!({
                "object": "list",
                "data": [{
                    "id": "served-qwen",
                    "root": "Qwen/Qwen2.5-0.5B-Instruct"
                }]
            }))
        }

        async fn completions(
            State(state): State<MockState>,
            Json(body): Json<serde_json::Value>,
        ) -> Json<serde_json::Value> {
            state.seen_models.write().await.push(
                body.get("model")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
            );
            Json(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "ok"
                    }
                }]
            }))
        }

        let state = MockState::default();
        let app = Router::new()
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(completions))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let backend = VllmBackend::new(format!("http://{addr}"), None);
        let result = backend
            .execute(
                &JobRequest {
                    model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
                    job_type: JobType::ChatCompletion,
                    prompt: "hello".into(),
                    session_id: None,
                    preferred_backends: vec!["vllm".into()],
                    max_tokens: 8,
                    temperature: 0.0,
                    determinism: DeterminismPolicy {
                        strict_correctness: true,
                        deterministic: true,
                        low_latency: true,
                        high_availability: false,
                    },
                },
                &JobPlan {
                    session_id: uuid::Uuid::nil(),
                    mode: ExecutionMode::Solo,
                    compatibility: CompatibilityReport {
                        outcome: CompatibilityOutcome::FullyCompatible,
                        execution_mode: ExecutionMode::Solo,
                        convertible: false,
                        reasons: vec![],
                        selected_peers: vec!["vllm".into()],
                    },
                    participants: vec![],
                    tensor_layout: TensorLayout::RowMajorContiguous,
                    cache: Some(CacheDescriptor {
                        version: "zgc-1".into(),
                        dtype: DType::F16,
                        layout: TensorLayout::RowMajorContiguous,
                        head_grouping: "grouped-query".into(),
                        rope_state: PositionEncoding::Rope,
                        sequence_indexing: "absolute".into(),
                        eviction: "lru".into(),
                        compression: None,
                        transferable: true,
                    }),
                    fallback_modes: vec![ExecutionMode::Solo],
                    estimated_cost: 1,
                    replan_generation: 0,
                    partial_failure_tolerance: true,
                },
            )
            .await
            .unwrap();

        assert_eq!(result.text, "ok");
        assert_eq!(state.seen_models.read().await.as_slice(), ["served-qwen"]);
    }
}
