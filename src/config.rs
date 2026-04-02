use crate::{
    backend::{MlxBackend, SharedBackend, SyntheticBackend, VllmBackend, default_backends},
    types::*,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    pub node_id: Option<String>,
    pub protocol_version: Option<String>,
    pub auth_token: Option<String>,
    pub backends: Option<Vec<BackendConfig>>,
    pub mesh: Option<MeshConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendConfig {
    Synthetic {
        name: String,
    },
    Mlx {
        python: String,
        model: String,
    },
    Vllm {
        base_url: String,
        api_key: Option<String>,
    },
}

pub fn load(path: Option<&Path>) -> Result<RuntimeConfig> {
    match path {
        Some(path) => {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("failed to read config {}", path.display()))?;
            Ok(toml::from_str(&raw).context("failed to parse TOML runtime config")?)
        }
        None => {
            let default = Path::new("zeitgeist.toml");
            if default.exists() {
                let raw = fs::read_to_string(default)
                    .with_context(|| format!("failed to read config {}", default.display()))?;
                Ok(toml::from_str(&raw).context("failed to parse TOML runtime config")?)
            } else {
                Ok(RuntimeConfig::default())
            }
        }
    }
}

pub fn node_identity(config: &RuntimeConfig) -> NodeIdentity {
    NodeIdentity {
        node_id: config
            .node_id
            .clone()
            .unwrap_or_else(|| format!("zeitgeist-{}", uuid::Uuid::new_v4())),
        protocol_version: config
            .protocol_version
            .clone()
            .unwrap_or_else(|| "0.1.0".into()),
        transports: vec!["in_process".into(), "tcp".into(), "quic".into()],
        trust_level: TrustLevel::TrustedExecutor,
        auth_modes: vec![
            AuthMode::None,
            AuthMode::SharedToken,
            AuthMode::SignedNodeIdentity,
        ],
        hardware: HardwareProfile {
            architecture: std::env::consts::ARCH.into(),
            accelerator: if cfg!(target_os = "macos") {
                "metal".into()
            } else {
                "cuda_or_cpu".into()
            },
            total_memory_mb: 32 * 1024,
            shared_memory: cfg!(target_os = "macos"),
        },
        memory: MemoryProfile {
            available_memory_mb: 24 * 1024,
            kv_cache_budget_mb: 8 * 1024,
        },
        health: HealthStatus::Healthy,
    }
}

pub fn backends(config: &RuntimeConfig) -> Vec<SharedBackend> {
    let Some(configured) = &config.backends else {
        return default_backends();
    };
    configured
        .iter()
        .map(|entry| match entry {
            BackendConfig::Synthetic { name } => {
                let mut descriptor = default_descriptor(name);
                descriptor
                    .metadata
                    .insert("mode".into(), "synthetic".into());
                Arc::new(SyntheticBackend::new(descriptor)) as SharedBackend
            }
            BackendConfig::Mlx { python, model } => {
                Arc::new(MlxBackend::new(python.clone(), model.clone())) as SharedBackend
            }
            BackendConfig::Vllm { base_url, api_key } => {
                Arc::new(VllmBackend::new(base_url.clone(), api_key.clone())) as SharedBackend
            }
        })
        .collect()
}

pub fn mesh(config: &RuntimeConfig) -> Option<MeshConfig> {
    config.mesh.clone().map(|mut mesh| {
        if mesh.sync_interval_ms == 0 {
            mesh.sync_interval_ms = 30_000;
        }
        mesh
    })
}

pub fn models() -> Vec<ModelIdentity> {
    vec![
        ModelIdentity {
            model_id: "llama-3.2-3b-instruct".into(),
            family: "llama".into(),
            architecture: "decoder_only_transformer".into(),
            parameter_count: 3_210_000_000,
            tokenizer_id: "llama3.2".into(),
            tokenizer_hash: "sha256:tokenizer-llama32".into(),
            vocabulary_hash: "sha256:vocab-llama32".into(),
            position_encoding: PositionEncoding::Rope,
            rope_scaling: Some("dynamic".into()),
            attention_variant: AttentionVariant::Flash,
            hidden_size: 3072,
            layer_count: 28,
            expert_count: None,
            quantization: QuantizationDescriptor {
                format: QuantFormat::Q4,
                group_size: Some(64),
                scale_dtype: Some(DType::F16),
                zero_point_dtype: None,
                packing_layout: Some("grouped".into()),
                calibration: None,
            },
            tensor_layout: TensorLayout::RowMajorContiguous,
            artifact_hash: "sha256:artifact-llama32-3b".into(),
            revision: "main".into(),
        },
        ModelIdentity {
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            family: "qwen".into(),
            architecture: "decoder_only_transformer".into(),
            parameter_count: 494_000_000,
            tokenizer_id: "qwen2.5".into(),
            tokenizer_hash: "sha256:tokenizer-qwen25-05b".into(),
            vocabulary_hash: "sha256:vocab-qwen25".into(),
            position_encoding: PositionEncoding::Rope,
            rope_scaling: Some("dynamic".into()),
            attention_variant: AttentionVariant::Flash,
            hidden_size: 896,
            layer_count: 24,
            expert_count: None,
            quantization: QuantizationDescriptor {
                format: QuantFormat::None,
                group_size: None,
                scale_dtype: None,
                zero_point_dtype: None,
                packing_layout: None,
                calibration: None,
            },
            tensor_layout: TensorLayout::RowMajorContiguous,
            artifact_hash: "sha256:artifact-qwen25-05b".into(),
            revision: "main".into(),
        },
    ]
}

pub fn kernels() -> Vec<KernelDescriptor> {
    vec![
        KernelDescriptor {
            name: "metal_flash_attention".into(),
            implementation_target: "metal".into(),
            op_type: "attention".into(),
            supported_dtypes: vec![DType::F16, DType::BF16],
            supported_layouts: vec![TensorLayout::RowMajorContiguous],
            supported_hardware: vec!["apple_silicon".into()],
            deterministic: true,
            memory_requirement_mb: 256,
        },
        KernelDescriptor {
            name: "cuda_paged_attention".into(),
            implementation_target: "cuda".into(),
            op_type: "attention".into(),
            supported_dtypes: vec![DType::Fp8E4m3, DType::Fp8E5m2, DType::F16, DType::BF16],
            supported_layouts: vec![
                TensorLayout::RowMajorContiguous,
                TensorLayout::BackendBlocked,
            ],
            supported_hardware: vec!["nvidia_gpu".into()],
            deterministic: false,
            memory_requirement_mb: 512,
        },
    ]
}

fn default_descriptor(name: &str) -> BackendDescriptor {
    let mut metadata = BTreeMap::new();
    metadata.insert("mode".into(), "synthetic".into());
    BackendDescriptor {
        name: name.into(),
        version: "synthetic".into(),
        trust_level: TrustLevel::TrustedExecutor,
        topology: BackendTopologyHints {
            locality: "local".into(),
            zone: "default".into(),
            hop_count: 0,
            base_latency_ms: 1,
        },
        memory_budget_mb: 8 * 1024,
        attestation: Some(BackendAttestation {
            format: "artifact_hash".into(),
            signer: "zeitgeist-synthetic".into(),
            artifact_hash: format!("sha256:synthetic-{name}"),
            verified: true,
            signature: Some("sig:verified".into()),
        }),
        execution_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing],
        model_families: vec!["llama".into()],
        quantization: vec![QuantFormat::None, QuantFormat::Q4],
        dtypes: vec![DType::F16, DType::BF16],
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
            ExecutionMode::RoutedServing,
            ExecutionMode::PipelineParallel,
            ExecutionMode::TensorParallel,
            ExecutionMode::ExpertParallel,
            ExecutionMode::Hybrid,
        ],
        streaming: true,
        batching: true,
        extensions: vec![],
        metadata,
    }
}
