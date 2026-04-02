use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    None,
    SharedToken,
    MutualTls,
    SignedNodeIdentity,
    BackendSignedAttestation,
    Extension,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    TrustedExecutor,
    TrustedCachePeer,
    TrustedTensorPeer,
    ApiOnlyPeer,
    UntrustedExternalClient,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Fp8E4m3,
    Fp8E5m2,
    F16,
    BF16,
    F32,
    I8,
    U8,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    RowMajorContiguous,
    ColumnMajorContiguous,
    QuantizedTile,
    BackendBlocked,
    Sparse,
    Extension,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttentionVariant {
    Standard,
    Flash,
    GroupedQuery,
    SlidingWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PositionEncoding {
    Rope,
    Alibi,
    LearnedAbsolute,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QuantFormat {
    None,
    Fp8,
    Q4,
    Q6,
    Q8,
    GgufQ4Km,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct QuantizationDescriptor {
    pub format: QuantFormat,
    pub group_size: Option<u32>,
    pub scale_dtype: Option<DType>,
    pub zero_point_dtype: Option<DType>,
    pub packing_layout: Option<String>,
    pub calibration: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct HardwareProfile {
    pub architecture: String,
    pub accelerator: String,
    pub total_memory_mb: u64,
    pub shared_memory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct MemoryProfile {
    pub available_memory_mb: u64,
    pub kv_cache_budget_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct NodeIdentity {
    pub node_id: String,
    pub protocol_version: String,
    pub transports: Vec<String>,
    pub trust_level: TrustLevel,
    pub auth_modes: Vec<AuthMode>,
    pub hardware: HardwareProfile,
    pub memory: MemoryProfile,
    pub health: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CacheDescriptor {
    pub version: String,
    pub dtype: DType,
    pub layout: TensorLayout,
    pub head_grouping: String,
    pub rope_state: PositionEncoding,
    pub sequence_indexing: String,
    pub eviction: String,
    pub compression: Option<String>,
    pub transferable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct KernelDescriptor {
    pub name: String,
    pub implementation_target: String,
    pub op_type: String,
    pub supported_dtypes: Vec<DType>,
    pub supported_layouts: Vec<TensorLayout>,
    pub supported_hardware: Vec<String>,
    pub deterministic: bool,
    pub memory_requirement_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_id: String,
    pub family: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub tokenizer_id: String,
    pub tokenizer_hash: String,
    pub vocabulary_hash: String,
    pub position_encoding: PositionEncoding,
    pub rope_scaling: Option<String>,
    pub attention_variant: AttentionVariant,
    pub hidden_size: u32,
    pub layer_count: u32,
    pub expert_count: Option<u32>,
    pub quantization: QuantizationDescriptor,
    pub tensor_layout: TensorLayout,
    pub artifact_hash: String,
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct BackendDescriptor {
    pub name: String,
    pub version: String,
    pub trust_level: TrustLevel,
    pub topology: BackendTopologyHints,
    pub memory_budget_mb: u64,
    pub attestation: Option<BackendAttestation>,
    pub execution_modes: Vec<ExecutionMode>,
    pub model_families: Vec<String>,
    pub quantization: Vec<QuantFormat>,
    pub dtypes: Vec<DType>,
    pub attention: Vec<AttentionVariant>,
    pub cache: Vec<CacheDescriptor>,
    pub tensor_layouts: Vec<TensorLayout>,
    pub parallelism: Vec<ExecutionMode>,
    pub streaming: bool,
    pub batching: bool,
    pub extensions: Vec<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct BackendTopologyHints {
    pub locality: String,
    pub zone: String,
    pub hop_count: u32,
    pub base_latency_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct BackendAttestation {
    pub format: String,
    pub signer: String,
    pub artifact_hash: String,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TensorEnvelope {
    pub tensor_id: String,
    pub op_context_id: String,
    pub session_id: Uuid,
    pub role: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub layout: TensorLayout,
    pub quantization: QuantizationDescriptor,
    pub compression: bool,
    pub checksum: String,
    pub sequence_number: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TensorFrame {
    pub envelope: TensorEnvelope,
    pub payload: Vec<u8>,
}

impl TensorFrame {
    pub fn checksum_hex(payload: &[u8]) -> String {
        hex::encode(Sha256::digest(payload))
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let checksum = Self::checksum_hex(&self.payload);
        if checksum != self.envelope.checksum {
            anyhow::bail!("tensor payload checksum mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CacheBlob {
    pub cache_id: String,
    pub session_id: Uuid,
    pub model_id: String,
    pub descriptor: CacheDescriptor,
    pub token_count: u32,
    pub checksum: String,
    pub payload: Vec<u8>,
}

impl CacheBlob {
    pub fn checksum_hex(payload: &[u8]) -> String {
        hex::encode(Sha256::digest(payload))
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let checksum = Self::checksum_hex(&self.payload);
        if checksum != self.checksum {
            anyhow::bail!("cache payload checksum mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    ChatCompletion,
    TextCompletion,
    Embedding,
    Ranking,
    TokenVerification,
    SpeculativeDecodeCoordination,
    ModelWarmup,
    CacheExportImport,
    TensorOpExecution,
    DistributedShardExecution,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Solo,
    RoutedServing,
    TensorParallel,
    PipelineParallel,
    ExpertParallel,
    Hybrid,
    ClientOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityOutcome {
    FullyCompatible,
    CompatibleWithConversion,
    CompatibleAsApiOnlyPeer,
    CompatibleOnlyForSoloServing,
    Incompatible,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct DeterminismPolicy {
    pub strict_correctness: bool,
    pub deterministic: bool,
    pub low_latency: bool,
    pub high_availability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct JobRequest {
    pub model_id: String,
    pub job_type: JobType,
    pub prompt: String,
    pub session_id: Option<Uuid>,
    pub preferred_backends: Vec<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub determinism: DeterminismPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CompatibilityRequest {
    pub model_id: String,
    pub job_type: JobType,
    pub peers: Vec<String>,
    pub desired_mode: Option<ExecutionMode>,
    pub determinism: DeterminismPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub outcome: CompatibilityOutcome,
    pub execution_mode: ExecutionMode,
    pub convertible: bool,
    pub reasons: Vec<String>,
    pub selected_peers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PlannedParticipant {
    pub backend: String,
    pub role: String,
    pub model_id: String,
    pub cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct JobPlan {
    pub session_id: Uuid,
    pub mode: ExecutionMode,
    pub compatibility: CompatibilityReport,
    pub participants: Vec<PlannedParticipant>,
    pub tensor_layout: TensorLayout,
    pub cache: Option<CacheDescriptor>,
    pub fallback_modes: Vec<ExecutionMode>,
    pub estimated_cost: u64,
    pub replan_generation: u32,
    pub partial_failure_tolerance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Proposed,
    Admitted,
    Planned,
    Assigned,
    Acknowledged,
    Executing,
    Streaming,
    Completed,
    Failed,
    Cancelled,
    Recovered,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct JobResult {
    pub text: String,
    pub tokens: u32,
    pub latency_ms: u64,
    pub backend: String,
    pub embeddings: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct JobRecord {
    pub job_id: Uuid,
    pub session_id: Uuid,
    pub status: JobStatus,
    pub plan: JobPlan,
    pub result: Option<JobResult>,
    pub error: Option<String>,
    pub attempts: Vec<ExecutionAttempt>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttemptStatus {
    Planned,
    Retrying,
    Succeeded,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ExecutionAttempt {
    pub attempt: u32,
    pub backend: String,
    pub mode: ExecutionMode,
    pub status: AttemptStatus,
    pub error: Option<String>,
    pub same_peer_retry: bool,
    pub replanned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub model_id: String,
    pub execution_mode: ExecutionMode,
    pub status: JobStatus,
    pub job_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct EventEnvelope {
    pub event_id: Uuid,
    pub category: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CapabilitySnapshot {
    pub node: NodeIdentity,
    pub backends: Vec<BackendDescriptor>,
    pub models: Vec<ModelIdentity>,
    pub kernels: Vec<KernelDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransportStatus {
    Healthy,
    Degraded,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TransportHealth {
    pub name: String,
    pub status: TransportStatus,
    pub latency_class: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannerDecisionKind {
    Compatibility,
    Plan,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PlannerDecisionRecord {
    pub decision_id: Uuid,
    pub kind: PlannerDecisionKind,
    pub model_id: String,
    pub execution_mode: ExecutionMode,
    pub outcome: CompatibilityOutcome,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct JobCancellation {
    pub job_id: Uuid,
    pub status: JobStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct JobStreamChunk {
    pub job_id: Uuid,
    pub session_id: Uuid,
    pub index: u32,
    pub token: String,
    pub done: bool,
    pub status: JobStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TensorRoundTripRequest {
    pub frame: TensorFrame,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TensorRoundTripResponse {
    pub byte_len: usize,
    pub checksum: String,
    pub sequence_number: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CacheRoundTripRequest {
    pub blob: CacheBlob,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CacheRoundTripResponse {
    pub byte_len: usize,
    pub checksum: String,
    pub token_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TopologyNode {
    pub node_id: String,
    pub health: HealthStatus,
    pub transports: Vec<String>,
    pub backend_names: Vec<String>,
    pub model_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TopologyView {
    pub protocol_version: String,
    pub compatibility_mode: String,
    pub nodes: Vec<TopologyNode>,
    pub active_sessions: usize,
    pub active_jobs: usize,
}
