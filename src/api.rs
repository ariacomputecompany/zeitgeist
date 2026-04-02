use crate::{codec, runtime::Runtime, types::*};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
    Json, Router,
};
use schemars::schema_for;
use std::{convert::Infallible, time::Duration};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use uuid::Uuid;

pub fn router(runtime: Runtime) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/version-policy", get(version_policy))
        .route("/v1/node", get(node))
        .route("/v1/backends", get(backends))
        .route("/v1/models", get(models))
        .route("/v1/kernels", get(kernels))
        .route("/v1/sessions", get(list_sessions))
        .route("/v1/transport-health", get(transport_health))
        .route("/v1/planner-decisions", get(planner_decisions))
        .route("/v1/topology", get(topology))
        .route("/v1/schema", get(schema))
        .route("/v1/compatibility", post(compatibility))
        .route("/v1/plan", post(plan))
        .route("/v1/jobs", post(submit_job).get(list_jobs))
        .route("/v1/jobs/stream", post(stream_job))
        .route("/v1/jobs/{job_id}", get(job))
        .route("/v1/jobs/{job_id}/cancel", post(cancel_job))
        .route("/v1/tensors/roundtrip", post(tensor_roundtrip))
        .route("/v1/cache/roundtrip", post(cache_roundtrip))
        .route("/v1/events", get(events))
        .route("/v1/events/stream", get(events_stream))
        .with_state(runtime)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

async fn version_policy(State(runtime): State<Runtime>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "protocol_version": runtime.protocol_version(),
        "compatibility_mode": crate::runtime::VERSION_POLICY,
        "backwards_compatibility": false
    }))
}

async fn node(State(runtime): State<Runtime>) -> Json<NodeIdentity> {
    Json(runtime.capabilities().node)
}

async fn backends(State(runtime): State<Runtime>) -> Json<Vec<BackendDescriptor>> {
    Json(runtime.capabilities().backends)
}

async fn models(State(runtime): State<Runtime>) -> Json<Vec<ModelIdentity>> {
    Json(runtime.capabilities().models)
}

async fn kernels(State(runtime): State<Runtime>) -> Json<Vec<KernelDescriptor>> {
    Json(runtime.capabilities().kernels)
}

async fn list_sessions(State(runtime): State<Runtime>) -> Json<Vec<SessionSummary>> {
    Json(runtime.sessions().await)
}

async fn transport_health(State(runtime): State<Runtime>) -> Json<Vec<TransportHealth>> {
    Json(runtime.transport_health())
}

async fn planner_decisions(State(runtime): State<Runtime>) -> Json<Vec<PlannerDecisionRecord>> {
    Json(runtime.planner_decisions().await)
}

async fn topology(State(runtime): State<Runtime>) -> Json<TopologyView> {
    Json(runtime.topology().await)
}

async fn schema() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "capability_snapshot": schema_for!(CapabilitySnapshot),
        "compatibility_request": schema_for!(CompatibilityRequest),
        "compatibility_report": schema_for!(CompatibilityReport),
        "job_request": schema_for!(JobRequest),
        "job_record": schema_for!(JobRecord),
        "session_summary": schema_for!(SessionSummary),
        "transport_health": schema_for!(TransportHealth),
        "planner_decision_record": schema_for!(PlannerDecisionRecord),
        "topology_view": schema_for!(TopologyView),
        "job_stream_chunk": schema_for!(JobStreamChunk),
        "tensor_roundtrip_request": schema_for!(TensorRoundTripRequest),
        "cache_roundtrip_request": schema_for!(CacheRoundTripRequest)
    }))
}

async fn compatibility(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Json(request): Json<CompatibilityRequest>,
) -> Result<Json<CompatibilityReport>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    Json(runtime.compatibility(&request))
        .pipe(Ok)
}

async fn plan(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Json(request): Json<JobRequest>,
) -> Result<Json<JobPlan>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    runtime.plan(&request).map(Json).map_err(internal_error)
}

async fn submit_job(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Json(request): Json<JobRequest>,
) -> Result<Json<JobRecord>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    runtime.submit_job(request).await.map(Json).map_err(internal_error)
}

async fn stream_job(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Json(request): Json<JobRequest>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    let (record, stream) = runtime.submit_job_stream(request).await.map_err(internal_error)?;
    let initial = tokio_stream::once(Ok(Event::default().event("job").data(
        serde_json::to_string(&serde_json::json!({
            "job_id": record.job_id,
            "session_id": record.session_id,
            "status": record.status
        }))
        .unwrap(),
    )));
    let chunks = stream.map(|item| match item {
        Ok(chunk) => Ok(Event::default().event("chunk").data(serde_json::to_string(&chunk).unwrap())),
        Err(error) => Ok(Event::default().event("error").data(serde_json::json!({"error": error.to_string()}).to_string())),
    });
    Ok(Sse::new(initial.chain(chunks)).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))))
}

async fn list_jobs(State(runtime): State<Runtime>) -> Json<Vec<JobRecord>> {
    Json(runtime.jobs().await)
}

async fn job(
    State(runtime): State<Runtime>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobRecord>, (StatusCode, Json<serde_json::Value>)> {
    runtime
        .job(job_id)
        .await
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "job not found" }))))
}

async fn cancel_job(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Path(job_id): Path<Uuid>,
) -> Result<Json<JobCancellation>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    runtime.cancel_job(job_id).await.map(Json).map_err(internal_error)
}

async fn tensor_roundtrip(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Json(request): Json<TensorRoundTripRequest>,
) -> Result<Json<TensorRoundTripResponse>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    let encoded = codec::encode_tensor_frame(&request.frame).map_err(internal_error)?;
    let decoded = codec::decode_tensor_frame(&encoded).map_err(internal_error)?;
    Ok(Json(TensorRoundTripResponse {
        byte_len: encoded.len(),
        checksum: decoded.envelope.checksum,
        sequence_number: decoded.envelope.sequence_number,
    }))
}

async fn cache_roundtrip(
    State(runtime): State<Runtime>,
    headers: HeaderMap,
    Json(request): Json<CacheRoundTripRequest>,
) -> Result<Json<CacheRoundTripResponse>, (StatusCode, Json<serde_json::Value>)> {
    require_protocol_version(&headers, &runtime)?;
    require_auth(&headers, &runtime)?;
    let encoded = codec::encode_cache_blob(&request.blob).map_err(internal_error)?;
    let decoded = codec::decode_cache_blob(&encoded).map_err(internal_error)?;
    Ok(Json(CacheRoundTripResponse {
        byte_len: encoded.len(),
        checksum: decoded.checksum,
        token_count: decoded.token_count,
    }))
}

async fn events(State(runtime): State<Runtime>) -> Json<Vec<EventEnvelope>> {
    Json(runtime.events().await)
}

async fn events_stream(
    State(runtime): State<Runtime>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(runtime.subscribe())
        .filter_map(|item| item.ok())
        .map(|event| {
            Ok(Event::default()
                .event(event.category.clone())
                .data(serde_json::to_string(&event).unwrap()))
        });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
}

fn internal_error(error: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": error.to_string()
        })),
    )
}

fn require_protocol_version(
    headers: &HeaderMap,
    runtime: &Runtime,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let Some(version) = headers.get("x-zeitgeist-protocol-version") else {
        return Err((
            StatusCode::PRECONDITION_REQUIRED,
            Json(serde_json::json!({
                "error": "missing x-zeitgeist-protocol-version header",
                "required_protocol_version": runtime.protocol_version(),
                "compatibility_mode": crate::runtime::VERSION_POLICY
            })),
        ));
    };
    let Ok(version) = version.to_str() else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid x-zeitgeist-protocol-version header"
            })),
        ));
    };
    if version != runtime.protocol_version() {
        return Err((
            StatusCode::PRECONDITION_FAILED,
            Json(serde_json::json!({
                "error": "protocol version mismatch",
                "received_protocol_version": version,
                "required_protocol_version": runtime.protocol_version(),
                "compatibility_mode": crate::runtime::VERSION_POLICY
            })),
        ));
    }
    Ok(())
}

fn require_auth(
    headers: &HeaderMap,
    runtime: &Runtime,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let Some(expected) = runtime.auth_token() else {
        return Ok(());
    };
    let Some(value) = headers.get(axum::http::header::AUTHORIZATION) else {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "missing authorization header",
                "auth_mode": "shared_token"
            })),
        ));
    };
    let Ok(value) = value.to_str() else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid authorization header"
            })),
        ));
    };
    if value != format!("Bearer {expected}") {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "shared token mismatch",
                "auth_mode": "shared_token"
            })),
        ));
    }
    Ok(())
}

pub async fn serve(runtime: Runtime, bind: &str) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, router(runtime)).await?;
    Ok(())
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::default_backends, config};

    fn test_runtime() -> Runtime {
        Runtime::new(
            config::node_identity(&Default::default()),
            default_backends(),
            config::models(),
            config::kernels(),
            None,
        )
    }

    #[tokio::test]
    async fn version_policy_is_exposed() {
        let runtime = test_runtime();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });

        let payload: serde_json::Value = reqwest::get(format!("http://{addr}/v1/version-policy"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(payload["compatibility_mode"], "exact_only");
        assert_eq!(payload["backwards_compatibility"], false);

        server.abort();
    }

    #[tokio::test]
    async fn submit_job_rejects_missing_protocol_header() {
        let runtime = test_runtime();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });

        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{addr}/v1/jobs"))
            .json(&serde_json::json!({
                "model_id": "llama-3.2-3b-instruct",
                "job_type": "chat_completion",
                "prompt": "hello",
                "session_id": null,
                "preferred_backends": ["mlx"],
                "max_tokens": 8,
                "temperature": 0.0,
                "determinism": {
                    "strict_correctness": true,
                    "deterministic": true,
                    "low_latency": true,
                    "high_availability": false
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PRECONDITION_REQUIRED);

        server.abort();
    }

    #[tokio::test]
    async fn submit_job_and_sessions_work_with_exact_protocol_version() {
        let runtime = test_runtime();
        let protocol = runtime.protocol_version().to_string();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });

        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{addr}/v1/jobs"))
            .header("x-zeitgeist-protocol-version", protocol)
            .json(&serde_json::json!({
                "model_id": "llama-3.2-3b-instruct",
                "job_type": "chat_completion",
                "prompt": "hello",
                "session_id": null,
                "preferred_backends": ["mlx"],
                "max_tokens": 8,
                "temperature": 0.0,
                "determinism": {
                    "strict_correctness": true,
                    "deterministic": true,
                    "low_latency": true,
                    "high_availability": false
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let sessions: Vec<SessionSummary> = reqwest::get(format!("http://{addr}/v1/sessions"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].model_id, "llama-3.2-3b-instruct");

        server.abort();
    }

    #[tokio::test]
    async fn tensor_roundtrip_works() {
        let runtime = test_runtime();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });
        let payload = TensorRoundTripRequest {
            frame: TensorFrame {
                envelope: TensorEnvelope {
                    tensor_id: "t1".into(),
                    op_context_id: "ctx".into(),
                    session_id: Uuid::nil(),
                    role: "activation".into(),
                    shape: vec![1, 2],
                    dtype: DType::F16,
                    layout: TensorLayout::RowMajorContiguous,
                    quantization: QuantizationDescriptor {
                        format: QuantFormat::None,
                        group_size: None,
                        scale_dtype: None,
                        zero_point_dtype: None,
                        packing_layout: None,
                        calibration: None,
                    },
                    compression: false,
                    checksum: TensorFrame::checksum_hex(&[1, 2, 3]),
                    sequence_number: 9,
                },
                payload: vec![1, 2, 3],
            },
        };
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{addr}/v1/tensors/roundtrip"))
            .header("x-zeitgeist-protocol-version", "0.1.0")
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        server.abort();
    }

    #[tokio::test]
    async fn shared_token_is_enforced_when_configured() {
        let runtime = Runtime::new(
            config::node_identity(&Default::default()),
            default_backends(),
            config::models(),
            config::kernels(),
            Some("secret-token".into()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });

        let client = reqwest::Client::new();
        let unauth = client
            .post(format!("http://{addr}/v1/jobs"))
            .header("x-zeitgeist-protocol-version", "0.1.0")
            .json(&serde_json::json!({
                "model_id": "llama-3.2-3b-instruct",
                "job_type": "chat_completion",
                "prompt": "hello",
                "session_id": null,
                "preferred_backends": ["mlx"],
                "max_tokens": 8,
                "temperature": 0.0,
                "determinism": {
                    "strict_correctness": true,
                    "deterministic": true,
                    "low_latency": true,
                    "high_availability": false
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

        let auth = client
            .post(format!("http://{addr}/v1/jobs"))
            .header("x-zeitgeist-protocol-version", "0.1.0")
            .header("authorization", "Bearer secret-token")
            .json(&serde_json::json!({
                "model_id": "llama-3.2-3b-instruct",
                "job_type": "chat_completion",
                "prompt": "hello",
                "session_id": null,
                "preferred_backends": ["mlx"],
                "max_tokens": 8,
                "temperature": 0.0,
                "determinism": {
                    "strict_correctness": true,
                    "deterministic": true,
                    "low_latency": true,
                    "high_availability": false
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(auth.status(), StatusCode::OK);

        server.abort();
    }

    #[tokio::test]
    async fn cache_roundtrip_works() {
        let runtime = test_runtime();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });
        let payload = CacheRoundTripRequest {
            blob: CacheBlob {
                cache_id: "cache1".into(),
                session_id: Uuid::nil(),
                model_id: "llama-3.2-3b-instruct".into(),
                descriptor: CacheDescriptor {
                    version: "zgc-1".into(),
                    dtype: DType::F16,
                    layout: TensorLayout::RowMajorContiguous,
                    head_grouping: "grouped-query".into(),
                    rope_state: PositionEncoding::Rope,
                    sequence_indexing: "absolute".into(),
                    eviction: "lru".into(),
                    compression: None,
                    transferable: true,
                },
                token_count: 4,
                checksum: CacheBlob::checksum_hex(&[5, 4, 3, 2]),
                payload: vec![5, 4, 3, 2],
            },
        };
        let client = reqwest::Client::new();
        let response = client
            .post(format!("http://{addr}/v1/cache/roundtrip"))
            .header("x-zeitgeist-protocol-version", "0.1.0")
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        server.abort();
    }

    #[tokio::test]
    async fn topology_endpoint_works() {
        let runtime = test_runtime();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(runtime)).await.unwrap();
        });
        let payload: TopologyView = reqwest::get(format!("http://{addr}/v1/topology"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(payload.protocol_version, "0.1.0");
        assert_eq!(payload.nodes.len(), 1);
        server.abort();
    }
}
