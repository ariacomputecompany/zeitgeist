use crate::{
    backend::SharedBackend,
    planner,
    types::*,
};
use anyhow::{anyhow, Result};
use std::{collections::BTreeMap, sync::Arc};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

pub const VERSION_POLICY: &str = "exact_only";

#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}

struct Inner {
    node: NodeIdentity,
    backends: Vec<SharedBackend>,
    models: Vec<ModelIdentity>,
    kernels: Vec<KernelDescriptor>,
    jobs: RwLock<BTreeMap<Uuid, JobRecord>>,
    event_log: RwLock<Vec<EventEnvelope>>,
    planner_decisions: RwLock<Vec<PlannerDecisionRecord>>,
    cancelled_jobs: RwLock<BTreeMap<Uuid, JobStatus>>,
    auth_token: Option<String>,
    events: broadcast::Sender<EventEnvelope>,
}

impl Runtime {
    pub fn new(
        node: NodeIdentity,
        backends: Vec<SharedBackend>,
        models: Vec<ModelIdentity>,
        kernels: Vec<KernelDescriptor>,
        auth_token: Option<String>,
    ) -> Self {
        let (events, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(Inner {
                node,
                backends,
                models,
                kernels,
                jobs: RwLock::new(BTreeMap::new()),
                event_log: RwLock::new(Vec::new()),
                planner_decisions: RwLock::new(Vec::new()),
                cancelled_jobs: RwLock::new(BTreeMap::new()),
                auth_token,
                events,
            }),
        }
    }

    pub fn capabilities(&self) -> CapabilitySnapshot {
        CapabilitySnapshot {
            node: self.inner.node.clone(),
            backends: self.inner.backends.iter().map(|backend| backend.descriptor()).collect(),
            models: self.inner.models.clone(),
            kernels: self.inner.kernels.clone(),
        }
    }

    pub fn protocol_version(&self) -> &str {
        &self.inner.node.protocol_version
    }

    pub fn auth_token(&self) -> Option<&str> {
        self.inner.auth_token.as_deref()
    }

    pub fn compatibility(&self, request: &CompatibilityRequest) -> CompatibilityReport {
        let report = planner::compatibility(
            request,
            &self.inner.backends.iter().map(|backend| backend.descriptor()).collect::<Vec<_>>(),
            &self.inner.models,
        );
        self.record_planner_decision(PlannerDecisionKind::Compatibility, &request.model_id, &report);
        report
    }

    pub fn plan(&self, request: &JobRequest) -> Result<JobPlan> {
        if !self.inner.models.iter().any(|model| model.model_id == request.model_id) {
            return Err(anyhow!("unknown model_id {}", request.model_id));
        }
        let plan = planner::plan(
            request,
            &self.inner.backends.iter().map(|backend| backend.descriptor()).collect::<Vec<_>>(),
            &self.inner.models,
        );
        self.record_planner_decision(PlannerDecisionKind::Plan, &request.model_id, &plan.compatibility);
        Ok(plan)
    }

    pub async fn submit_job(&self, request: JobRequest) -> Result<JobRecord> {
        let plan = self.plan(&request)?;
        if matches!(plan.compatibility.outcome, CompatibilityOutcome::Incompatible) {
            return Err(anyhow!("request is incompatible: {}", plan.compatibility.reasons.join("; ")));
        }
        let job_id = Uuid::new_v4();
        let mut record = JobRecord {
            job_id,
            session_id: plan.session_id,
            status: JobStatus::Proposed,
            plan: plan.clone(),
            result: None,
            error: None,
        };
        self.store(record.clone()).await;
        self.emit("job".into(), format!("job {} proposed", job_id));

        record.status = JobStatus::Executing;
        self.store(record.clone()).await;
        self.emit("job".into(), format!("job {} executing", job_id));

        match self.execute_with_recovery(&request, &record.plan, job_id).await {
            Ok((result, recovered)) => {
                record.status = if recovered {
                    JobStatus::Recovered
                } else {
                    JobStatus::Completed
                };
                record.result = Some(result);
                self.emit(
                    "job".into(),
                    format!(
                        "job {} {}",
                        job_id,
                        if recovered { "recovered" } else { "completed" }
                    ),
                );
            }
            Err(error) => {
                record.status = JobStatus::Failed;
                record.error = Some(error.to_string());
                self.emit("job".into(), format!("job {} failed", job_id));
            }
        }

        self.store(record.clone()).await;
        Ok(record)
    }

    pub async fn jobs(&self) -> Vec<JobRecord> {
        self.inner.jobs.read().await.values().cloned().collect()
    }

    pub async fn job(&self, job_id: Uuid) -> Option<JobRecord> {
        self.inner.jobs.read().await.get(&job_id).cloned()
    }

    pub async fn sessions(&self) -> Vec<SessionSummary> {
        let mut sessions = BTreeMap::<Uuid, SessionSummary>::new();
        for job in self.inner.jobs.read().await.values() {
            let entry = sessions.entry(job.session_id).or_insert_with(|| SessionSummary {
                session_id: job.session_id,
                model_id: job
                    .plan
                    .participants
                    .first()
                    .map(|participant| participant.model_id.clone())
                    .unwrap_or_else(|| "unknown".into()),
                execution_mode: job.plan.mode.clone(),
                status: job.status.clone(),
                job_ids: Vec::new(),
            });
            entry.status = job.status.clone();
            entry.job_ids.push(job.job_id);
        }
        sessions.into_values().collect()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.inner.events.subscribe()
    }

    pub async fn events(&self) -> Vec<EventEnvelope> {
        self.inner.event_log.read().await.clone()
    }

    pub async fn planner_decisions(&self) -> Vec<PlannerDecisionRecord> {
        self.inner.planner_decisions.read().await.clone()
    }

    pub fn transport_health(&self) -> Vec<TransportHealth> {
        vec![
            TransportHealth {
                name: "in_process".into(),
                status: TransportStatus::Healthy,
                latency_class: "ultra_low".into(),
                notes: "reference runtime local execution path".into(),
            },
            TransportHealth {
                name: "http_management".into(),
                status: TransportStatus::Healthy,
                latency_class: "low".into(),
                notes: "axum control plane surface".into(),
            },
            TransportHealth {
                name: "tcp_peer".into(),
                status: TransportStatus::Healthy,
                latency_class: "low".into(),
                notes: "peer negotiation, planning, and remote execution supported".into(),
            },
            TransportHealth {
                name: "quic_peer".into(),
                status: TransportStatus::Healthy,
                latency_class: "low".into(),
                notes: "peer negotiation, planning, and remote execution supported over QUIC".into(),
            },
            TransportHealth {
                name: "unix_peer".into(),
                status: TransportStatus::Healthy,
                latency_class: "ultra_low".into(),
                notes: "unix domain socket peer transport supported".into(),
            },
        ]
    }

    pub async fn topology(&self) -> TopologyView {
        let jobs = self.inner.jobs.read().await;
        let active_jobs = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Executing | JobStatus::Streaming | JobStatus::Recovered | JobStatus::Completed))
            .count();
        let active_sessions = jobs
            .values()
            .map(|job| job.session_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        TopologyView {
            protocol_version: self.protocol_version().to_string(),
            compatibility_mode: VERSION_POLICY.into(),
            nodes: vec![TopologyNode {
                node_id: self.inner.node.node_id.clone(),
                health: self.inner.node.health.clone(),
                transports: self.inner.node.transports.clone(),
                backend_names: self
                    .inner
                    .backends
                    .iter()
                    .map(|backend| backend.name().to_string())
                    .collect(),
                model_ids: self.inner.models.iter().map(|model| model.model_id.clone()).collect(),
            }],
            active_sessions,
            active_jobs,
        }
    }

    pub async fn cancel_job(&self, job_id: Uuid) -> Result<JobCancellation> {
        let mut jobs = self.inner.jobs.write().await;
        let record = jobs.get_mut(&job_id).ok_or_else(|| anyhow!("job not found"))?;
        record.status = JobStatus::Cancelled;
        record.error = Some("cancelled by operator".into());
        self.inner
            .cancelled_jobs
            .write()
            .await
            .insert(job_id, JobStatus::Cancelled);
        self.emit("job".into(), format!("job {} cancelled", job_id));
        Ok(JobCancellation {
            job_id,
            status: JobStatus::Cancelled,
        })
    }

    pub async fn submit_job_stream(
        &self,
        request: JobRequest,
    ) -> Result<(JobRecord, ReceiverStream<Result<JobStreamChunk>>)> {
        let plan = self.plan(&request)?;
        if matches!(plan.compatibility.outcome, CompatibilityOutcome::Incompatible) {
            return Err(anyhow!("request is incompatible: {}", plan.compatibility.reasons.join("; ")));
        }
        let job_id = Uuid::new_v4();
        let mut record = JobRecord {
            job_id,
            session_id: plan.session_id,
            status: JobStatus::Streaming,
            plan: plan.clone(),
            result: None,
            error: None,
        };
        self.store(record.clone()).await;
        self.emit("job".into(), format!("job {} streaming", job_id));

        let runtime = self.clone();
        let request_clone = request.clone();
        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            match runtime.execute_with_recovery(&request_clone, &plan, job_id).await {
                Ok((result, recovered)) => {
                    for (index, token) in result.text.split_whitespace().enumerate() {
                        if runtime
                            .inner
                            .cancelled_jobs
                            .read()
                            .await
                            .contains_key(&job_id)
                        {
                            let mut jobs = runtime.inner.jobs.write().await;
                            if let Some(record) = jobs.get_mut(&job_id) {
                                record.status = JobStatus::Cancelled;
                                record.error = Some("cancelled during streaming".into());
                            }
                            let _ = tx
                                .send(Ok(JobStreamChunk {
                                    job_id,
                                    session_id: plan.session_id,
                                    index: index as u32,
                                    token: String::new(),
                                    done: true,
                                    status: JobStatus::Cancelled,
                                }))
                                .await;
                            return;
                        }
                        let _ = tx
                            .send(Ok(JobStreamChunk {
                                job_id,
                                session_id: plan.session_id,
                                index: index as u32,
                                token: token.to_string(),
                                done: false,
                                status: JobStatus::Streaming,
                            }))
                            .await;
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    }
                    let _ = tx
                        .send(Ok(JobStreamChunk {
                            job_id,
                            session_id: plan.session_id,
                            index: result.tokens,
                            token: String::new(),
                            done: true,
                            status: JobStatus::Completed,
                        }))
                        .await;
                    let mut jobs = runtime.inner.jobs.write().await;
                    if let Some(stored) = jobs.get_mut(&job_id) {
                        stored.status = if recovered {
                            JobStatus::Recovered
                        } else {
                            JobStatus::Completed
                        };
                        stored.result = Some(result);
                    }
                }
                Err(error) => {
                    let mut jobs = runtime.inner.jobs.write().await;
                    if let Some(stored) = jobs.get_mut(&job_id) {
                        stored.status = JobStatus::Failed;
                        stored.error = Some(error.to_string());
                    }
                    let _ = tx.send(Err(error)).await;
                }
            }
        });

        record.status = JobStatus::Streaming;
        Ok((record, ReceiverStream::new(rx)))
    }

    async fn execute_with_recovery(
        &self,
        request: &JobRequest,
        plan: &JobPlan,
        job_id: Uuid,
    ) -> Result<(JobResult, bool)> {
        let candidates = self.execution_candidates(request, plan);
        let mut last_error = None;
        let mut recovered = false;

        for (index, backend) in candidates.into_iter().enumerate() {
            match backend.execute(request, plan).await {
                Ok(result) => {
                    return Ok((result, recovered));
                }
                Err(error) => {
                    let backend_name = backend.name().to_string();
                    self.emit(
                        "recovery".into(),
                        format!("job {} backend {} failed: {}", job_id, backend_name, error),
                    );
                    last_error = Some(error);
                    if index == 0 {
                        recovered = true;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("no execution candidates available")))
    }

    fn execution_candidates(&self, request: &JobRequest, plan: &JobPlan) -> Vec<SharedBackend> {
        let mut ordered = Vec::<SharedBackend>::new();
        let mut seen = std::collections::BTreeSet::<String>::new();

        for participant in &plan.participants {
            if let Some(backend) = self
                .inner
                .backends
                .iter()
                .find(|backend| backend.name() == participant.backend)
            {
                if seen.insert(backend.name().to_string()) {
                    ordered.push(backend.clone());
                }
            }
        }

        if plan.fallback_modes.contains(&ExecutionMode::Solo) || plan.fallback_modes.contains(&ExecutionMode::RoutedServing) {
            for backend in &self.inner.backends {
                let descriptor = backend.descriptor();
                let supports_model = descriptor
                    .model_families
                    .iter()
                    .any(|family| self.model_family(&request.model_id).is_some_and(|candidate| family == candidate));
                let supports_fallback = descriptor.execution_modes.contains(&ExecutionMode::Solo)
                    || descriptor.execution_modes.contains(&ExecutionMode::RoutedServing);
                if supports_model && supports_fallback && seen.insert(backend.name().to_string()) {
                    ordered.push(backend.clone());
                }
            }
        }

        ordered
    }

    fn model_family(&self, model_id: &str) -> Option<&str> {
        self.inner
            .models
            .iter()
            .find(|model| model.model_id == model_id)
            .map(|model| model.family.as_str())
    }

    async fn store(&self, record: JobRecord) {
        self.inner.jobs.write().await.insert(record.job_id, record);
    }

    fn emit(&self, category: String, detail: String) {
        let event = EventEnvelope {
            event_id: Uuid::new_v4(),
            category,
            detail,
        };
        let _ = self.inner.events.send(event.clone());
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let mut log = inner.event_log.write().await;
            log.push(event);
            if log.len() > 256 {
                let overflow = log.len() - 256;
                log.drain(0..overflow);
            }
        });
    }

    fn record_planner_decision(
        &self,
        kind: PlannerDecisionKind,
        model_id: &str,
        report: &CompatibilityReport,
    ) {
        let inner = self.inner.clone();
        let record = PlannerDecisionRecord {
            decision_id: Uuid::new_v4(),
            kind,
            model_id: model_id.to_string(),
            execution_mode: report.execution_mode.clone(),
            outcome: report.outcome.clone(),
            reasons: report.reasons.clone(),
        };
        tokio::spawn(async move {
            let mut decisions = inner.planner_decisions.write().await;
            decisions.push(record);
            if decisions.len() > 256 {
                let overflow = decisions.len() - 256;
                decisions.drain(0..overflow);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{default_backends, SyntheticBackend},
        config,
    };
    use std::{collections::BTreeMap, sync::Arc};

    #[tokio::test]
    async fn synthetic_job_completes() {
        let runtime = Runtime::new(
            config::node_identity(&Default::default()),
            default_backends(),
            config::models(),
            config::kernels(),
            None,
        );
        let record = runtime
            .submit_job(JobRequest {
                model_id: "llama-3.2-3b-instruct".into(),
                job_type: JobType::ChatCompletion,
                prompt: "hello mesh".into(),
                session_id: None,
                preferred_backends: vec!["mlx".into()],
                max_tokens: 32,
                temperature: 0.2,
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: false,
                },
            })
            .await
            .unwrap();
        assert_eq!(record.status, JobStatus::Completed);
    }

    #[tokio::test]
    async fn recovers_to_alternate_backend_when_primary_fails() {
        let mut failing = config::backends(&Default::default())[0].descriptor();
        failing.name = "mlx".into();
        failing.metadata = BTreeMap::from([("force_fail".into(), "always".into())]);
        let mut alternate = config::backends(&Default::default())[1].descriptor();
        alternate.name = "vllm".into();

        let runtime = Runtime::new(
            config::node_identity(&Default::default()),
            vec![
                Arc::new(SyntheticBackend::new(failing)),
                Arc::new(SyntheticBackend::new(alternate)),
            ],
            config::models(),
            config::kernels(),
            None,
        );

        let record = runtime
            .submit_job(JobRequest {
                model_id: "llama-3.2-3b-instruct".into(),
                job_type: JobType::ChatCompletion,
                prompt: "recover me".into(),
                session_id: None,
                preferred_backends: vec!["mlx".into(), "vllm".into()],
                max_tokens: 16,
                temperature: 0.0,
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: true,
                },
            })
            .await
            .unwrap();

        assert_eq!(record.status, JobStatus::Recovered);
        assert_eq!(record.result.unwrap().backend, "vllm");
    }

    #[tokio::test]
    async fn degrades_to_fallback_backend_even_when_only_primary_is_requested() {
        let mut failing = config::backends(&Default::default())[0].descriptor();
        failing.name = "mlx".into();
        failing.metadata = BTreeMap::from([("force_fail".into(), "always".into())]);
        let mut fallback = config::backends(&Default::default())[1].descriptor();
        fallback.name = "vllm".into();

        let runtime = Runtime::new(
            config::node_identity(&Default::default()),
            vec![
                Arc::new(SyntheticBackend::new(failing)),
                Arc::new(SyntheticBackend::new(fallback)),
            ],
            config::models(),
            config::kernels(),
            None,
        );

        let record = runtime
            .submit_job(JobRequest {
                model_id: "llama-3.2-3b-instruct".into(),
                job_type: JobType::ChatCompletion,
                prompt: "degrade me".into(),
                session_id: None,
                preferred_backends: vec!["mlx".into()],
                max_tokens: 16,
                temperature: 0.0,
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: true,
                },
            })
            .await
            .unwrap();

        assert_eq!(record.status, JobStatus::Recovered);
        assert_eq!(record.result.unwrap().backend, "vllm");
    }
}
