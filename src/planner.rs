use crate::types::*;

pub fn compatibility(
    request: &CompatibilityRequest,
    backends: &[BackendDescriptor],
    models: &[ModelIdentity],
) -> CompatibilityReport {
    let Some(model) = models.iter().find(|model| model.model_id == request.model_id) else {
        return CompatibilityReport {
            outcome: CompatibilityOutcome::Incompatible,
            execution_mode: ExecutionMode::ClientOnly,
            convertible: false,
            reasons: vec!["model_id is unknown".into()],
            selected_peers: vec![],
        };
    };

    let requested: Vec<_> = if request.peers.is_empty() {
        backends.to_vec()
    } else {
        backends
            .iter()
            .filter(|backend| request.peers.iter().any(|peer| peer == &backend.name))
            .cloned()
            .collect()
    };

    if requested.is_empty() {
        return CompatibilityReport {
            outcome: CompatibilityOutcome::Incompatible,
            execution_mode: ExecutionMode::ClientOnly,
            convertible: false,
            reasons: vec!["no backends matched the requested peers".into()],
            selected_peers: vec![],
        };
    }

    let mut reasons = Vec::new();
    let selected = trusted_backends(request, requested, &mut reasons);
    if selected.is_empty() {
        return CompatibilityReport {
            outcome: CompatibilityOutcome::Incompatible,
            execution_mode: ExecutionMode::ClientOnly,
            convertible: false,
            reasons,
            selected_peers: vec![],
        };
    }

    let exact_model = selected
        .iter()
        .all(|backend| backend.model_families.iter().any(|family| family == &model.family));
    let exact_layout = selected.iter().all(|backend| {
        backend
            .tensor_layouts
            .iter()
            .any(|layout| layout == &model.tensor_layout)
    });
    let exact_cache = selected.iter().all(|backend| {
        backend
            .cache
            .iter()
            .any(|cache| cache.transferable && cache.layout == model.tensor_layout)
    });

    let estimated_memory_mb = model_memory_mb(model);
    let max_backend_memory_mb = selected
        .iter()
        .map(|backend| backend.memory_budget_mb)
        .max()
        .unwrap_or_default();
    let aggregate_memory_mb: u64 = selected.iter().map(|backend| backend.memory_budget_mb).sum();
    let supports_tensor = selected
        .iter()
        .all(|backend| backend.parallelism.contains(&ExecutionMode::TensorParallel));
    let supports_pipeline = selected
        .iter()
        .all(|backend| backend.parallelism.contains(&ExecutionMode::PipelineParallel));
    let supports_expert = model.expert_count.is_some()
        && selected
            .iter()
            .all(|backend| backend.parallelism.contains(&ExecutionMode::ExpertParallel));

    let desired_mode = request.desired_mode.clone().unwrap_or_else(|| {
        choose_execution_mode(
            selected.len(),
            estimated_memory_mb,
            max_backend_memory_mb,
            aggregate_memory_mb,
            supports_tensor,
            supports_pipeline,
            supports_expert,
        )
    });

    if estimated_memory_mb > max_backend_memory_mb && selected.len() > 1 {
        reasons.push(format!(
            "memory pressure exceeds a single backend budget ({} MB > {} MB), repartitioning is required",
            estimated_memory_mb, max_backend_memory_mb
        ));
    }

    let outcome = if exact_model && exact_layout && exact_cache {
        reasons.push("selected peers advertise the canonical model, tensor layout, and transferable cache schema".into());
        CompatibilityOutcome::FullyCompatible
    } else if exact_model {
        reasons.push("model family matches, but tensor/cache conversion is required".into());
        CompatibilityOutcome::CompatibleWithConversion
    } else if selected.iter().all(|backend| backend.streaming) {
        reasons.push("peers can serve the request only as API-compatible routers".into());
        CompatibilityOutcome::CompatibleAsApiOnlyPeer
    } else {
        reasons.push("model family and execution requirements do not overlap".into());
        CompatibilityOutcome::Incompatible
    };

    let execution_mode = match outcome {
        CompatibilityOutcome::FullyCompatible => desired_mode,
        CompatibilityOutcome::CompatibleWithConversion => ExecutionMode::RoutedServing,
        CompatibilityOutcome::CompatibleAsApiOnlyPeer => ExecutionMode::RoutedServing,
        CompatibilityOutcome::CompatibleOnlyForSoloServing => ExecutionMode::Solo,
        CompatibilityOutcome::Incompatible => ExecutionMode::ClientOnly,
    };

    CompatibilityReport {
        outcome,
        execution_mode,
        convertible: exact_model && !exact_layout,
        reasons,
        selected_peers: selected.into_iter().map(|backend| backend.name).collect(),
    }
}

pub fn plan(
    request: &JobRequest,
    backends: &[BackendDescriptor],
    models: &[ModelIdentity],
) -> JobPlan {
    let compat_request = CompatibilityRequest {
        model_id: request.model_id.clone(),
        job_type: request.job_type.clone(),
        peers: request.preferred_backends.clone(),
        desired_mode: None,
        determinism: request.determinism.clone(),
    };
    let compatibility = compatibility(&compat_request, backends, models);
    let session_id = request.session_id.unwrap_or_else(uuid::Uuid::new_v4);
    let model = models
        .iter()
        .find(|model| model.model_id == request.model_id)
        .expect("model must exist after compatibility validation");
    let ordered = ordered_backends(
        compatibility
            .selected_peers
            .iter()
            .filter_map(|name| backends.iter().find(|backend| &backend.name == name).cloned())
            .collect(),
    );
    let participants: Vec<_> = ordered
        .iter()
        .enumerate()
        .map(|(index, backend)| PlannedParticipant {
            backend: backend.name.clone(),
            role: participant_role(&compatibility.execution_mode, index),
            model_id: request.model_id.clone(),
            cost: backend_cost(backend),
        })
        .collect();
    let cache = ordered
        .iter()
        .flat_map(|backend| backend.cache.iter())
        .find(|cache| cache.transferable && cache.layout == model.tensor_layout)
        .cloned();

    JobPlan {
        session_id,
        mode: compatibility.execution_mode.clone(),
        compatibility,
        estimated_cost: participants.iter().map(|participant| participant.cost).sum(),
        participants,
        tensor_layout: model.tensor_layout.clone(),
        cache,
        fallback_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing],
        replan_generation: 0,
        partial_failure_tolerance: true,
    }
}

fn trusted_backends(
    request: &CompatibilityRequest,
    candidates: Vec<BackendDescriptor>,
    reasons: &mut Vec<String>,
) -> Vec<BackendDescriptor> {
    let require_attestation = request.determinism.strict_correctness;
    let mut selected = Vec::new();
    for backend in candidates {
        let trusted = matches!(
            backend.trust_level,
            TrustLevel::TrustedExecutor | TrustLevel::TrustedCachePeer | TrustLevel::TrustedTensorPeer
        );
        if !trusted {
            reasons.push(format!(
                "backend {} excluded because trust level {:?} is not executable",
                backend.name, backend.trust_level
            ));
            continue;
        }
        if require_attestation
            && !backend
                .attestation
                .as_ref()
                .is_some_and(|attestation| attestation.verified)
        {
            reasons.push(format!(
                "backend {} excluded because verified attestation is required",
                backend.name
            ));
            continue;
        }
        selected.push(backend);
    }
    selected
}

fn ordered_backends(mut backends: Vec<BackendDescriptor>) -> Vec<BackendDescriptor> {
    backends.sort_by_key(backend_cost);
    backends
}

fn participant_role(mode: &ExecutionMode, index: usize) -> String {
    match mode {
        ExecutionMode::TensorParallel => format!("tensor_shard_{index}"),
        ExecutionMode::PipelineParallel => format!("pipeline_stage_{index}"),
        ExecutionMode::ExpertParallel => format!("expert_{index}"),
        ExecutionMode::Hybrid => format!("hybrid_worker_{index}"),
        _ if index == 0 => "primary".into(),
        _ => "fallback".into(),
    }
}

fn choose_execution_mode(
    backend_count: usize,
    estimated_memory_mb: u64,
    max_backend_memory_mb: u64,
    aggregate_memory_mb: u64,
    supports_tensor: bool,
    supports_pipeline: bool,
    supports_expert: bool,
) -> ExecutionMode {
    if backend_count == 0 {
        return ExecutionMode::ClientOnly;
    }
    if backend_count == 1 {
        return ExecutionMode::Solo;
    }
    let single_backend_pressure = estimated_memory_mb > max_backend_memory_mb;
    if supports_tensor && supports_expert && single_backend_pressure && estimated_memory_mb <= aggregate_memory_mb {
        return ExecutionMode::Hybrid;
    }
    if supports_expert {
        return ExecutionMode::ExpertParallel;
    }
    if supports_tensor && single_backend_pressure && estimated_memory_mb <= aggregate_memory_mb {
        return ExecutionMode::TensorParallel;
    }
    if supports_pipeline {
        return ExecutionMode::PipelineParallel;
    }
    ExecutionMode::RoutedServing
}

fn backend_cost(backend: &BackendDescriptor) -> u64 {
    let locality_cost = match backend.topology.locality.as_str() {
        "local" => 0,
        "regional" => 15,
        _ => 40,
    };
    locality_cost + backend.topology.base_latency_ms as u64 + (backend.topology.hop_count as u64 * 10)
}

fn model_memory_mb(model: &ModelIdentity) -> u64 {
    let bytes_per_param = match model.quantization.format {
        QuantFormat::None => 2_u64,
        QuantFormat::Fp8 => 1,
        _ => 1,
    };
    ((model.parameter_count * bytes_per_param) / (1024 * 1024)).max(512)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(name: &str, locality: &str, latency_ms: u32, memory_budget_mb: u64) -> BackendDescriptor {
        BackendDescriptor {
            name: name.into(),
            version: "test".into(),
            trust_level: TrustLevel::TrustedExecutor,
            topology: BackendTopologyHints {
                locality: locality.into(),
                zone: "test".into(),
                hop_count: if locality == "local" { 0 } else { 1 },
                base_latency_ms: latency_ms,
            },
            memory_budget_mb,
            attestation: Some(BackendAttestation {
                format: "artifact_hash".into(),
                signer: "test".into(),
                artifact_hash: format!("sha256:{name}"),
                verified: true,
            }),
            execution_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing],
            model_families: vec!["llama".into()],
            quantization: vec![QuantFormat::None],
            dtypes: vec![DType::F16],
            attention: vec![AttentionVariant::Flash],
            cache: vec![CacheDescriptor {
                version: "zgc-1".into(),
                dtype: DType::F16,
                layout: TensorLayout::RowMajorContiguous,
                head_grouping: "grouped".into(),
                rope_state: PositionEncoding::Rope,
                sequence_indexing: "absolute".into(),
                eviction: "lru".into(),
                compression: None,
                transferable: true,
            }],
            tensor_layouts: vec![TensorLayout::RowMajorContiguous],
            parallelism: vec![
                ExecutionMode::PipelineParallel,
                ExecutionMode::TensorParallel,
                ExecutionMode::ExpertParallel,
                ExecutionMode::Hybrid,
            ],
            streaming: true,
            batching: true,
            extensions: vec![],
            metadata: Default::default(),
        }
    }

    fn model() -> ModelIdentity {
        ModelIdentity {
            model_id: "llama-3.2-3b".into(),
            family: "llama".into(),
            architecture: "decoder".into(),
            parameter_count: 3_000_000_000,
            tokenizer_id: "llama".into(),
            tokenizer_hash: "tok".into(),
            vocabulary_hash: "vocab".into(),
            position_encoding: PositionEncoding::Rope,
            rope_scaling: None,
            attention_variant: AttentionVariant::Flash,
            hidden_size: 3072,
            layer_count: 28,
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
            artifact_hash: "hash".into(),
            revision: "r1".into(),
        }
    }

    #[test]
    fn chooses_tensor_parallel_under_memory_pressure() {
        let report = compatibility(
            &CompatibilityRequest {
                model_id: "llama-3.2-3b".into(),
                job_type: JobType::ChatCompletion,
                peers: vec!["mlx".into(), "vllm".into()],
                desired_mode: None,
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: false,
                },
            },
            &[backend("mlx", "local", 2, 2048), backend("vllm", "remote", 8, 4096)],
            &[model()],
        );
        assert_eq!(report.outcome, CompatibilityOutcome::FullyCompatible);
        assert_eq!(report.execution_mode, ExecutionMode::TensorParallel);
    }

    #[test]
    fn excludes_untrusted_backends() {
        let mut untrusted = backend("bad", "remote", 40, 4096);
        untrusted.trust_level = TrustLevel::UntrustedExternalClient;
        let report = compatibility(
            &CompatibilityRequest {
                model_id: "llama-3.2-3b".into(),
                job_type: JobType::ChatCompletion,
                peers: vec![],
                desired_mode: None,
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: false,
                },
            },
            &[backend("good", "local", 2, 4096), untrusted],
            &[model()],
        );
        assert_eq!(report.selected_peers, vec!["good"]);
        assert!(report.reasons.iter().any(|reason| reason.contains("excluded")));
    }

    #[test]
    fn orders_lower_cost_backends_first_in_plan() {
        let plan = plan(
            &JobRequest {
                model_id: "llama-3.2-3b".into(),
                job_type: JobType::DistributedShardExecution,
                prompt: "hello".into(),
                session_id: None,
                preferred_backends: vec!["remote".into(), "local".into()],
                max_tokens: 8,
                temperature: 0.0,
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: true,
                },
            },
            &[backend("remote", "remote", 40, 4096), backend("local", "local", 2, 4096)],
            &[model()],
        );
        assert_eq!(plan.participants[0].backend, "local");
    }
}
