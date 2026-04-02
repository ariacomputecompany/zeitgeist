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

    let selected: Vec<_> = if request.peers.is_empty() {
        backends.to_vec()
    } else {
        backends
            .iter()
            .filter(|backend| request.peers.iter().any(|peer| peer == &backend.name))
            .cloned()
            .collect()
    };

    if selected.is_empty() {
        return CompatibilityReport {
            outcome: CompatibilityOutcome::Incompatible,
            execution_mode: ExecutionMode::ClientOnly,
            convertible: false,
            reasons: vec!["no backends matched the requested peers".into()],
            selected_peers: vec![],
        };
    }

    let mut reasons = Vec::new();
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

    let requested_mode = request.desired_mode.clone().unwrap_or_else(|| {
        if selected.len() == 1 {
            ExecutionMode::Solo
        } else {
            ExecutionMode::RoutedServing
        }
    });

    let outcome = if exact_model && exact_layout && exact_cache {
        if selected.len() > 1
            && selected
                .iter()
                .all(|backend| backend.parallelism.contains(&ExecutionMode::PipelineParallel))
        {
            reasons.push("all peers advertise the canonical model, tensor layout, and cache schema".into());
            CompatibilityOutcome::FullyCompatible
        } else {
            reasons.push("peers can cooperate, but only one backend advertises a distributed role".into());
            CompatibilityOutcome::CompatibleOnlyForSoloServing
        }
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
        CompatibilityOutcome::FullyCompatible => {
            if selected.len() > 1 {
                request
                    .desired_mode
                    .clone()
                    .unwrap_or(ExecutionMode::PipelineParallel)
            } else {
                ExecutionMode::Solo
            }
        }
        CompatibilityOutcome::CompatibleWithConversion => ExecutionMode::RoutedServing,
        CompatibilityOutcome::CompatibleAsApiOnlyPeer => ExecutionMode::RoutedServing,
        CompatibilityOutcome::CompatibleOnlyForSoloServing => ExecutionMode::Solo,
        CompatibilityOutcome::Incompatible => ExecutionMode::ClientOnly,
    };

    if requested_mode == ExecutionMode::TensorParallel && !exact_layout {
        reasons.push("tensor parallel requires exact tensor layout agreement".into());
    }

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
    let participants = compatibility
        .selected_peers
        .iter()
        .enumerate()
        .map(|(index, backend)| PlannedParticipant {
            backend: backend.clone(),
            role: if compatibility.execution_mode == ExecutionMode::PipelineParallel {
                format!("pipeline_stage_{index}")
            } else if index == 0 {
                "primary".into()
            } else {
                "fallback".into()
            },
            model_id: request.model_id.clone(),
        })
        .collect();
    let cache = backends
        .iter()
        .flat_map(|backend| backend.cache.iter())
        .find(|cache| cache.transferable && cache.layout == model.tensor_layout)
        .cloned();

    JobPlan {
        session_id,
        mode: compatibility.execution_mode.clone(),
        compatibility,
        participants,
        tensor_layout: model.tensor_layout.clone(),
        cache,
        fallback_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(name: &str, pipeline: bool, layout: TensorLayout) -> BackendDescriptor {
        BackendDescriptor {
            name: name.into(),
            version: "test".into(),
            execution_modes: vec![ExecutionMode::Solo, ExecutionMode::RoutedServing],
            model_families: vec!["llama".into()],
            quantization: vec![QuantFormat::None],
            dtypes: vec![DType::F16],
            attention: vec![AttentionVariant::Flash],
            cache: vec![CacheDescriptor {
                version: "zgc-1".into(),
                dtype: DType::F16,
                layout: layout.clone(),
                head_grouping: "grouped".into(),
                rope_state: PositionEncoding::Rope,
                sequence_indexing: "absolute".into(),
                eviction: "lru".into(),
                compression: None,
                transferable: true,
            }],
            tensor_layouts: vec![layout],
            parallelism: if pipeline {
                vec![ExecutionMode::PipelineParallel]
            } else {
                vec![ExecutionMode::Solo]
            },
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
    fn prefers_pipeline_when_exact_match_exists() {
        let report = compatibility(
            &CompatibilityRequest {
                model_id: "llama-3.2-3b".into(),
                job_type: JobType::ChatCompletion,
                peers: vec!["mlx".into(), "vllm".into()],
                desired_mode: Some(ExecutionMode::PipelineParallel),
                determinism: DeterminismPolicy {
                    strict_correctness: true,
                    deterministic: true,
                    low_latency: true,
                    high_availability: false,
                },
            },
            &[
                backend("mlx", true, TensorLayout::RowMajorContiguous),
                backend("vllm", true, TensorLayout::RowMajorContiguous),
            ],
            &[model()],
        );
        assert_eq!(report.outcome, CompatibilityOutcome::FullyCompatible);
        assert_eq!(report.execution_mode, ExecutionMode::PipelineParallel);
    }
}
