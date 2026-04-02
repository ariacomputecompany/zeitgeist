**Zeitgeist**
A backend-neutral distributed inference protocol and runtime for heterogeneous local AI systems

## 1. Overview

Zeitgeist is an open protocol and modular runtime layer that allows different inference engines, hardware targets, and execution backends to participate in a shared distributed inference fabric.

Its purpose is to let systems built on:

- MLX
- vLLM
- llama.cpp
- TensorRT-LLM
- custom CUDA kernels
- custom Metal kernels
- custom local runtimes

interoperate through one common execution protocol.

Zeitgeist is not just a model-serving API. It is a full protocol for:

- capability discovery
- model identity
- backend negotiation
- execution planning
- tensor interchange
- KV-cache interchange
- distributed job orchestration
- failure handling
- transport framing
- modular backend and kernel extensibility

Zeitgeist is designed to be:

- open source
- protocol-first
- backend-neutral
- hardware-aware
- modular
- embeddable into our Mesh architecture
- usable by third parties independently of our product

Our project would integrate Zeitgeist natively as the shared execution layer, while still keeping our stronger control-plane, topology, and cooperative architecture.

---

## 2. Product Goals

### Primary Goals

- Allow heterogeneous backends to participate in a shared inference network.
- Standardize distributed inference interoperability across local runtimes.
- Make model serving portable across Apple Silicon, Linux GPU, consumer desktops, workstations, and custom hardware.
- Allow backend authors to plug into the protocol without rewriting the whole mesh stack.
- Allow runtime-specific optimizations without breaking network compatibility.
- Support both solo serving and distributed serving under one common protocol family.

### Secondary Goals

- Standardize operator and orchestration surfaces.
- Enable backend-aware routing and scheduling.
- Provide a stable foundation for open ecosystem adoption.
- Make backend and kernel innovation compatible with shared infrastructure.

### Non-Goals

- Replacing every backend’s internal scheduler or executor.
- Forcing all runtimes into one internal implementation.
- Hiding all backend differences completely.
- Supporting every model architecture in the first protocol release.
- Guaranteeing zero-copy interchange across all runtimes in all cases.

---

## 3. Product Positioning

Zeitgeist sits between:

- application-facing APIs
- and
- backend-specific execution engines

It is the interoperability plane.

### Above Zeitgeist

- OpenAI-compatible APIs
- app SDKs
- orchestration layers
- management consoles
- cluster/pool coordination
- billing, quota, and policy layers

### Below Zeitgeist

- MLX
- vLLM
- llama.cpp
- CUDA kernels
- Metal kernels
- CPU runtimes
- custom tensor engines
- custom quantization implementations

### In Our Stack

In our product, Zeitgeist becomes:

- the shared execution protocol
- the capability negotiation layer
- the backend and tensor interoperability contract

Our existing architecture remains stronger at:

- topology
- dispatch
- cooperative pooling
- recovery
- peer coordination
- data-plane design

---

## 4. Core Product Surfaces

Zeitgeist should have 4 major product surfaces:

### 4.1 Protocol Spec
Canonical open specification for wire format, capability model, execution semantics, and compatibility rules.

### 4.2 Runtime SDK
Reference implementation libraries for:

- Rust
- Python
- C/C++
- optional Swift for Apple ecosystem use

### 4.3 Reference Runtime
A modular runtime that implements the protocol and provides:

- transport
- capability registry
- execution planner
- backend adapters
- kernel registry
- debugging and observability tools

### 4.4 Certification/Test Suite
A compatibility suite that validates:

- backend conformance
- tensor correctness
- protocol compliance
- failure semantics
- distributed interoperability

---

## 5. Architecture

## 5.1 Layer Model

### Layer A: Application/API Layer
User-facing inference APIs and SDKs.

### Layer B: Orchestration Layer
Scheduling, job assignment, topology, policy, routing, and admission.

### Layer C: Zeitgeist Protocol Layer
Backend-neutral interoperability plane.

### Layer D: Backend Adapter Layer
Adapters for MLX, vLLM, llama.cpp, TensorRT-LLM, etc.

### Layer E: Kernel Layer
Backend-native kernels and custom pluggable kernel implementations.

### Layer F: Hardware Layer
CPU, Metal, CUDA, ROCm, Vulkan, etc.

---

## 5.2 Internal Runtime Components

A conformant Zeitgeist runtime should include:

- Capability Registry
- Model Registry
- Backend Manager
- Kernel Registry
- Execution Planner
- Session Manager
- KV Cache Manager
- Tensor Transport Manager
- Protocol Gateway
- Metrics/Tracing Layer
- Recovery Manager
- Compatibility Validator

---

## 6. Protocol Objects

Zeitgeist needs explicit protocol objects for all major runtime concepts.

## 6.1 Node Identity

A node advertises:

- node ID
- protocol version
- supported transports
- supported backends
- hardware profile
- memory profile
- trust/auth information
- runtime health

## 6.2 Backend Descriptor

A backend descriptor includes:

- backend name
- backend version
- execution mode support
- supported model families
- supported quantization types
- supported dtypes
- supported attention variants
- supported cache formats
- supported tensor formats
- supported parallelism modes
- streaming support
- batching support
- custom extension support

Example:

- `mlx`
- `vllm`
- `llama_cpp`
- `tensorrt_llm`
- `custom/<vendor>/<runtime>`

## 6.3 Model Identity Descriptor

This is critical.

A model must be identified by more than a filename.

It needs:

- canonical model family
- architecture ID
- parameter count
- tokenizer ID/hash
- vocabulary hash
- positional encoding type
- rope/scaling params
- attention variant
- hidden size
- layer count
- expert config if MoE
- quantization schema
- tensor layout schema
- model artifact hash
- revision/build metadata

Without this, cross-backend interoperability becomes chaos.

## 6.4 Kernel Descriptor

Kernels must also be visible as protocol-level capabilities.

A kernel descriptor should include:

- kernel name
- implementation target
- operation type
- supported dtypes
- supported tensor layouts
- supported hardware
- precision characteristics
- determinism characteristics
- memory requirements
- optional vendor extensions

This allows custom kernels to plug in cleanly.

---

## 7. Capability Negotiation

Zeitgeist must define capability negotiation as a first-class protocol phase.

## 7.1 Why It Matters

An MLX node and a vLLM node cannot safely cooperate unless they agree on:

- model identity
- tensor schema
- cache schema
- distributed execution mode
- transport framing
- precision compatibility

## 7.2 Negotiation Phases

### Phase 1: Protocol Handshake
Agree on protocol version, transport, auth mode, compression, framing.

### Phase 2: Runtime Capability Exchange
Exchange backend, hardware, tensor, cache, and execution capabilities.

### Phase 3: Model Capability Exchange
Exchange supported models and compatible model identities.

### Phase 4: Execution Compatibility Resolution
Determine whether nodes can participate in the same job and under what mode.

### Phase 5: Plan Finalization
Commit to execution mode, partition plan, tensor schema, and cache schema.

## 7.3 Compatibility Outcomes

Possible outcomes:

- fully compatible
- compatible with conversion
- compatible as API-only peer
- compatible only for solo serving
- incompatible

This distinction is essential.

---

## 8. Execution Modes

Zeitgeist should support multiple execution modes, even if not all are implemented at once.

## 8.1 Solo Execution
One node serves locally using its preferred backend.

## 8.2 Routed Serving
Different nodes serve requests independently; orchestration routes per request.

## 8.3 Tensor Parallel Execution
Nodes share layers/tensors in a synchronized distributed execution graph.

## 8.4 Pipeline Parallel Execution
Layers are partitioned across nodes.

## 8.5 Expert Parallel / MoE Execution
Experts distributed across nodes with shared routing semantics.

## 8.6 Hybrid Execution
Combination of tensor, pipeline, and expert parallelism.

## 8.7 Client-Only Participation
Node does not execute model kernels; it only consumes the mesh.

---

## 9. Tensor Interoperability Spec

This is the heart of the protocol.

Zeitgeist must define a canonical tensor interchange contract.

## 9.1 Tensor Envelope

Each tensor exchange should include:

- tensor ID
- op context ID
- model/job/session ID
- tensor role
- shape
- dtype
- layout
- quantization descriptor
- endian/framing
- compression flag
- checksum
- sequence number
- optional chunk metadata

## 9.2 Supported Tensor Layouts

The protocol should define canonical layout families, for example:

- contiguous row-major
- contiguous column-major
- backend-specific blocked layouts
- quantized tile layouts
- sparse layouts
- extension layouts

Backends may use internal layouts, but protocol interchange needs canonical forms or declared conversion paths.

## 9.3 Precision Rules

Must define:

- exact compatibility
- safe promotion
- safe demotion
- unsupported casts
- lossy conversion flags

## 9.4 Quantization Interchange

Quantization must not be “just backend-specific magic.”

Zeitgeist needs a quantization schema model for:

- quant format name
- grouping/block size
- scale format
- zero-point format
- packing layout
- tensor-specific exceptions
- calibration metadata where relevant

---

## 10. KV Cache Interoperability Spec

This is one of the hardest parts.

A true cross-backend distributed protocol requires an explicit cache contract.

## 10.1 Cache Descriptor

Must include:

- cache format version
- key/value dtype
- layout
- head grouping
- rope state assumptions
- sequence indexing semantics
- eviction semantics
- compression options
- backend-native extensions

## 10.2 Cache Interchange Modes

### Mode A: Native Shared Format
All participating backends use the same canonical cache format.

### Mode B: Protocol Conversion
Backends support import/export to canonical Zeitgeist cache format.

### Mode C: Non-Transferable
Backend can participate only when cache mobility is not required.

This allows graceful compatibility instead of pretending all caches are portable.

---

## 11. Session and Job Model

## 11.1 Job Types

Zeitgeist should define explicit job types:

- chat completion
- text completion
- embedding
- ranking
- token verification
- speculative decode coordination
- model warmup
- cache export/import
- tensor op execution
- distributed shard execution

## 11.2 Session Object

A session includes:

- session ID
- model identity
- tokenizer identity
- backend context
- execution mode
- cache state
- routing affinity
- consistency mode
- trace context

## 11.3 Job Lifecycle

A standard lifecycle should include:

- proposed
- admitted
- planned
- assigned
- acknowledged
- executing
- streaming
- completed
- failed
- cancelled
- recovered

---

## 12. Transport Layer

Zeitgeist should be transport-agnostic but define transport requirements.

## 12.1 Supported Transport Classes

- QUIC
- TCP
- Unix domain sockets
- shared memory transport
- in-process transport
- backend-specific high-speed plugins

## 12.2 Requirements

Transport must support:

- streaming
- framing
- multiplexing
- backpressure
- ordered and optionally unordered flows
- checksums
- cancellation
- retry semantics
- optional compression

## 12.3 Channels

At minimum, define logical channels for:

- control
- capability negotiation
- tensor transport
- cache transport
- job state
- metrics/events
- debug/trace

---

## 13. Authentication and Trust

Because Zeitgeist is open-source and likely widely embedded, auth must be modular.

## 13.1 Auth Modes

- none
- shared token
- mTLS
- signed node identity
- backend-signed attestation
- extension auth providers

## 13.2 Trust Levels

Define trust categories such as:

- trusted executor
- trusted cache peer
- trusted tensor peer
- API-only peer
- untrusted external client

Not every peer should be allowed into the same execution role.

---

## 14. Modular Backend Architecture

This is essential.

## 14.1 Backend Plugin Contract

A backend must implement something like:

- capability reporting
- model loading
- tokenizer binding
- inference execution
- tensor export/import
- cache export/import
- session state hooks
- metrics hooks
- error mapping
- shutdown/recovery behavior

## 14.2 Backend Classes

- local serving backend
- distributed execution backend
- cache-capable backend
- tensor-export backend
- planner-only backend
- client-only backend

A backend does not need to support all classes.

## 14.3 Examples

### MLX Adapter
- Apple Silicon optimized
- likely excellent solo/local serving backend
- maybe limited initially in distributed tensor role depending on cache/tensor support maturity

### vLLM Adapter
- Linux GPU serving
- high-throughput serving
- strong API backend
- likely excellent routed/solo backend
- distributed execution role depends on protocol adapter completeness

### Custom Backend
Vendors or researchers can plug in niche runtimes while still participating in the ecosystem.

---

## 15. Modular Kernel Architecture

Zeitgeist should support custom kernels as protocol-visible acceleration modules.

## 15.1 Why

Users may want:

- custom attention kernels
- custom quant matmul kernels
- vendor-specific fused ops
- optimized MoE routing kernels
- experimental research kernels

## 15.2 Kernel API Requirements

Each kernel plugin should declare:

- op types
- supported tensor schemas
- supported dtypes
- target hardware
- determinism guarantees
- memory requirements
- compatibility with protocol layouts
- fallback path if unavailable

## 15.3 Kernel Resolution

At execution planning time, Zeitgeist should choose:

- required kernel
- preferred kernel
- acceptable fallback kernel
- backend-native fallback
- protocol-level incompatibility if none available

---

## 16. Planning and Scheduling

Zeitgeist needs a planning layer, even if orchestration is external.

## 16.1 Planner Responsibilities

- determine compatibility set
- choose execution mode
- assign partitions
- choose transport schemas
- choose cache schema
- choose backend roles
- validate determinism/precision policy
- define fallback paths

## 16.2 Planning Constraints

Planner must consider:

- backend support
- hardware capability
- transport costs
- topology
- latency
- memory
- quantization compatibility
- kernel availability
- cache portability
- trust level

---

## 17. Failure Handling and Recovery

A real protocol needs explicit failure semantics.

## 17.1 Failure Classes

- transport failure
- backend crash
- kernel failure
- incompatible tensor schema
- cache import failure
- peer timeout
- partition invalidation
- model mismatch
- unsupported conversion
- resource exhaustion

## 17.2 Recovery Modes

- retry same peer
- retry alternate peer
- degrade to solo
- degrade to routed serving
- replan partition
- rehydrate cache
- abort session
- surface partial failure with reason

## 17.3 Recovery Policy Model

Recovery should be configurable by:

- strict correctness mode
- best-effort mode
- low-latency mode
- deterministic mode
- high-availability mode

---

## 18. Observability and Operator Surfaces

Zeitgeist should have first-class operator semantics, not leave everything to logs.

## 18.1 Required Introspection APIs

- node capabilities
- backend inventory
- model inventory
- current sessions
- current jobs
- transport health
- tensor throughput
- cache mobility status
- planner decisions
- fallback reasons
- failure events

## 18.2 Event Stream

Must support a live event stream for:

- joins/leaves
- planner decisions
- backend changes
- model load/unload
- kernel resolution
- failure and recovery events
- transport degradation
- cache portability failures

## 18.3 UI Implications

A management console should be able to render:

- topology
- backend types per node
- model compatibility state
- current execution plan
- active jobs
- fallback/recovery status

---

## 19. API Surfaces

Zeitgeist itself should expose both machine-oriented and human/operator-oriented APIs.

## 19.1 Protocol API
Strict protocol endpoints for peers and orchestrators.

## 19.2 Management API
JSON and SSE surfaces for operator tools and dashboards.

## 19.3 Compatibility API
Allows tooling to ask:

- can these nodes cooperate?
- under what execution mode?
- what conversion penalties apply?
- what kernel or backend limitations block execution?

This is extremely valuable.

---

## 20. Ecosystem and Open Source Strategy

Since Zeitgeist will be open-sourced as a protocol, ecosystem design matters.

## 20.1 Deliverables

- open protocol spec
- reference Rust SDK
- conformance suite
- backend adapter examples
- kernel plugin examples
- compatibility test harness
- sample management UI schema

## 20.2 Third-Party Extension Model

Third parties should be able to add:

- custom backends
- custom kernels
- custom transports
- auth extensions
- model-family extensions
- quantization extensions

## 20.3 Stability Model

Need explicit versioning for:

- protocol version
- model descriptor version
- tensor schema version
- cache schema version
- backend plugin API version
- kernel plugin API version

---

## 21. Full Functional Requirements

To make Zeitgeist fully functional, we need all of the following categories.

## 21.1 Protocol Foundation
- wire format
- versioning
- framing
- auth hooks
- transport abstractions

## 21.2 Compatibility System
- node capabilities
- backend descriptors
- model identity descriptors
- kernel descriptors
- cache descriptors
- compatibility negotiation

## 21.3 Execution System
- job lifecycle
- session lifecycle
- planning
- assignment
- tensor exchange
- cache exchange
- streaming output
- cancellation

## 21.4 Modularity System
- backend plugin API
- kernel plugin API
- transport plugin API
- extension registry

## 21.5 Reliability System
- health model
- failure semantics
- recovery semantics
- replayability
- observability

## 21.6 Ecosystem System
- SDKs
- docs
- conformance tests
- certification suite
- examples

---

## 22. Narrow V1 Section

You asked for the full spec, not a narrow V1, but a narrow V1 is still important.

A realistic narrow V1 should include:

- one protocol version
- one canonical tensor schema family
- one canonical cache schema
- one model family only
- MLX backend adapter
- one Linux backend adapter, likely vLLM or llama.cpp depending practicality
- solo mode
- routed serving mode
- one mixed-backend distributed execution mode
- strict capability negotiation
- no silent compatibility assumptions
- strong conformance suite

A narrow V1 should explicitly exclude:

- all model families
- all quantization formats
- all distributed modes
- arbitrary cache mobility
- arbitrary backend/kernel extensions in the first release

That narrowness is what makes the full long-term vision reachable.

---

## 23. Product Risks

## 23.1 Scope Explosion
Trying to standardize too much at once.

## 23.2 Fake Interoperability
Claiming mixed-backend compatibility without hard canonical contracts.

## 23.3 Backend Drift
Each backend evolves independently and breaks assumptions.

## 23.4 Debugging Complexity
Cross-backend distributed failures can be extremely hard to debug.

## 23.5 Performance Tax
Canonical interchange can introduce conversion or transport overhead.

---

## 24. Strategic Value

If Zeitgeist works, it becomes a serious moat.

Why:

- backend-neutral execution is strategically powerful
- the ecosystem value compounds
- it prevents lock-in to one runtime
- it fits the democratized AI thesis
- it gives us the strongest long-term technical narrative

This is not a small feature.
This is a platform.

---

## 25. Final Recommendation

Build Zeitgeist.

But treat it as a first-class product and protocol initiative with:

- its own spec
- its own modular runtime
- its own conformance suite
- its own extension system
- its own versioning and compatibility story

And most importantly:

Do not define it as “backend adapters for our product.”

Define it as:

- an open inference interoperability protocol
- which our product uses natively
- and which others can adopt independently

That gives us the right architecture, the right open-source posture, and the right long-term leverage.

If you want, next I can turn this into a concrete artifact in the repo as `ZEITGEIST_SPEC.md`, with:
- normative requirements
- recommended interfaces
- message schemas
- plugin traits
- protocol phases
- implementation roadmap

---

## 26. Implementation Checklist

This checklist is the current source-of-truth status against this spec.

### 26.1 Product Surfaces

- ✅ Protocol-oriented Rust reference runtime exists.
- ✅ Canonical standalone normative protocol document exists in-repo.
- ⬜ Python SDK.
- ⬜ C/C++ SDK.
- ⬜ Swift SDK.
- ✅ Conformance-oriented test suite scaffold exists with deterministic and host-backed verification.

### 26.2 Architecture and Runtime Components

- ✅ Capability Registry.
- ✅ Model Registry.
- ✅ Backend Manager.
- ✅ Kernel Registry.
- ✅ Execution Planner.
- ✅ Session Manager.
- ⬜ KV Cache Manager with real cache import/export implementation.
- ⬜ Tensor Transport Manager with real binary interchange paths.
- ✅ Protocol Gateway.
- ✅ Metrics/Tracing layer via runtime events plus Fozzy artifacts/profile outputs.
- ✅ Recovery Manager with active failover execution.
- ✅ Recovery Manager with active replan execution.
- ⬜ Compatibility Validator as a dedicated external certification tool.

### 26.3 Protocol Objects

- ✅ Node identity object.
- ✅ Backend descriptor.
- ✅ Model identity descriptor.
- ✅ Kernel descriptor.
- ✅ Cache descriptor.
- ✅ Tensor envelope type.
- ✅ Job/session/result/event objects.

### 26.4 Capability Negotiation

- ✅ Compatibility phases modeled through compatibility and planning APIs.
- ✅ Compatibility outcomes encoded explicitly.
- ✅ Exact-only protocol version policy enforced on negotiation and job submission endpoints.
- ✅ Authenticated handshake over a real TCP transport session.

### 26.5 Execution Modes

- ✅ Solo execution planning.
- ✅ Routed serving planning.
- ✅ Pipeline-parallel planning surface.
- ✅ Tensor-parallel runtime execution.
- ✅ Expert-parallel runtime execution.
- ✅ Hybrid runtime execution.
- ✅ Client-only incompatibility outcome.

### 26.6 Tensor Interoperability

- ✅ Canonical tensor envelope schema defined in code.
- ✅ Canonical layout and dtype enums defined in code.
- ✅ Quantization descriptor schema defined in code.
- ✅ Canonical binary tensor framing with checksum validation.
- ✅ Tensor roundtrip verification API.
- ⬜ Binary tensor transport conversion across real heterogeneous backends.
- ⬜ Checksum-verified chunked tensor streaming.

### 26.7 KV Cache Interoperability

- ✅ Cache descriptor schema defined in code.
- ✅ Transferability declared explicitly.
- ✅ Canonical cache serialization format.
- ✅ Cache roundtrip verification path.
- ⬜ Cache export/import execution paths across real backends.
- ⬜ Cache conversion engine across real backends.

### 26.8 Session and Job Model

- ✅ Job types defined.
- ✅ Job lifecycle states defined.
- ✅ Session summaries exposed in the management API.
- ✅ Deterministic synthetic job execution path.
- ✅ Streaming token output API.
- ✅ Cancellation API for in-flight jobs.
- ✅ Recovery flow.

### 26.9 Transport Layer

- ✅ In-process and HTTP management transport surfaces exist.
- ✅ Transport health API.
- ✅ QUIC transport.
- ✅ TCP peer transport for handshake/capability/plan exchange.
- ✅ Unix domain socket transport.
- ⬜ Shared memory transport.
- ✅ Framed tensor/cache logical channels over peer transport.

### 26.10 Authentication and Trust

- ✅ Auth mode and trust level types exist.
- ✅ Enforced shared-token auth.
- ⬜ mTLS.
- ⬜ Signed node identity.
- ⬜ Backend attestation.

### 26.11 Modular Backend and Kernel Architecture

- ✅ Backend adapter trait.
- ✅ Synthetic backend for deterministic certification.
- ✅ MLX-shaped adapter.
- ✅ vLLM OpenAI-compatible proxy-shaped adapter.
- ✅ Kernel descriptor surface.
- ⬜ Dynamic plugin loading ABI.
- ⬜ Real llama.cpp adapter.
- ⬜ Real TensorRT-LLM adapter.

### 26.12 Planning and Scheduling

- ✅ Compatibility-aware planning.
- ✅ Fallback modes included in plans.
- ✅ Latency-aware deterministic policy input modeled.
- ✅ Topology-aware cost model.
- ✅ Memory-pressure-aware repartitioning.
- ✅ Trust-aware peer exclusion in live scheduling.

### 26.13 Failure Handling and Recovery

- ✅ Failure classes are modeled as explicit incompatible/error outcomes.
- ✅ Failed jobs are recorded with reasons.
- ✅ Operator cancellation path.
- ✅ Retry alternate peer.
- ✅ Degrade to solo/routed fallback automatically.
- ✅ Retry same peer.
- ⬜ Rehydrate cache.
- ✅ Partial failure surfacing for distributed execution.

### 26.14 Observability and Operator Surfaces

- ✅ Node/backends/models/kernels/jobs/sessions APIs.
- ✅ Event log and SSE event stream.
- ✅ Fozzy artifact, memory, replay, CI, and profiling outputs.
- ✅ Transport health API.
- ✅ Planner decision audit API with persistent histories.
- ✅ Topology UI schema/API.

### 26.15 API Surfaces

- ✅ Protocol/management HTTP API exists.
- ✅ Compatibility API exists.
- ✅ Schema discovery endpoint exists.
- ✅ Tensor/cache roundtrip verification APIs exist.
- ✅ Peer-wire TCP endpoints for node-to-node negotiation/planning.
- ✅ Peer-wire TCP remote execution endpoint.
- ⬜ gRPC transport.

### 26.16 Ecosystem and Open Source Strategy

- ✅ Concise technical README centered on functionality.
- ✅ Open-source-friendly backend-neutral positioning in docs.
- ✅ Version policy explicitly states no backwards compatibility.
- ⬜ Separate conformance certification packaging.
- ⬜ Example third-party extension packages.

### 26.17 Verification

- ✅ `cargo test` passes.
- ✅ Quilt Linux `cargo test` pass.
- ✅ Fozzy deterministic doctor pass.
- ✅ Fozzy deterministic `test` pass.
- ✅ Fozzy deterministic `run` with recorded trace.
- ✅ Fozzy `trace verify` pass.
- ✅ Fozzy `replay` pass.
- ✅ Fozzy `ci` pass.
- ✅ Fozzy QUIC deterministic doctor/test/run/trace verify/replay/ci pass.
- ✅ Fozzy `explore` pass.
- ✅ Fozzy `fuzz` pass.
- ✅ Fozzy memory inspection commands exercised.
- ✅ Fozzy profile commands exercised.
- ✅ Fozzy corpus commands exercised.
- ✅ Host-backed live HTTP validation against a running runtime.
- ✅ Live QUIC peer handshake/capabilities/remote execution validation against a running runtime.
- ✅ Quilt Linux vLLM installation/import verification.
- ✅ Live MLX end-to-end inference through the real adapter.
- ✅ Live vLLM end-to-end inference through a real remote server.
