use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use zeitgeist::{Runtime, api, config, peer};

#[derive(Parser)]
#[command(author, version, about = "Zeitgeist reference runtime")]
struct Cli {
    #[arg(long)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
    },
    ServePeer {
        #[arg(long, default_value = "127.0.0.1:9090")]
        bind: String,
    },
    ServePeerQuic {
        #[arg(long, default_value = "127.0.0.1:9443")]
        bind: String,
        #[arg(long)]
        cert: PathBuf,
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        client_ca_cert: Option<PathBuf>,
    },
    ServePeerUnix {
        #[arg(long, default_value = "/tmp/zeitgeist-peer.sock")]
        path: String,
    },
    Describe {
        #[arg(long)]
        pretty: bool,
    },
    Smoke {
        #[arg(long, default_value = "mesh prompt")]
        prompt: String,
        #[arg(long, default_value = "llama-3.2-3b-instruct")]
        model_id: String,
        #[arg(long, default_value = "mlx")]
        backend: String,
    },
    PeerPing {
        #[arg(long, default_value = "127.0.0.1:9090")]
        addr: String,
    },
    PeerCapabilities {
        #[arg(long, default_value = "127.0.0.1:9090")]
        addr: String,
    },
    PeerCapabilitiesQuic {
        #[arg(long, default_value = "127.0.0.1:9443")]
        addr: String,
        #[arg(long, default_value = "localhost")]
        server_name: String,
        #[arg(long)]
        ca_cert: PathBuf,
        #[arg(long)]
        client_cert: Option<PathBuf>,
        #[arg(long)]
        client_key: Option<PathBuf>,
    },
    PeerPingUnix {
        #[arg(long, default_value = "/tmp/zeitgeist-peer.sock")]
        path: String,
    },
    PeerPingQuic {
        #[arg(long, default_value = "127.0.0.1:9443")]
        addr: String,
        #[arg(long, default_value = "localhost")]
        server_name: String,
        #[arg(long)]
        ca_cert: PathBuf,
        #[arg(long)]
        client_cert: Option<PathBuf>,
        #[arg(long)]
        client_key: Option<PathBuf>,
    },
    PeerExecute {
        #[arg(long, default_value = "127.0.0.1:9090")]
        addr: String,
        #[arg(long, default_value = "remote execute")]
        prompt: String,
    },
    PeerExecuteQuic {
        #[arg(long, default_value = "127.0.0.1:9443")]
        addr: String,
        #[arg(long, default_value = "localhost")]
        server_name: String,
        #[arg(long)]
        ca_cert: PathBuf,
        #[arg(long)]
        client_cert: Option<PathBuf>,
        #[arg(long)]
        client_key: Option<PathBuf>,
        #[arg(long, default_value = "remote execute")]
        prompt: String,
    },
    MeshPeers,
    MeshSync,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,zeitgeist=debug")),
        )
        .init();

    let cli = Cli::parse();
    let config = config::load(cli.config.as_deref())?;
    let runtime = Runtime::new(
        config::node_identity(&config),
        config::backends(&config),
        config::models(),
        config::kernels(),
        config.auth_token.clone(),
        config::mesh(&config),
    );

    match cli.command {
        Command::Serve { bind } => api::serve(runtime, &bind).await,
        Command::ServePeer { bind } => peer::serve(runtime, &bind).await,
        Command::ServePeerQuic {
            bind,
            cert,
            key,
            client_ca_cert,
        } => peer::serve_quic(runtime, &bind, &cert, &key, client_ca_cert.as_deref()).await,
        Command::ServePeerUnix { path } => peer::serve_unix(runtime, &path).await,
        Command::Describe { pretty } => {
            let payload = runtime.capabilities();
            if pretty {
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                println!("{}", serde_json::to_string(&payload)?);
            }
            Ok(())
        }
        Command::Smoke {
            prompt,
            model_id,
            backend,
        } => {
            let record = runtime
                .submit_job(zeitgeist::types::JobRequest {
                    model_id,
                    job_type: zeitgeist::types::JobType::ChatCompletion,
                    prompt,
                    session_id: Some(uuid::Uuid::nil()),
                    preferred_backends: vec![backend],
                    max_tokens: 16,
                    temperature: 0.0,
                    determinism: zeitgeist::types::DeterminismPolicy {
                        strict_correctness: true,
                        deterministic: true,
                        low_latency: true,
                        high_availability: false,
                    },
                })
                .await?;
            let summary = serde_json::json!({
                "status": record.status,
                "mode": record.plan.mode,
                "backend": record.result.as_ref().map(|r| r.backend.clone()),
                "tokens": record.result.as_ref().map(|r| r.tokens),
                "text": record.result.as_ref().map(|r| r.text.clone()),
            });
            println!("{}", serde_json::to_string(&summary)?);
            Ok(())
        }
        Command::PeerPing { addr } => {
            let response = peer::send(
                &addr,
                &peer::PeerRequest::Handshake {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                    node_id: "zeitgeist-cli".into(),
                    signed_identity: Some(runtime.signed_identity_for("zeitgeist-cli")),
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::PeerCapabilities { addr } => {
            let response = peer::send(
                &addr,
                &peer::PeerRequest::Capabilities {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::PeerCapabilitiesQuic {
            addr,
            server_name,
            ca_cert,
            client_cert,
            client_key,
        } => {
            let response = peer::send_quic(
                &addr,
                &server_name,
                &ca_cert,
                client_cert.as_deref(),
                client_key.as_deref(),
                &peer::PeerRequest::Capabilities {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::PeerPingUnix { path } => {
            let response = peer::send_unix(
                &path,
                &peer::PeerRequest::Handshake {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                    node_id: "zeitgeist-cli".into(),
                    signed_identity: Some(runtime.signed_identity_for("zeitgeist-cli")),
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::PeerPingQuic {
            addr,
            server_name,
            ca_cert,
            client_cert,
            client_key,
        } => {
            let response = peer::send_quic(
                &addr,
                &server_name,
                &ca_cert,
                client_cert.as_deref(),
                client_key.as_deref(),
                &peer::PeerRequest::Handshake {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                    node_id: "zeitgeist-cli".into(),
                    signed_identity: Some(runtime.signed_identity_for("zeitgeist-cli")),
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::PeerExecute { addr, prompt } => {
            let response = peer::send(
                &addr,
                &peer::PeerRequest::ExecuteJob {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                    request: zeitgeist::types::JobRequest {
                        model_id: "llama-3.2-3b-instruct".into(),
                        job_type: zeitgeist::types::JobType::DistributedShardExecution,
                        prompt,
                        session_id: None,
                        preferred_backends: vec!["mlx".into()],
                        max_tokens: 16,
                        temperature: 0.0,
                        determinism: zeitgeist::types::DeterminismPolicy {
                            strict_correctness: true,
                            deterministic: true,
                            low_latency: true,
                            high_availability: true,
                        },
                    },
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::PeerExecuteQuic {
            addr,
            server_name,
            ca_cert,
            client_cert,
            client_key,
            prompt,
        } => {
            let response = peer::send_quic(
                &addr,
                &server_name,
                &ca_cert,
                client_cert.as_deref(),
                client_key.as_deref(),
                &peer::PeerRequest::ExecuteJob {
                    protocol_version: runtime.protocol_version().to_string(),
                    auth_token: runtime.auth_token().map(|token| token.to_string()),
                    request: zeitgeist::types::JobRequest {
                        model_id: "llama-3.2-3b-instruct".into(),
                        job_type: zeitgeist::types::JobType::DistributedShardExecution,
                        prompt,
                        session_id: None,
                        preferred_backends: vec!["mlx".into()],
                        max_tokens: 16,
                        temperature: 0.0,
                        determinism: zeitgeist::types::DeterminismPolicy {
                            strict_correctness: true,
                            deterministic: true,
                            low_latency: true,
                            high_availability: true,
                        },
                    },
                },
            )
            .await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::MeshPeers => {
            println!("{}", serde_json::to_string(&runtime.mesh_peers())?);
            Ok(())
        }
        Command::MeshSync => {
            println!(
                "{}",
                serde_json::to_string(&runtime.sync_mesh_once().await?)?
            );
            Ok(())
        }
    }
}
