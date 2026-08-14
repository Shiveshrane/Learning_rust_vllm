mod api;
mod worker;

use axum::{extract::State, http::StatusCode, routing::{get, post}, Router, Json};
use api::{CompletionRequest, CompletionResponse, Usage};
use worker::{Event, Job};
use tokio::sync::mpsc;



#[derive(Clone)]
struct AppState {
    jobs: mpsc::UnboundedSender<Job>,
}



async fn health()->&'static str{
    "OK"
}


async fn completions(State(st):State<AppState>,Json(req):Json<CompletionRequest>)->Result<Json<CompletionResponse>, (StatusCode, String)>{
    let (tx, mut rx) = mpsc::unbounded_channel();
    st.jobs.send(Job { req, tx })
        .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "worker gone".into()))?;

    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token(s) => text.push_str(&s),
            Event::Done { reason, prompt_tokens, completion_tokens } => {
                return Ok(Json(CompletionResponse {
                    text,
                    finish_reason: format!("{reason:?}"),
                    usage: Usage { prompt_tokens, completion_tokens },
                }));
            }
            Event::Error(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e)),
        }
    }
    Err((StatusCode::INTERNAL_SERVER_ERROR, "worker closed".into()))
}


#[tokio::main]
async fn main() ->anyhow::Result<()>{
    let state=AppState{
        jobs:worker::spawn(),
    };
    let app=Router::new()
    .route("/health", get(health))
    .route("/v1/completions", post(completions))
    .with_state(state);
    let listener=tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}
