mod api;
mod worker;

use axum::{extract::State, http::StatusCode, routing::{get, post}, Router, Json};
use api::{CompletionRequest, CompletionResponse, Usage};
use axum::response::{sse::Event as SseEvent, IntoResponse, Response, Sse};
use engine::sampling::Params;
use engine::scheduler::{Event, Job, Request};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::{Stream, StreamExt};
use std::convert::Infallible;


#[derive(Clone)]
struct AppState {
    jobs: mpsc::UnboundedSender<Job>,
}



async fn health()->&'static str{
    "OK"
}

fn stream_response(rx:mpsc::UnboundedReceiver<Event>)->Sse<impl Stream<Item = Result<SseEvent, Infallible>>>{
    let stream=UnboundedReceiverStream::new(rx).map(|ev|{
        let data=match ev{
            Event::Token(s)=>s,
            Event::Done{reason, prompt_tokens, completion_tokens}=>{
                format!("DONE: reason={:?}, prompt_tokens={}, completion_tokens={}", reason, prompt_tokens, completion_tokens)
            }
            Event::Error(e)=>format!("ERROR: {}", e),
        };
        Ok(SseEvent::default().data(data))
    });
    Sse::new(stream)
}

async fn collect_response(mut rx: mpsc::UnboundedReceiver<Event>)->Result<Json<CompletionResponse>,(StatusCode, String)>{
    let mut text=String::new();
    while let Some(ev)=rx.recv().await{
        match ev{
            Event::Token(s)=>text.push_str(&s),
            Event::Done{
                reason, prompt_tokens, completion_tokens
            }=>{
                return Ok(Json(CompletionResponse{
                    text,
                    finish_reason:format!("{reason:?}"),
                    usage:Usage{prompt_tokens, completion_tokens},
                }));
            }
            Event::Error(e)=>return Err((StatusCode::INTERNAL_SERVER_ERROR, e)),
        }
    }
    Err((StatusCode::INTERNAL_SERVER_ERROR, "worker closed".into()))
}



async fn completions(State(st):State<AppState>,Json(req):Json<CompletionRequest>)->Response{
    let (tx, mut rx) = mpsc::unbounded_channel();
    let stream=req.stream;

    // st.jobs.send(Job { req, tx })
    //     .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "worker gone".into()))?;

    // let mut text = String::new();
    // while let Some(ev) = rx.recv().await {
    //     match ev {
    //         Event::Token(s) => text.push_str(&s),
    //         Event::Done { reason, prompt_tokens, completion_tokens } => {
    //             return Ok(Json(CompletionResponse {
    //                 text,
    //                 finish_reason: format!("{reason:?}"),
    //                 usage: Usage { prompt_tokens, completion_tokens },
    //             }));
    //         }
    //         Event::Error(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e)),
    //     }
    // }
    // Err((StatusCode::INTERNAL_SERVER_ERROR, "worker closed".into()))

    // The layering boundary: serde/axum types stay in `server`, the engine
    // receives plain data. This is the only place the two vocabularies meet.
    let job = Job {
        req: Request {
            prompt: req.prompt,
            max_tokens: req.max_tokens,
            params: Params {
                temperature: req.temperature,
                top_k: req.top_k,
                top_p: req.top_p,
                min_prob: req.min_prob,
                repetition_penalty: req.repeat_penalty,
                seed: req.seed,
            },
            stop: req.stop,
        },
        tx,
    };
    if st.jobs.send(job).is_err(){
        return (StatusCode::SERVICE_UNAVAILABLE, "worker gone").into_response();
    }
    if stream{
        stream_response(rx).into_response()
    }else{
        collect_response(rx).await.into_response()
    }
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
