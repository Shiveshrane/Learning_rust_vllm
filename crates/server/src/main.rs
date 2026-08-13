use axum::{routing::get, Router};

async fn health()->&'static str{
    "OK"
}

#[tokio::main]
async fn main() ->anyhow::Result<()>{
    let app=Router::new().route("/health", get(health));
    let listener=tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}