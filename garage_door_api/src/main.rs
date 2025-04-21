use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Router,
};
use headers::{Authorization, HeaderMapExt};
use headers::authorization::Bearer;
use rppal::gpio::{Gpio, OutputPin};
use std::{net::SocketAddr, sync::Arc, fs::File};
use tokio::{sync::Mutex, time::{sleep, Duration}};
use axum_server::bind;
use log::{info, LevelFilter};
use simplelog::{WriteLogger, Config};
use std::env;
use dotenvy::dotenv;

#[tokio::main]
async fn main() {
    dotenv().ok(); // Load from .env

    let secret = env::var("SHARED_SECRET").expect("SHARED_SECRET must be set in .env");

    // Init logger
    WriteLogger::init(
        LevelFilter::Info,
        Config::default(),
        File::create("./garage_door_api.log").expect("Could not create log file"),
    ).expect("Failed to initialize logger");

    // Init GPIO
    let gpio = Gpio::new().expect("Failed to access GPIO");
    let pin = gpio.get(17).expect("Failed to get GPIO 17").into_output();
    let shared_pin = Arc::new(Mutex::new(pin));

    // Build app
    let app = Router::new()
        .route("/toggle", post(move |state, headers| {
            toggle_handler(state, headers, secret.clone())
        }))
        .with_state(shared_pin);

    // Listen on all interfaces
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Listening on http://{}", addr);

    bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn toggle_handler(
    State(pin): State<Arc<Mutex<OutputPin>>>,
    headers: axum::http::HeaderMap,
    secret: String,
) -> Result<&'static str, StatusCode> {
    let auth = headers.typed_get::<Authorization<Bearer>>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth.token() != secret {
        return Err(StatusCode::UNAUTHORIZED);
    }

    info!("Garage door toggled at {}", chrono::Local::now().to_rfc3339());

    let pin = pin.clone();
    tokio::spawn(async move {
        let mut pin = pin.lock().await;
        pin.set_high();
        sleep(Duration::from_secs(1)).await;
        pin.set_low();
    });

    Ok("Garage door toggled")
}

