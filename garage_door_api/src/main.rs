use axum::{Router, extract::State, http::StatusCode, routing::post};
use axum_server::bind;
use dotenvy::dotenv;
use headers::authorization::Bearer;
use headers::{Authorization, HeaderMapExt};
use log::LevelFilter;
use rppal::gpio::{Gpio, OutputPin};
use simplelog::{Config, WriteLogger};
use std::env;
use std::{fs::File, net::SocketAddr, sync::Arc};
use tokio::{
    sync::Mutex,
    time::{Duration, sleep},
};

#[tokio::main]
async fn main() {
    dotenv().ok();
    let secret = env::var("SHARED_SECRET").expect("SHARED_SECRET must be set in .env");

    let toggle_secret = secret.clone();
    let test_secret = secret;

    // Init logger
    WriteLogger::init(
        LevelFilter::Info,
        Config::default(),
        File::create("./garage_door_api.log").expect("Could not create log file"),
    )
    .expect("Failed to initialize logger");

    // Init GPIO
    let gpio = Gpio::new().expect("Failed to access GPIO");
    let pin = gpio.get(17).expect("Failed to get GPIO 17").into_output();
    let shared_pin = Arc::new(Mutex::new(pin));

    // Build app
    let app = Router::new()
        .route(
            "/toggle",
            post(move |state, headers| toggle_handler(state, headers, toggle_secret.clone())),
        )
        .route(
            "/test",
            post(move |headers| test_handler(headers, test_secret.clone())),
        )
        .with_state(shared_pin);

    // Listen on all interfaces
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Listening on http://{}", addr);

    bind(addr).serve(app.into_make_service()).await.unwrap();
}

async fn test_handler(
    headers: axum::http::HeaderMap,
    secret: String,
) -> Result<&'static str, StatusCode> {
    let auth = headers
        .typed_get::<Authorization<Bearer>>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth.token() != secret {
        println!(
            "UNAUTHORIZED test attempt at {}",
            chrono::Local::now().to_rfc3339()
        );
        return Ok("UNAUTHORIZED, but request receieved");
    }

    println!("Test toggle at {}", chrono::Local::now().to_rfc3339());

    Ok("toggle test complete")
}

async fn toggle_handler(
    State(pin): State<Arc<Mutex<OutputPin>>>,
    headers: axum::http::HeaderMap,
    secret: String,
) -> Result<&'static str, StatusCode> {
    let auth = headers
        .typed_get::<Authorization<Bearer>>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth.token() != secret {
        println!(
            "UNAUTHORIZED access attempt at {}",
            chrono::Local::now().to_rfc3339()
        );
        return Err(StatusCode::UNAUTHORIZED);
    }

    println!(
        "Garage door toggled at {}",
        chrono::Local::now().to_rfc3339()
    );

    let pin = pin.clone();
    tokio::spawn(async move {
        {
            let mut pin = pin.lock().await;
            pin.set_high();
        } // Release mutex here
        sleep(Duration::from_secs(1)).await;
        {
            let mut pin = pin.lock().await;
            pin.set_low();
        }
    });

    Ok("Garage door toggled")
}
