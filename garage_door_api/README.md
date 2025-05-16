### Build for the Pi:
cross build --release --target=arm-unknown-linux-gnueabihf

### Make request:
curl -X POST http://pi.local:3000/toggle -H 'Authorization: Bearer $token$'
