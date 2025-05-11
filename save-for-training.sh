#!/bin/bash

# URL of the image to download
IMAGE_URL="http://admin:@192.168.69.66/Snapshot/1/1/RemoteImageCaptureV2?ImageFormat=jpg"

# Directory to save images
SAVE_DIR="/media-pool/garage/training_imgs"
mkdir -p "$SAVE_DIR"

while true; do
    # Current timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # Output filename
    FILENAME="gd_$TIMESTAMP.jpg"

    # Full path
    FILEPATH="$SAVE_DIR/$FILENAME"

    # Download image
    curl -s -o "$FILEPATH" "$IMAGE_URL"

    if [ $? -eq 0 ]; then
        echo "Saved: $FILEPATH"
    else
        echo "Failed to download image at $TIMESTAMP"
    fi

    # Get current day of the week (1 = Monday, ..., 7 = Sunday) and hour
    DAY=$(date +%u)
    HOUR=$(date +%H)

    # Determine sleep interval based on specified times
    if { [ "$DAY" -eq 1 ] && { [ "$HOUR" -eq 9 ] || [ "$HOUR" -eq 15 ]; }; } || \
       { [ "$DAY" -eq 2 ] && [ "$HOUR" -ge 17 ] && [ "$HOUR" -lt 19 ]; } || \
       { [ "$DAY" -eq 3 ] && [ "$HOUR" -ge 20 ] && [ "$HOUR" -lt 22 ]; } || \
       { [ "$DAY" -eq 6 ] && { [ "$HOUR" -eq 8 ] || [ "$HOUR" -eq 16 ]; }; }; then
        SLEEP_INTERVAL=1
    else
        SLEEP_INTERVAL=60
    fi

    # Wait for the determined interval
    sleep $SLEEP_INTERVAL
done
