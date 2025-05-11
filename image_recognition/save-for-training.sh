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

    # Wait for 10 seconds
    sleep 10
done

