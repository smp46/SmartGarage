#!/bin/sh

# Read the image URL from an environment variable
IMAGE_URL="http://admin:@192.168.69.66/Snapshot/1/1/RemoteImageCaptureV2?ImageFormat=jpg"

# Output directory
OUTPUT_DIR="/mnt/ramdisk"
OUTPUT_FILE="${OUTPUT_DIR}/garage.jpg"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Download the image using curl
curl -o "${OUTPUT_FILE}" "${IMAGE_URL}"
