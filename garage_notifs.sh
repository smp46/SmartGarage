#!/bin/bash

STATUS_URL="http://0.0.0.0:5000/status"
EMAIL_TO="me@smp46.me"
EMAIL_FROM="Garage Watcher"
SUBJECT="Garage has been open for more than 10 minutes"
BODY_HTML="<p>Heads up! The garage has been open for more than 10 minutes.</p><p><a href=\"https://garage.smp46.me\">Go here to close it</a></p>"

# Fetch status from API
response=$(curl -s "$STATUS_URL")

# Parse JSON fields
status=$(echo "$response" | jq -r '.status')
last_changed=$(echo "$response" | jq -r '.last_changed')

# Only alert if it's open
if [[ "$status" == "Open" ]]; then
    # Convert last_changed to epoch
    last_changed_epoch=$(date -d "$last_changed" +%s)
    now_epoch=$(date +%s)

    # Calculate time difference in seconds
    elapsed=$((now_epoch - last_changed_epoch))

    # Debugging output
    echo "last_changed_epoch: $last_changed_epoch"
    echo "now_epoch: $now_epoch"
    echo "elapsed: $elapsed seconds"

    # 600 seconds = 10 minutes
    if (( elapsed > 600 )); then
        {
            echo "From: $EMAIL_FROM"
            echo "To: $EMAIL_TO"
            echo "Subject: $SUBJECT"
            echo "Content-Type: text/html"
            echo
            echo "$BODY_HTML"
        } | msmtp "$EMAIL_TO"
    fi
fi

