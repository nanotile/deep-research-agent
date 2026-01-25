#!/bin/bash
CONTAINER_NAME="research-agent"
echo "!!! EMERGENCY STOP !!!"
docker kill "$CONTAINER_NAME" 2>/dev/null
echo "Container killed."
