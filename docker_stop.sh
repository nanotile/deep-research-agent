#!/bin/bash
# Docker stop script for Deep Research Agent

CONTAINER_NAME="research-agent"
IMAGE_NAME="deep-research-agent"

echo "=== Deep Research Agent - Docker Stop ==="

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' not found"
    exit 0
fi

# Graceful stop with 30 second timeout
echo "Stopping container (30s timeout)..."
docker stop -t 30 $CONTAINER_NAME

if [ $? -eq 0 ]; then
    echo "Container stopped"
else
    echo "Warning: Stop command returned non-zero exit code"
fi

# Remove container
echo "Removing container..."
docker rm $CONTAINER_NAME 2>/dev/null

echo ""
echo "Container stopped and removed"

# Option to remove image
if [ "$1" = "--clean" ]; then
    echo ""
    echo "Removing image..."
    docker rmi $IMAGE_NAME
    echo "Image removed"
fi

echo ""
echo "To also remove the Docker image, run: ./docker_stop.sh --clean"
