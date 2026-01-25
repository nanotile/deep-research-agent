#!/bin/bash
# Docker start script for Deep Research Agent
# Uses on-failure:3 restart policy to prevent DoS from crash loops

CONTAINER_NAME="research-agent"
IMAGE_NAME="deep-research-agent"

echo "=== Deep Research Agent - Docker Start ==="

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Copy .env.example to .env and add your API keys"
    exit 1
fi

# Build image (uses cache if no changes)
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

# Stop and remove existing container if running
echo "Cleaning up existing container..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Create cache directory if needed
mkdir -p .cache

# Run container with restart policy (max 3 retries on crash)
echo "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart on-failure:3 \
    --env-file .env \
    -p 7860:7860 \
    -v "$(pwd)/.cache:/app/.cache" \
    $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start container"
    exit 1
fi

# Wait for container to initialize
sleep 2

# Show status
echo ""
echo "=== Container Status ==="
docker ps --filter name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Get external IP for access URL
EXTERNAL_IP=$(curl -s -m 5 http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null)

echo ""
echo "=== Access URLs ==="
echo "Local:    http://127.0.0.1:7860"
if [ -n "$EXTERNAL_IP" ]; then
    echo "External: http://$EXTERNAL_IP:7860"
fi

echo ""
echo "Use './docker_stop.sh' to stop the container"
echo "View logs with: docker logs -f $CONTAINER_NAME"
