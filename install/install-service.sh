#!/bin/bash
# Installation script that requires environment parameter

# Check if environment parameter is provided
if [ -z "$1" ]; then
  echo "Error: Environment parameter is required"
  echo "Usage: $0 <environment>"
  echo "Example: $0 production"
  exit 1
fi

# Set environment from parameter
DEPLOY_ENV="$1"
echo "Installing service for environment: $DEPLOY_ENV"

# Create the service file with environment variable
cat > /tmp/jina-embed-api.service << EOF
[Unit]
Description=Jina Embeddings API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/jina-embed-api
Environment="ENVIRONMENT=$DEPLOY_ENV"
ExecStart=/home/ec2-user/.local/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --log-level info --limit-concurrency 24 --timeout-keep-alive 120 --backlog 1024
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install and start the service
sudo mv /tmp/jina-embed-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jina-embed-api
sudo systemctl restart jina-embed-api

echo "Service installed with $DEPLOY_ENV environment"