#!/bin/bash
# Installation script that requires environment parameter

sudo cp /home/ec2-user/jina-embed-api/install/jina-embed-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jina-embed-api
sudo systemctl restart jina-embed-api

echo "Service installed "