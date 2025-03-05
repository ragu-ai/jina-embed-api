#!/bin/bash

# Update system packages
sudo yum update -y

# Install development tools
sudo yum groupinstall "Development Tools" -y
sudo yum install wget git cmake python3 python3-devel python3-pip -y

echo "Installing CUDA"
sudo $HOME/jina-embed-api/install/install-cuda.sh


# Show final instructions
echo "=========================================="
echo "Installation complete!"
echo "The embeddings API service is running on port 8000"
echo ""
echo "To test the API, run:"
echo "source ~/embeddings_env/bin/activate"
echo "python ~/embeddings_api/test_api.py"
echo ""
echo "Remember to open port 8000 in your security group if you want to access the API remotely"
echo "=========================================="