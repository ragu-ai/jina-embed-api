# Jina Embeddings API

A high-performance API server for generating embeddings using the Jina Embeddings V3 model, optimized for GPU acceleration.

## Features

- Fast, GPU-accelerated embeddings using CUDA
- Support for batch processing of texts
- Configurable max token length (up to 8192 tokens)
- Task-specific embeddings for different use cases
- Half-precision (FP16) for improved performance
- Simple RESTful API interface
- Resilient client with retry/backoff capabilities for LlamaIndex integration

## Requirements

- NVIDIA GPU with CUDA support (tested on G5.xlarge instances)
- Python 3.8+
- NVIDIA drivers and CUDA toolkit

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/jina-embeddings-api.git
   cd jina-embeddings-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   # Flash Attention is optional but recommended for improved performance
   pip install flash-attention --no-build-isolation
   ```

## Usage

### Starting the API Server

```bash
# Start the server with default settings
python app/main.py

# Or use uvicorn directly with custom settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

For production deployments, you can use the provided systemd service file:

```bash
sudo cp jina-embeddings-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jina-embeddings-api
sudo systemctl start jina-embeddings-api
```

### API Endpoints

#### GET /
Returns information about the running API.

#### POST /embed
Generates embeddings for a list of texts.

**Request Body:**
```json
{
  "texts": [
    "Your first text to embed",
    "Your second text to embed",
    "..."
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [...],  // First embedding vector
    [...],  // Second embedding vector
    ...
  ],
  "dimensions": 1024,
  "processing_time_ms": 152.45,
  "texts_processed": 2
}
```

### Using the LlamaIndex Client

The repository includes a client class for easy integration with LlamaIndex:

```python
from embedding_client import JinaEmbedAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Initialize the embedding client
embedder = JinaEmbedAPI(api_url="http://your-server:8000")

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embedder)

# Search with the same embedding client
query_engine = index.as_query_engine()
response = query_engine.query("Your question here")
```

## Task-Specific Embeddings

Jina Embeddings V3 supports different tasks for specialized embeddings:

- `retrieval.query`: For encoding search queries
- `retrieval.passage`: For encoding documents to be retrieved
- `separation`: For document separation tasks
- `classification`: For text classification tasks
- `text-matching`: For text similarity tasks

For optimal retrieval performance, use `retrieval.passage` when embedding documents for storage and `retrieval.query` when embedding search queries.

## Performance Optimization

- The server uses half-precision (FP16) by default on CUDA-enabled devices
- For best performance, use batch processing when possible
- The maximum context length supports up to 8192 tokens
- On G5.xlarge instances, 2 workers is recommended for optimal throughput

## About

This project is maintained by Kevin Williams at [Ragu.ai](https://ragu.ai).

### Authors

- [Kevin Williams] - Co-founder, Ragu.ai


### Contact

For questions or support, please reach out to us at:
- Email: [kevin@ragu.ai or your preferred email]
- GitHub Issues: Please file any bugs or feature requests through the issue tracker on this repository

Ragu.ai provides AI tools and APIs to help businesses integrate AI into their workflows.

## License

This project is licensed under the MIT License - see below for details:

```markdown
MIT License

Copyright (c) 2025 Ragu.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements

- This project uses the [Jina Embeddings V3](https://huggingface.co/jinaai/jina-embeddings-v3) model
- Built with FastAPI, PyTorch, and Transformers