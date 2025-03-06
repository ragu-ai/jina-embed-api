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

### Using with  LlamaIndex

```python
from enum import Enum
from pydantic import Field
import requests
import time
import random
import logging
from typing import List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from llama_index.core.embeddings.base import BaseEmbedding, Embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddingsAPI(BaseEmbedding):
    """Embedding class for Jina Embeddings API with retry and backoff."""

    class EmbedTask(Enum):
        PASSAGE = "retrieval.passage"
        QUERY = "retrieval.query"
    
    api_url: str = Field(description="URL of the Jina embeddings API")
    api_timeout: int = Field(description="API Timeout")
    embed_task: Optional[EmbedTask] = Field(description="Embedding Mode")
    max_length: Optional[int] = Field(description="Maximum length of input text to use")
    
    def __init__(
        self,
        api_url: str,
        model_name: str = "jinaai/jina-embeddings-v3",
        embed_batch_size: int = 10,
        num_workers: Optional[int] = None,
        api_timeout: int = 60,
        embed_task: EmbedTask = None,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the CustomEmbeddingsAPI.
        
        Args:
            api_url: URL of the Jina embeddings API endpoint
            model_name: Name of the model
            embed_batch_size: Maximum batch size for embedding requests
            num_workers: Number of async workers
            api_timeout: Timeout for API requests in seconds
            embed_task: Task-specific embedding mode
            max_length: Maximum token length for inputs
        """
        # Only pass parameters that BaseEmbedding expects
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        # Set instance variables
        self.api_url = api_url
        self.api_timeout = api_timeout
        self.embed_task = embed_task
        self.max_length = max_length
        
        # Verify API connectivity on initialization
        self._check_api_connection()

    # Rest of implementation...

# Example usage:
embedding_client = CustomEmbeddingsAPI(
    api_url="http://localhost:8000",
    embed_task=CustomEmbeddingsAPI.EmbedTask.PASSAGE,
    embed_batch_size=20
)

# Use with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index with custom embeddings
index = VectorStoreIndex.from_documents(
    documents, 
    embed_model=embedding_client
)

# Create a query engine
query_engine = index.as_query_engine()

# Execute a search query
response = query_engine.query("Your search query here")
print(response)
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
- Email: kevin@ragu.ai
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