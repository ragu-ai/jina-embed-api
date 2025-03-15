from functools import cached_property
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer
import sentry_sdk
import os


environment = os.getenv("ENVIRONMENT", "development")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_max_length = 8192

print(f"Using device: {device}")
print(f"Environment: {environment}")


sentry_sdk.init(
        dsn="https://56ae6f42993cb26c70fd9c8e9c8dd9ae@o4508978501058560.ingest.us.sentry.io/4508982717972480",
        send_default_pii=True,
        shutdown_timeout=5,
        # Add unique process identification to help debugging
        release=f"ragu-embeddings-{environment}-{os.getpid()}",
        # Set environment based on your settings
        environment=environment,
    )

app = FastAPI(title="Jina Embeddings API")

# Load the model and tokenizer
model_name = "jinaai/jina-embeddings-v3"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    # Use half-precision for faster inference
    if device.type == "cuda":
        model = model.half()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class EmbeddingRequest(BaseModel):
    texts: List[str]

    @cached_property
    def size(self) -> int:
        return len(self.texts)

def use_torch(request: EmbeddingRequest, max_length: int = None):
    # Tokenize the input texts
    inputs = tokenizer(request.texts, padding=True, truncation=True, 
                        return_tensors="pt", max_length=(max_length or _max_length)).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy().tolist()
    
    return embeddings
    

def use_task(request: EmbeddingRequest, task: str, max_length: int = None):
        embeddings = model.encode(request.texts, task=task, max_length=(max_length or _max_length))
        return embeddings.tolist()


@app.post("/embed")
async def create_embeddings(request: EmbeddingRequest, task: Optional[str] = None, max_length: Optional[int] = None):
    print(f"Received batch request of size: {request.size}")
    start_time = time.time()
    try:
        if task:
            embeddings = use_task(request, task, max_length)
        else:
            embeddings = use_torch(request, max_length)
        
        process_time = time.time() - start_time
        print(f"Processed {request.size} texts in {process_time:.2f} seconds")
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]),
            "processing_time_ms": round(process_time * 1000, 2),
            "texts_processed": request.size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    

@app.get("/")
async def healthcheck():
    return {"message": "Jina Embeddings API is running", "model": model_name, "environment": environment}

@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
    return {"message": "This will never be reached"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)