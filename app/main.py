from functools import cached_property
import time
import gc
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer
import sentry_sdk
import os
from dotenv import load_dotenv

load_dotenv(override=False)

environment = os.getenv("ENVIRONMENT", "development")
sentry_dsn = os.getenv("SENTRY_DSN", None)

# Configure CUDA and PyTorch for memory efficiency
if torch.cuda.is_available():
    # Empty cache at startup
    torch.cuda.empty_cache()
    
    # Configure PyTorch settings for efficiency
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    
    # Print GPU information for monitoring
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Current allocated memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_max_length = 8192

print(f"Using device: {device}")
print(f"Environment: {environment}")
print(f"sentry_dsn: {sentry_dsn}")

if sentry_dsn:
    sentry_sdk.init(
            dsn=sentry_dsn,
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
    # Load model with optimized settings
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Always use fp16 for better memory efficiency
        low_cpu_mem_usage=True      # Reduce CPU memory usage during loading
    ).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Print memory usage after model loading
if torch.cuda.is_available():
    print(f"Memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

class EmbeddingRequest(BaseModel):
    texts: List[str]

    @cached_property
    def size(self) -> int:
        return len(self.texts)

# Define a function to clean up memory after processing
def cleanup_memory():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Batch processing function to handle larger input sets efficiently
def batch_process(texts, batch_size=8, max_length=None):
    """Process texts in smaller batches to manage memory better"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize the input texts
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          return_tensors="pt", max_length=(max_length or _max_length)).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)
            
        # Clear intermediate tensors
        del inputs, outputs
        if i % (batch_size * 4) == 0:  # Periodically clear cache for large jobs
            cleanup_memory()
    
    return all_embeddings

def use_torch(request: EmbeddingRequest, max_length: int = None):
    # For larger batches, use our batching function
    if request.size > 16:
        return batch_process(request.texts, batch_size=8, max_length=max_length)
    
    # For smaller batches, process normally
    inputs = tokenizer(request.texts, padding=True, truncation=True, 
                      return_tensors="pt", max_length=(max_length or _max_length)).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy().tolist()
    
    return embeddings
    
def use_task(request: EmbeddingRequest, task: str, max_length: int = None):
    # If model's encode method supports batching, you could implement similar logic here
    embeddings = model.encode(request.texts, task=task, max_length=(max_length or _max_length))
    return embeddings.tolist()

@app.post("/embed")
async def create_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks, 
                           task: Optional[str] = None, max_length: Optional[int] = None):
    print(f"Received batch request of size: {request.size}")
    
    # Log memory state before processing
    if torch.cuda.is_available():
        print(f"Memory before processing: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    start_time = time.time()
    try:
        if task:
            embeddings = use_task(request, task, max_length)
        else:
            embeddings = use_torch(request, max_length)
        
        process_time = time.time() - start_time
        print(f"Processed {request.size} texts in {process_time:.2f} seconds")
        
        # Schedule cleanup after response
        background_tasks.add_task(cleanup_memory)
        
        # Log memory after processing
        if torch.cuda.is_available():
            print(f"Memory after processing: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]),
            "processing_time_ms": round(process_time * 1000, 2),
            "texts_processed": request.size
        }
    except Exception as e:
        # Ensure we clean up even on error
        background_tasks.add_task(cleanup_memory)
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
@app.get("/")
async def healthcheck():
    # Include memory stats in healthcheck
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            "max_allocated_memory_gb": torch.cuda.max_memory_allocated(0) / 1e9,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    
    return {
        "message": "Jina Embeddings API is running", 
        "model": model_name, 
        "environment": environment,
        "sentry": sentry_dsn is not None,
        "memory_stats": memory_info
    }

@app.get("/memory")
async def memory_stats():
    """Endpoint to check current memory usage"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
        "max_allocated_memory_gb": torch.cuda.max_memory_allocated(0) / 1e9,
        "cached_memory_gb": torch.cuda.memory_reserved(0) / 1e9,
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
    }

@app.post("/clear-memory")
async def clear_memory():
    """Endpoint to manually clear CUDA memory"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    before = torch.cuda.memory_allocated(0) / 1e9
    cleanup_memory()
    after = torch.cuda.memory_allocated(0) / 1e9
    
    return {
        "message": "Memory cleared",
        "before_gb": before,
        "after_gb": after,
        "difference_gb": before - after
    }

@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
    return {"message": "This will never be reached"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)