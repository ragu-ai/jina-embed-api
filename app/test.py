import requests
import time

url = "http://localhost:8000/embed"
texts = ["This is a test sentence for embedding.", 
         "Let's see how fast our embedding API performs!",
         "GPU acceleration should make this very quick."]

payload = {"texts": texts}

print("Testing embedding API with 3 sample texts...")
start = time.time()
response = requests.post(url, json=payload)
end = time.time()

if response.status_code == 200:
    result = response.json()
    print(f"Success! API responded in {end - start:.4f} seconds")
    print(f"API processing time: {result['processing_time_ms']} ms")
    print(f"Embedding dimensions: {result['dimensions']}")
    # Print first 5 dimensions of first embedding
    print(f"Sample of first embedding: {result['embeddings'][0][:5]}...")
else:
    print(f"Error: {response.status_code}")
    print(response.text)