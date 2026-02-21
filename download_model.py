from pathlib import Path
import os

# Create cache directory
cache = Path("./hf_cache").resolve()
cache.mkdir(exist_ok=True)

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = str(cache)
os.environ["HF_HOME"] = str(cache)

# Download model
from transformers import pipeline

p = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    cache_dir=str(cache),
)

print("Model downloaded and ready!")