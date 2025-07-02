"""
Author: Ioannis Karagiannis

Launch server: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFacePipeline

# ─── CONFIG ─────────────────────────────────────────────────────────────────

# Where to stash your downloaded models
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Mistral Instruct
MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.1"
LOCAL_MISTRAL = os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.1")

# link for the sharded model
MISTRAL_SHARDER_REPO = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded" 
LOCAL_MISTRAL_SHARDED = os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.1-sharded")

DEBUG = True

# ─── DOWNLOAD IF MISSING ────────────────────────────────────────────────────
def ensure_snapshot(repo_id: str, local_path: str):
    if not os.path.isdir(local_path):
        print(f"⏳ Downloading {repo_id} → {local_path}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            use_auth_token=True
        )
    else:
        print(f"✅ {local_path} already exists, skipping download")

# Mistral
ensure_snapshot(MISTRAL_REPO, LOCAL_MISTRAL)

# Mistral-sharder
ensure_snapshot(MISTRAL_SHARDER_REPO, LOCAL_MISTRAL_SHARDED)

# ─── LOAD MISTRAL MODEL ─────────────────────────────────────────────────────
# Configuration for loading the model in 4-bit precision to save memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load model in 4-bit precision
    bnb_4bit_use_double_quant=True,       # Use double quantization for better compression
    bnb_4bit_quant_type="nf4",            # Specify quantization type as NF4 (a specific format for quantization)
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 as the compute data type for operations
)
print("✅ BitsAndBytesConfig ready.")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MISTRAL, local_files_only=True)
print("✅ Tokenizer ready.")

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MISTRAL_SHARDED,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)

model.eval()
print("✅ Mistral ready.")

text_gen_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.85, 
    eos_token_id=tokenizer.eos_token_id, 
    pad_token_id=tokenizer.eos_token_id, 
    repetition_penalty=1.1, 
    return_full_text=False, # if set to True it will return the prompt as well
    max_new_tokens=256,
    device_map="auto",
)

print("✅ Text-Generator Pipeline ready.")

mistral_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

print("✅ HuggingFacePipeline Mistral-LLM ready.")

# ─── FASTAPI APP ────────────────────────────────────────────────────────────

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

# Add this new endpoint for the root URL
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    if DEBUG:
        print("User message:", user_msg)

    # Vanilla generation
    mistral_response = mistral_llm.invoke(user_msg)
    if DEBUG:
        print("Answer:", mistral_response)
    return JSONResponse({"role": "assistant", "content": mistral_response})
