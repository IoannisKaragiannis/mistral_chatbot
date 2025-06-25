"""
Author: Ioannis Karagiannis

When you set a high token limit, you must also give the model a clear “stop” signal—either via EOS tokens, 
early stopping, custom stop sequences, or an instruct-tuned checkpoint that’s been trained to wrap things
up cleanly. Otherwise, it will keep sampling until it hits your budget, which is exactly why you’re seeing
those echoes and topic jumps.

use https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 instead of
https://huggingface.co/mistralai/Mistral-7B-v0.1

so that the model knows when to stop its response

launch server: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import torch
import pickle
import faiss
import wikipedia
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ——— CONFIG —————————————————————————————————————————————————————————————————————————————
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely. "
    "If you are not certain, say you don't know."
)
RAG_INDEX = "wiki.index"
RAG_PASSAGES = "passages.pkl"
DEBUG = True

# ——— LOAD MODEL ———————————————————————————————————————————————————————————————————————————
print("⏳ Loading Mistral instruction-tuned model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb,
)
model.eval()
print("✅ Mistral ready.")

# ——— OPTIONAL RAG SETUP —————————————————————————————————————————————————————————————————
rag_enabled = False #os.path.exists(RAG_INDEX) and os.path.exists(RAG_PASSAGES)
if rag_enabled:
    print("⏳ Loading FAISS index…")
    index = faiss.read_index(RAG_INDEX)
    with open(RAG_PASSAGES, "rb") as f:
        passages = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ RAG ready.")
else:
    print("⚠️ RAG disabled; running vanilla generation only.")

# ——— UTILS —————————————————————————————————————————————————————————————————————————————
def fetch_wiki_summary(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2, auto_suggest=True).split("\n")[0]
    except:
        results = wikipedia.search(query, results=1)
        if not results:
            raise
        return wikipedia.summary(results[0], sentences=2, auto_suggest=False).split("\n")[0]

# ——— FASTAPI SETUP ———————————————————————————————————————————————————————————————————
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "rag_enabled": rag_enabled}
    )

@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if DEBUG:
        print("User message:", user_msg)

    # If RAG disabled: directly invoke Mistral on chat history
    if not rag_enabled:
        prompt_lines = [SYSTEM_PROMPT, ""]
        for m in req.messages:
            role = m.role.capitalize()
            prompt_lines.append(f"{role}: {m.content}")
        prompt_lines.append("Assistant:")
        prompt = "\n".join(prompt_lines)
        if DEBUG:
            print("Prompt for vanilla:", prompt[:300])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=(req.temperature>0),
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        gen = out[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if DEBUG:
            print("Vanilla answer:", answer)
        return JSONResponse({"role": "assistant", "content": answer})

    # RAG enabled branch: Wikipedia fallback for simple factoids
    lower = user_msg.lower()
    if lower.startswith(("who is ", "what is ", "when is ", "when did ", "where is ", "where did ")):
        entity = user_msg.split(maxsplit=2)[-1].rstrip("? !.")
        try:
            summary = fetch_wiki_summary(entity)
            if DEBUG:
                print("Wiki summary:", summary)
            return JSONResponse({"role": "assistant", "content": summary})
        except Exception as e:
            if DEBUG:
                print("Wiki failed:", e)

    # RAG extractive context
    q_emb = embedder.encode([user_msg])
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, k=1)
    doc = passages[I[0][0]]
    if len(doc) > 600:
        doc = doc[:600].rsplit(" ", 1)[0] + "…"
    context = f"Context: {doc}"
    if DEBUG:
        print("RAG context:", context)

    # build prompt with context
    prompt_lines = [SYSTEM_PROMPT, "", context, ""]
    for m in req.messages:
        role = m.role.capitalize()
        prompt_lines.append(f"{role}: {m.content}")
    prompt_lines.append("Assistant:")
    prompt = "\n".join(prompt_lines)
    if DEBUG:
        print("RAG prompt:", prompt[:300])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=(req.temperature>0),
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    gen = out[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if DEBUG:
        print("RAG answer:", answer)
    return JSONResponse({"role": "assistant", "content": answer})
