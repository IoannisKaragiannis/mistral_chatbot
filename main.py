import os
import torch
from uuid import uuid4
from typing import Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import snapshot_download

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate  # to manage reusable, parameterized prompts
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# This monolith is hard to split across GPUs
MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.1"
LOCAL_MISTRAL = os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.1")
# instead of having one huge bin file we will be using the sharded version
# that it's easier to be allocated and fit on small GPUs
# It also helps faster parallel loads of multiple shards
MISTRAL_SHARDER_REPO = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"
LOCAL_MISTRAL_SHARDED = os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.1-sharded")
DEBUG = True
LANGCHAIN_ENABLED = True

# ─── DOWNLOAD IF MISSING ─────────────────────────────────────────────────────
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

ensure_snapshot(MISTRAL_REPO, LOCAL_MISTRAL)
ensure_snapshot(MISTRAL_SHARDER_REPO, LOCAL_MISTRAL_SHARDED)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
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
    return_full_text=False,
    max_new_tokens=256,
    device_map="auto",
)
print("✅ Text-Generator ready.")

mistral_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
print("✅ HuggingFacePipeline ready.")

# ─── MEMORY STORE ───────────────────────────────────────────────────────────
# session_id -> memory
_memory_store: Dict[str, ConversationBufferMemory] = {}

# ─── FASTAPI APP ────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── DATA MODELS ────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# ─── PROMPT & CHAIN FACTORY ────────────────────────────────────────────────────
chat_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful assistant. Here is the conversation so far:
{history}
User: {input}
Assistant:"""
)

def get_conversation_chain(memory: ConversationBufferMemory) -> ConversationChain:
    return ConversationChain(
        llm=mistral_llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=DEBUG,
    )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(
            key="session_id", value=session_id,
            httponly=True, samesite="lax"
        )
    
    if LANGCHAIN_ENABLED:
        # if using LangChain, init session memory
        _memory_store.setdefault(
            session_id,
            ConversationBufferMemory(memory_key="history")
        )
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    session_id = request.cookies.get("session_id")
    new_cookie = False
    if not session_id:
        session_id = str(uuid4())
        new_cookie = True

    # extract last user message
    user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        ""
    )

    if DEBUG:
        print(f"[{session_id}] User: {user_msg}")

    # decide whether to use LangChain or raw pipeline
    if LANGCHAIN_ENABLED:
        
        # get/create memory, run with LangChain
        memory = _memory_store.setdefault(
            session_id,
            ConversationBufferMemory(memory_key="history")
        )
        
        # build chain bound to this session's memory
        conversation = get_conversation_chain(memory)

        # predict (memory is updated internally)
        assistant_reply = conversation.predict(input=user_msg)
    else:
        # raw pipeline: no history, just predict from the latest prompt
        outputs = text_gen_pipeline(user_msg)
        assistant_reply = outputs[0]["generated_text"].strip()
    
    if DEBUG:
        print(f"[{session_id}] Assistant: {assistant_reply}")

    # prepare and send response
    response = JSONResponse({"role": "assistant", "content": assistant_reply})
    if new_cookie:
        response.set_cookie(
            key="session_id", value=session_id,
            httponly=True, samesite="lax"
        )
    return response