import os
import torch
from uuid import uuid4
from typing import Dict, List
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate  # to manage reusable, parameterized prompts
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

# This monolith is hard to split across GPUs
MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.1"
# instead of having one huge bin file we will be using the sharded version
# that it's easier to be allocated and fit on small GPUs
# It also helps faster parallel loads of multiple shards
MISTRAL_SHARDED_REPO = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"

DEBUG = True
LANGCHAIN_ENABLED = True

USE_RAG = True  # enable retrieval-augmented generation
DOCS_DIR = "docs" # folder with source documents for RAG

os.makedirs(DOCS_DIR, exist_ok=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("✅ BitsAndBytesConfig ready.")

tokenizer = AutoTokenizer.from_pretrained(MISTRAL_REPO)
print("✅ Tokenizer ready.")

model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_SHARDED_REPO,
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

# ─── SET UP RAG ─────────────────────────────────────────────────────────────
# ─── SET UP RAG (only if enabled and docs exist) ────────────────────────────
vector_store = None
if USE_RAG:
    # try to load an existing FAISS index from disk
    index_dir = Path(DOCS_DIR)
    idx_file = index_dir / "index.faiss"
    mapping_file = index_dir / "index.pkl"

    if idx_file.exists() and mapping_file.exists():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded FAISS vector store from '{DOCS_DIR}' (deserialization allowed)")
    else:
        print(f"⚠️ No FAISS index found in '{DOCS_DIR}'; disabling RAG.")
        USE_RAG = False


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
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

    if DEBUG:
        print(f"[{session_id}] User: {user_msg}")

    assistant_reply = None

    # decide whether to use LangChain or raw pipeline
    if USE_RAG and LANGCHAIN_ENABLED and vector_store:
        # use a memory keyed as "chat_history" so the chain can inject it automatically
        rag_memory = _memory_store.setdefault(
            f"{session_id}_rag",
            ConversationBufferMemory(
                memory_key="chat_history",   # <<— must match the chain’s expected key
                return_messages=True
            )
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        conv_rag = ConversationalRetrievalChain.from_llm(
            llm=mistral_llm,
            retriever=retriever,
            memory=rag_memory,
            verbose=DEBUG,
        )
        # now we only need to pass the question
        result = conv_rag({"question": user_msg})
        assistant_reply = result["answer"]
    elif USE_RAG and vector_store:

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}  
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=mistral_llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=DEBUG,
        )
        assistant_reply = qa_chain.run(user_msg)
    elif LANGCHAIN_ENABLED:
        
        # get/create memory, run with LangChain
        memory = _memory_store.setdefault( session_id, ConversationBufferMemory(memory_key="history"))
        
        # build chain bound to this session's memory
        llm_chain = get_conversation_chain(memory)

        # predict (memory is updated internally)
        assistant_reply = llm_chain.predict(input=user_msg)
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