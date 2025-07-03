#!/usr/bin/env python3
# build_vectordb.py

import os
import argparse
from pathlib import Path
from googlesearch import search as google_search
import time
import asyncio # Import asyncio for running async functions

# set USER_AGENT so Playwright won’t warn
os.environ.setdefault(
    "USER_AGENT",
    "mistral_chatbot/1.0 (+https://github.com/yourname/mistral_chatbot)"
)

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Defaults ────────────────────────────────────────────────────────────────────
DOCS_DIR         = "docs"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
DEFAULT_NUM_URLS = 5

async def build_from_urls(urls, out_dir, embed_model, chunk_size, chunk_overlap):
    try:
        # 1) Fetch & render pages using the ASYNCHRONOUS method aload()
        loader = AsyncChromiumLoader(urls=urls, headless=True)
        print("Starting AsyncChromiumLoader.aload()...")
        raw_docs = await loader.aload()
        print(f"✅ Loaded {len(raw_docs)} raw documents")

        # Give Playwright's internal cleanup a moment before proceeding
        # This is particularly important right after the resource-intensive aload()
        await asyncio.sleep(0.5) # Reduced from 1s, 0.5s is usually sufficient
        print("✅ Async cleanup delay after aload() completed.")

        # 2) HTML → plain text
        transformer = Html2TextTransformer()
        docs = transformer.transform_documents(raw_docs)
        print(f"✅ Transformed to {len(docs)} text documents")

        # 3) Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        print(f"✅ Split into {len(chunks)} chunks")

        # 4) Embed & index
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        vectordb = FAISS.from_documents(chunks, embeddings)
        print("✅ FAISS index built")

        # 5) Persist
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True)
        vectordb.save_local(str(out_path))
        print(f"✅ Saved vector DB to '{out_path}'")

    except Exception as e:
        print(f"❌ An error occurred during build_from_urls: {e}")
    finally:
        # Final desperate measure: Ensure event loop cleanup if something is stuck
        # This explicit loop closing is usually handled by asyncio.run(),
        # but adding it here as an extra layer if something truly fails to exit.
        # This is more for debugging/ensuring graceful shutdown in complex scenarios.
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            # Gather all remaining tasks and cancel them
            pending_tasks = [
                task for task in asyncio.all_tasks(loop=loop)
                if task is not asyncio.current_task(loop=loop)
            ]
            if pending_tasks:
                print(f"ℹ️ Cancelling {len(pending_tasks)} pending tasks...")
                for task in pending_tasks:
                    task.cancel()
                # Wait for tasks to be cancelled
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                print("ℹ️ Pending tasks cancelled.")
            # loop.close() # Do NOT manually close the loop here if asyncio.run is managing it
                            # asyncio.run() handles loop creation and closing.
        print("✅ build_from_urls function completed its lifecycle.")

def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector DB from URLs or a search query"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--urls',  nargs='+', help='List of URLs to ingest')
    group.add_argument('--query',  help='Search query (uses googlesearch)')
    parser.add_argument('--num_urls',      type=int,   default=DEFAULT_NUM_URLS)
    parser.add_argument('--docs_dir',      default=DOCS_DIR)
    parser.add_argument('--embed_model',   default=EMBED_MODEL)
    parser.add_argument('--chunk_size',    type=int,   default=CHUNK_SIZE)
    parser.add_argument('--chunk_overlap', type=int,   default=CHUNK_OVERLAP)
    args = parser.parse_args()

    if args.query:
        if google_search is None:
            print("❌ googlesearch not installed; install with `pip install googlesearch-python`")
            return
        print(f"🔍 Searching for '{args.query}' ({args.num_urls} results)…")
        urls = list(google_search(args.query, num_results=args.num_urls))
        print("✅ Found URLs:\n  " + "\n  ".join(urls))
    else:
        urls = args.urls

    asyncio.run(build_from_urls(
        urls,
        out_dir=args.docs_dir,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    ))
    
if __name__ == "__main__":
    main()