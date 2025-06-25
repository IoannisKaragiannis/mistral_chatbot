# build_index.py

import faiss, pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

'''
Dataset builder code
Hugging Face downloads the “wikipedia” dataset repo, inspects its custom Python files
(the “dataset script”), and—because you passed trust_remote_code=True—executes them without prompting.

Data split
You ask for the first 1% of all English Wikipedia articles (roughly 16 k pages). Behind the scenes 
this uses streaming download, JSON parsing of MediaWiki dumps, and yields Python dicts like {"title": ..., "text": ...}.
'''

# 1. load corpus
ds = load_dataset(
     "wikipedia",
     "20220301.en",
     split="train[:1%]",
     trust_remote_code=True,
 )

'''
Paragraph segmentation
We simply split each article's full text on double-newlines.

Filtering
Only keep paragraphs longer than ~100 characters, to avoid tiny fragments.

Normalization
Replace internal newlines with spaces so each passage is a single line of plain text.
'''

# 2. chunk into passages
passages = []
for page in ds:
    text = page["text"]
    # simple split on paragraphs
    for para in text.split("\n\n"):
        if len(para) > 100:
            passages.append(para.replace("\n", " "))

'''
Model download
HF pulls down the all-MiniLM-L6-v2 weights and config.

Tokenization & batching
Each passage is tokenized (into subword IDs), batched, and fed through the MiniLM transformer.

Pooling
The model outputs a fixed-size vector (384 dimensional) per passage, typically by mean-pooling the last layer’s token embeddings.

Result
You end up with an (N x 384) NumPy array, where N is the number of passages.
'''

# 3. encode with SBERT
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(passages, show_progress_bar=True)

'''
Index type
IndexFlatIP is an exact inner-product index. By normalizing vectors to unit length (with normalize_L2), inner product = cosine similarity.

Normalization
We modify embs in-place so each row has Euclidean norm = 1.

Adding vectors
index.add(embs) copies all your normalized passage embeddings into the FAISS index's in-memory store.
'''

# 4. build FAISS index
d = embs.shape[1]
index = faiss.IndexFlatIP(d)  # inner product (cosine w/ normalized vectors)
faiss.normalize_L2(embs)
index.add(embs)

'''
Index file
wiki.index is a compact binary file containing your FAISS structure + vectors.

Passage store
passages.pkl is a simple Python pickle of your raw text passages list. You’ll need it later to map from FAISS result IDs → actual text.
'''

# 5. save
faiss.write_index(index, "wiki.index")
with open("passages.pkl", "wb") as f:
    pickle.dump(passages, f)
