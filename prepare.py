"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import torch
import pyarrow.parquet as pq
import rustbpe
import tiktoken
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 1200        # training time budget in seconds (5 minutes) -> 1800 -> 1200
EVAL_TOKENS = 52428  # reduced for CPU evaluation, originally 40 * 524288

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

REPO_ID = "karpathy/climbmix-400b-shuffle"
MAX_SHARD = 6542 
VAL_SHARD = MAX_SHARD  # pinned validation shard
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard using huggingface_hub."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        return True

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_DIR,
            local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already exist at {DATA_DIR}")
        return

    print(f"Data: downloading {len(ids) - existing} shards...")
    with Pool(processes=download_workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready.")

# ---------------------------------------------------------------------------
# Tokenizer Training
# ---------------------------------------------------------------------------

def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and f != VAL_FILENAME)
    parquet_paths = [os.path.join(DATA_DIR, f) for f in files]
    
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return

def train_tokenizer():
    """Train BPE tokenizer and save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    print("Tokenizer: training BPE (rustbpe)...")
    
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Convert to tiktoken encoding
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    special_tokens = {name: len(mergeable_ranks) + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    # Build token_bytes for BPB evaluation
    token_bytes_list = []
    for i in range(enc.n_vocab):
        token_str = enc.decode([i])
        if token_str in set(SPECIAL_TOKENS):
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s.")

# ---------------------------------------------------------------------------
# Runtime & Dataloading
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    def get_vocab_size(self):
        return self.enc.n_vocab

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            return cls(pickle.load(f))

    def encode(self, text, prepend=None):
        prepend_id = prepend if isinstance(prepend, (int, type(None))) else self.enc.encode_single_token(prepend)
        if isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=8)
            if prepend_id is not None:
                for row in ids: row.insert(0, prepend_id)
            return ids
        ids = self.enc.encode_ordinary(text)
        if prepend_id is not None: ids.insert(0, prepend_id)
        return ids

    def decode(self, ids): return self.enc.decode(ids)

def _document_batches(split, batch_size=128):
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    paths = [val_path] if split == "val" else [os.path.join(DATA_DIR, f) for f in files if f != VAL_FILENAME]
    
    epoch = 1
    while True:
        for p in paths:
            pf = pq.ParquetFile(p)
            for rg_idx in range(pf.num_row_groups):
                batch = pf.read_row_group(rg_idx).column('text').to_pylist()
                for i in range(0, len(batch), batch_size):
                    yield batch[i:i+batch_size], epoch
        epoch += 1

def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """BOS-aligned best-fit packing dataloader."""
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_id = tokenizer.bos_token_id
    doc_buffer = []

    # Pin memory buffers for fast GPU transfer
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cpu")
    
    while True:
        rows = torch.empty((B, row_capacity), dtype=torch.long)
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                if len(doc_buffer) < buffer_size:
                    texts, epoch = next(batches)
                    doc_buffer.extend(tokenizer.encode(texts, prepend=bos_id))
                
                rem = row_capacity - pos
                # Best fit
                best_idx = -1
                for i, d in enumerate(doc_buffer):
                    if len(d) <= rem and (best_idx == -1 or len(d) > len(doc_buffer[best_idx])):
                        best_idx = i
                
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    rows[row_idx, pos:pos+len(doc)] = torch.tensor(doc)
                    pos += len(doc)
                else:
                    # Crop shortest
                    sid = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(sid)
                    rows[row_idx, pos:pos+rem] = torch.tensor(doc[:rem])
                    pos += rem

        cpu_buffer[:B*T].view(B, T).copy_(rows[:, :-1])
        cpu_buffer[B*T:].view(B, T).copy_(rows[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield gpu_buffer[:B*T].view(B, T), gpu_buffer[B*T:].view(B, T), epoch

# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Bits per byte (BPB) calculation."""
    device = next(model.parameters()).device
    token_bytes = torch.load(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), map_location=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    
    total_nats, total_bytes = 0.0, 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss = model(x, y, reduction='none').view(-1)
        nbytes = token_bytes[y.view(-1)]
        mask = nbytes > 0
        total_nats += (loss * mask).sum().item()
        total_bytes += nbytes.sum().item()
        
    return total_nats / (math.log(2) * total_bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()

    download_data(args.num_shards if args.num_shards != -1 else MAX_SHARD, args.download_workers)
    train_tokenizer()
    print("\nPreparation complete.")
