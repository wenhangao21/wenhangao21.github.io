# Adopted from the original implementation: https://github.com/meta-llama/llama3/blob/main/llama/model.py
# Contact Wenhan -> wenhanacademia@gmail.com if you spot any error
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################### 1. Llama Initialization ######################
with open("original/params.json", "r") as f:
    config = json.load(f)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

print(config) # Check Config

model = torch.load("original/consolidated.00.pth") # Load the model weights
####################################################################


print("########################## 2.1 Input Text ##########################")
prompt = "The capital of France is"  # Hopefully, it will predict Paris
print("The input prompt is:", prompt)
print("####################################################################\n")


print("########################## 2.2 Tokenizer ###########################")
# From the original Llama code https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py lines 38-83
num_reserved_special_tokens = 256
pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501
tokenizer_path = "original/tokenizer.model"
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
num_base_tokens = len(mergeable_ranks)

special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens - 5)
        ]
special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=pat_str,
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}
)

assert tokenizer.decode(tokenizer.encode(prompt)) == prompt # check if tokenizer works

tokens = [128000] + tokenizer.encode(prompt) # 128000 is the <|begin_of_text|> token
tokens = torch.tensor(tokens).to(device)

# Check the  tokens of the input prompt
print([tokenizer.decode([token.item()]) for token in tokens])
print(tokens)
print("Token Shape:", tokens.shape)
print("####################################################################\n")


print("########################## 2.3 Embedding ###########################")
# nn.Embedding() is similar to nn.Parameters(), but  with efficient look up.
embedding_layer = torch.nn.Embedding(vocab_size, dim).to(device) # Initialize embedding layer
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"]) # Load weights
token_embeddings = embedding_layer(tokens).to(dtype=torch.bfloat16) # Look up the table for embeddings
print("Token Embedding Shape:", token_embeddings.shape)
####################################################################


print("###################### 2.4 Transformer Layers ######################")
# We will implement the first transformer layer for demonstration purposes and then write a class for all components at once. 

################## 2.4.1 Normalization ######################
x = token_embeddings # The input to the first layer is embedding; H^0 = E
def rms_norm(x, weight, eps=1e-5):
    norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * norm * weight
        
x_norm = rms_norm(x, model["layers.0.attention_norm.weight"], norm_eps) # (N, dim)
#############################################################

############### 2.4.3 Positional Encoding ###################
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # \Theta_k
    t = torch.arange(end, device=freqs.device, dtype=torch.float32) # i in our notation
    freqs = torch.outer(t, freqs)  # all \theta_i
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex numbers using Eulers Formula
    return freqs_cis # cis means cosine + i sine; the shape is (N, head_dim//2)


def apply_rotary_emb(
  xq: torch.Tensor, # (N, n_heads, head_dim)
  xk: torch.Tensor, # (N, n_kv_heads, head_dim)
  freqs_cis: torch.Tensor # (N, head_dim//2)
  ) -> Tuple[torch.Tensor, torch.Tensor]: 
    N, _, head_dim = xq.shape
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # convert to complex numbers
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.reshape(N, 1, head_dim//2) # reshape for broadcasting
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2) # convert back to real numbers
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk) # (N, n_heads, head_dim), (N, n_kv_heads, head_dim)
#############################################################


#################### 2.4.2 Attention ########################"
seqlen, _ = x_norm.shape
head_dim = dim//n_heads

wq = model["layers.0.attention.wq.weight"]  # (dim, dim) = (2048, 2048)
wk = model["layers.0.attention.wk.weight"]  # (dim/group_size, dim) = (512, 2048); group size = (n_heads/n_kv_heads)
wv = model["layers.0.attention.wv.weight"]  # (dim/group_size, dim) = (512, 2048)

xq = torch.matmul(x_norm, wq.T) # (N, dim) x (dim, dim) -> (N, dim) = (6, 2048)
xk = torch.matmul(x_norm, wk.T) # (N, dim) x (dim, dim/group_size) -> (N, dim/group_size) =  (6, 512)
xv = torch.matmul(x_norm, wv.T) # (N, dim) x (dim, dim/group_size) -> (N, dim/group_size) =  (6, 512)

xq = xq.view(seqlen, n_heads, head_dim) # (N, n_heads, head_dim) = (6, 32, 64)
xk = xk.view(seqlen, n_kv_heads, head_dim) # (N, n_kv_heads, head_dim) = (6, 8, 64)
xv = xv.view(seqlen, n_kv_heads, head_dim) # (N, n_kv_heads, head_dim) = (6, 8, 64)
print("The shapes of xq, xk, xv are:", xq.shape, xk.shape, xv.shape) 


# Computing heads individually is equivalent to a single matrix multiplciation then reshape
def verify_xq(x_norm, wq):
  for i in range(n_heads):
    wq = wq.view(n_heads, head_dim, dim)
    wq_i = wq[i]
    xq_i = torch.matmul(x_norm, wq_i.T) 
    if not torch.allclose(xq_i, xq[:,i,:], rtol=1e-2, atol=1e-2): # bfloat16 has low precision
      return False
    return True
print("Computing heads individually is equivalent to a single matrix multiplciation then reshape:", verify_xq(x_norm,wq))

freqs_cis = precompute_freqs_cis(head_dim, seqlen, rope_theta).to(device)
xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) # no change in dimension; (6, 32, 64), (6, 8, 64)
### Equation 2.3: ###
# repeat k,v to the shape of q for parallel computing
n_rep = n_heads//n_kv_heads 
xk, xv = torch.repeat_interleave(xk, repeats=n_rep, dim=1), torch.repeat_interleave(xv, repeats=n_rep, dim=1) # (6, 32, 64), (6, 32, 64)
# transpose for torch.matmul on the length and feature dimensions
xq, xk, xv = xq.transpose(0,1), xk.transpose(0,1), xv.transpose(0,1) # (32, 6, 64), (32, 6, 64), (32, 6, 64)
scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(head_dim)  # (32, 6, 6)
mask = torch.full((seqlen, seqlen), float("-inf"), device=device)    # (6, 6)
mask = torch.triu(mask, diagonal=1) # making it strictly upper triangular for causality
scores = scores + mask  # (32, 6, 6)
scores = F.softmax(scores.float(), dim=-1).type_as(x) # (32, 6, 6)
attention = torch.matmul(scores, xv) # (32, 6, 6) x (32, 6, 64) -> (32, 6, 64)
attention = attention.transpose(0, 1).contiguous().view(seqlen, -1) # (6, 2048) 
wo = model["layers.0.attention.wo.weight"] # (2048, 2048)
attention = torch.matmul(attention, wo.T) # (6, 2048)
#############################################################


################# 2.4.5 Residual and FFN #####################
h = x + attention # (6, 2048)
h_norm = rms_norm(h, model["layers.0.ffn_norm.weight"], norm_eps) # (6, 2048)
w1 = model["layers.0.feed_forward.w1.weight"] # (8192, 2048)
w3 = model["layers.0.feed_forward.w3.weight"] # (8192, 2048)
w2 = model["layers.0.feed_forward.w2.weight"] # (2048, 8192)
h1 = torch.matmul(h_norm, w1.T) # (6, 8192)
h2 = torch.matmul(h_norm, w3.T) # (6, 8192)
activated = torch.functional.F.silu(h1) * h2 # (6, 8192)
# The final output after the first transformer layer, same shape as the input. Features are casual
x = torch.matmul(activated, w2.T) + h # (6, 2048)
#############################################################


print("################# 2.X.X Repeat 31 Times #####################")
def transformer_block(x, layer):
  x_norm = rms_norm(x, model[f"layers.{layer}.attention_norm.weight"], norm_eps)
  seqlen, _ = x_norm.shape
  head_dim = dim//n_heads
  wq, wk, wv = model[f"layers.{layer}.attention.wq.weight"], model[f"layers.{layer}.attention.wk.weight"], model[f"layers.{layer}.attention.wv.weight"]
  xq, xk, xv = torch.matmul(x_norm, wq.T), torch.matmul(x_norm, wk.T), torch.matmul(x_norm, wv.T)
  xq, xk, xv = xq.view(seqlen, n_heads, head_dim),  xk.view(seqlen, n_kv_heads, head_dim), xv.view(seqlen, n_kv_heads, head_dim)
  freqs_cis = precompute_freqs_cis(head_dim, seqlen, rope_theta).to(device)
  xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
  n_rep = n_heads//n_kv_heads
  xk, xv = torch.repeat_interleave(xk, repeats=n_rep, dim=1), torch.repeat_interleave(xv, repeats=n_rep, dim=1) 
  xq, xk, xv = xq.transpose(0,1), xk.transpose(0,1), xv.transpose(0,1) 
  scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(head_dim)
  mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
  mask = torch.triu(mask, diagonal=1)
  scores = scores + mask
  scores = F.softmax(scores.float(), dim=-1).type_as(x)
  attention = torch.matmul(scores, xv)
  attention = attention.transpose(0, 1).contiguous().view(seqlen, -1)
  wo = model[f"layers.{layer}.attention.wo.weight"]
  attention = torch.matmul(attention, wo.T)
  h = x + attention
  h_norm = rms_norm(h, model[f"layers.{layer}.ffn_norm.weight"], norm_eps)
  w1, w2, w3 = model[f"layers.{layer}.feed_forward.w1.weight"], model[f"layers.{layer}.feed_forward.w2.weight"], model[f"layers.{layer}.feed_forward.w3.weight"]
  h1 = torch.matmul(h_norm, w1.T)
  h2 = torch.matmul(h_norm, w3.T)
  activated = torch.functional.F.silu(h1) * h2
  return torch.matmul(activated, w2.T) + h

for layer in range(1, n_layers):
  x = transformer_block(x, layer)
print("The shape of hidden feature after L transformer layers:",x.shape)
print("#############################################################")


print("################# 2.5 LM Head and Output ####################")
x_norm = rms_norm(x, model["norm.weight"], norm_eps)
logits = torch.matmul(x_norm[-1], model["output.weight"].T) # Only need the last row
next_token = torch.argmax(logits, dim=-1) # Greedy prediction
print("The next token is:", tokenizer.decode([next_token.item()]))
print("#############################################################")
