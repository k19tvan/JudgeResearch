# learning_path/problem_06/checker.py
import torch
import torch.nn as nn
from starter import TransformerDecoderLayer

def check_decoder_layer():
    module = TransformerDecoderLayer()
    
    # We just explicitly bypass implementation error when possible or mock internally
    N_obj, N_src, B, C = 10, 20, 2, 256
    tgt = torch.rand(N_obj, B, C)
    memory = torch.rand(N_src, B, C)
    query_embed = torch.rand(N_obj, B, C)
    pos_embed = torch.rand(N_src, B, C)
    
    try:
        out = module(tgt, memory, query_embed, pos_embed)
        assert out.shape == (N_obj, B, C), f"Output shape mismatch: {out.shape}"
        print("All Problem 06 checks passed")
    except NotImplementedError:
        print("Implement the Decoder Layer to pass.")
        raise

if __name__ == "__main__":
    check_decoder_layer()
