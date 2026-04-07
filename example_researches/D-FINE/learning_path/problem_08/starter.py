import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model=d_model; self.num_heads=num_heads
        self.d_k=d_model//num_heads; self.scale=math.sqrt(self.d_k)
        self.W_q=nn.Linear(d_model,d_model); self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model); self.W_o=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout)
    def _sh(self,x):
        B,T,_=x.shape; return x.reshape(B,T,self.num_heads,self.d_k).transpose(1,2)
    def forward(self,q,k,v,mask=None):
        B,Tq,_=q.shape
        Q=self._sh(self.W_q(q)); K=self._sh(self.W_k(k)); V=self._sh(self.W_v(v))
        s=Q@K.transpose(-2,-1)/self.scale
        if mask is not None: s=s.masked_fill(mask[:,None,None,:],float('-inf'))
        a=self.drop(s.softmax(-1))
        c=(a@V).transpose(1,2).contiguous().reshape(B,Tq,self.d_model)
        return self.W_o(c), a


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, src, pos, src_key_padding_mask=None):
        # Pre-LN self-attention with positional injection into q and k
        norm = self.norm1(src)
        q = k = norm + pos
        attn_out, _ = self.attn(q, k, norm, mask=src_key_padding_mask)
        src = src + self.drop(attn_out)

        # Pre-LN FFN
        norm2 = self.norm2(src)
        ffn_out = self.linear2(self.drop(self.act(self.linear1(norm2))))
        return src + self.drop(ffn_out)
