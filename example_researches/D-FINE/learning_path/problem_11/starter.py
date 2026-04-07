import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---- Reused modules from Problems 06-10 (inline copies) ----

def cbr(ic, oc, k=3, s=1, p=1, g=1):
    return nn.Sequential(nn.Conv2d(ic,oc,k,s,p,groups=g,bias=False),nn.BatchNorm2d(oc),nn.ReLU(True))


class HGNetV2Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(cbr(3,24,3,2,1),cbr(24,48,3,2,1))
        self.lcs=nn.Sequential(cbr(48,96,1,1,0),cbr(96,96,3,2,1,96),cbr(96,96,1,1,0))
    def forward(self,x): s=self.stem(x); return s, self.lcs(s)


class SinePE(nn.Module):
    def __init__(self,d,T=10000):
        super().__init__(); self.d=d; self.T=T
    def forward(self,x):
        B,C,H,W=x.shape; dev=x.device; dt=x.dtype
        yg=torch.arange(H,device=dev,dtype=dt).unsqueeze(1).expand(H,W)/H
        xg=torch.arange(W,device=dev,dtype=dt).unsqueeze(0).expand(H,W)/W
        hf=self.d//2; qf=hf//2
        dt_=torch.arange(qf,device=dev,dtype=dt)
        om=self.T**(2*dt_/self.d)
        ax=2*math.pi*xg.unsqueeze(-1)/om; ay=2*math.pi*yg.unsqueeze(-1)/om
        px=torch.stack([ax.sin(),ax.cos()],-1).flatten(-2); py=torch.stack([ay.sin(),ay.cos()],-1).flatten(-2)
        p=torch.cat([px,py],-1).unsqueeze(0).expand(B,-1,-1,-1)
        return p.flatten(1,2)


class MHA(nn.Module):
    def __init__(self,d,h,dr=0.0):
        super().__init__(); self.h=h;self.dk=d//h;self.sc=math.sqrt(self.dk);self.d=d
        self.wq=nn.Linear(d,d);self.wk=nn.Linear(d,d);self.wv=nn.Linear(d,d);self.wo=nn.Linear(d,d)
        self.drop=nn.Dropout(dr)
    def _sh(self,x): B,T,_=x.shape; return x.reshape(B,T,self.h,self.dk).transpose(1,2)
    def forward(self,q,k,v,mask=None):
        B,Tq,_=q.shape; Q=self._sh(self.wq(q));K=self._sh(self.wk(k));V=self._sh(self.wv(v))
        s=Q@K.transpose(-2,-1)/self.sc
        if mask is not None: s=s.masked_fill(mask[:,None,None,:],float('-inf'))
        a=self.drop(s.softmax(-1))
        c=(a@V).transpose(1,2).contiguous().reshape(B,Tq,self.d)
        return self.wo(c),a


class EncoderLayer(nn.Module):
    def __init__(self,d,h,df,dr=0.0):
        super().__init__(); self.attn=MHA(d,h,dr); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.l1=nn.Linear(d,df); self.l2=nn.Linear(df,d); self.act=nn.GELU(); self.drop=nn.Dropout(dr)
    def forward(self,src,pos):
        n=self.n1(src); q=k=n+pos; a,_=self.attn(q,k,n); src=src+self.drop(a)
        n2=self.n2(src); return src+self.drop(self.l2(self.drop(self.act(self.l1(n2)))))


class DecoderLayer(nn.Module):
    def __init__(self,d,h,df,dr=0.0):
        super().__init__(); self.sa=MHA(d,h,dr); self.ca=MHA(d,h,dr)
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d); self.n3=nn.LayerNorm(d)
        self.g1=nn.Linear(2*d,d); self.g2=nn.Linear(2*d,d)
        self.ff1=nn.Linear(d,df); self.ff2=nn.Linear(df,d); self.act=nn.GELU(); self.drop=nn.Dropout(dr)
    def forward(self,tgt,mem,qpos,mpos,msk=None):
        n1=self.n1(tgt); qs=ks=n1+qpos; sa,_=self.sa(qs,ks,n1); tgt=tgt+self.drop(sa)
        n2=self.n2(tgt); qc=n2+qpos; kc=mem+mpos; co,_=self.ca(qc,kc,mem,mask=msk)
        gi=torch.cat([tgt,co],-1); g1=torch.sigmoid(self.g1(gi)); g2=torch.sigmoid(self.g2(gi))
        tgt=g1*tgt+g2*co
        n3=self.n3(tgt); ff=self.ff2(self.drop(self.act(self.ff1(n3))))
        return tgt+self.drop(ff)


class DFINEMini(nn.Module):
    def __init__(self, num_classes=80, num_queries=100, d_model=256,
                 num_encoder_layers=2, num_decoder_layers=3):
        super().__init__()
        self.d = d_model; self.nq = num_queries; self.nc = num_classes
        self.backbone = HGNetV2Stem()
        self.proj     = nn.Conv2d(96, d_model, 1)
        self.pe       = SinePE(d_model)
        self.enc = nn.ModuleList([EncoderLayer(d_model, 8, d_model*4) for _ in range(num_encoder_layers)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, 8, d_model*4) for _ in range(num_decoder_layers)])
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos   = nn.Embedding(num_queries, d_model)
        self.cls_head    = nn.Linear(d_model, num_classes)
        self.box_head    = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, images):
        B = images.shape[0]
        _, feat = self.backbone(images)            # (B, 96, H', W')
        fp = self.proj(feat)                       # (B, d, H', W')
        src = fp.flatten(2).transpose(1,2)         # (B, T, d)
        pos = self.pe(fp)                          # (B, T, d)
        # Encode
        memory = src
        for enc in self.enc:
            memory = enc(memory, pos)
        # Decode
        tgt  = self.query_embed.weight.unsqueeze(0).expand(B,-1,-1)
        qpos = self.query_pos.weight.unsqueeze(0).expand(B,-1,-1)
        for dec in self.dec:
            tgt = dec(tgt, memory, qpos, pos)
        pred_logits = self.cls_head(tgt)
        pred_boxes  = self.box_head(tgt).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
