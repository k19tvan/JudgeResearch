"""
Problem 12 Solution: Training and Evaluation Loop for DFINEMini.
Imports model, criterion, matcher from Problems 11, 05, 03.
"""
import torch
import torch.nn as nn
import sys
import os
import math

# Make the solution self-contained by inlining all dependencies.
# In a real integration, import from your assembled project structure.

# ───── Inlined dependencies ────────────────────────────────────────────────

def cxcywh_to_xyxy(x):
    cx,cy,w,h = x.unbind(-1)
    return torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], -1)

def _ba(b): return (b[:,2]-b[:,0])*(b[:,3]-b[:,1])

def giou(b1, b2):
    a1=_ba(b1); a2=_ba(b2)
    lt=torch.max(b1[:,None,:2],b2[:,:2]); rb=torch.min(b1[:,None,2:],b2[:,2:])
    inter=((rb-lt).clamp(0)).prod(-1); union=a1[:,None]+a2-inter; iou=inter/union.clamp(1e-6)
    elt=torch.min(b1[:,None,:2],b2[:,:2]); erb=torch.max(b1[:,None,2:],b2[:,2:])
    encl=((erb-elt).clamp(0)).prod(-1)
    return iou-(encl-union)/encl.clamp(1e-6)

def box_iou(b1, b2):
    a1=_ba(b1); a2=_ba(b2)
    lt=torch.max(b1[:,None,:2],b2[:,:2]); rb=torch.min(b1[:,None,2:],b2[:,2:])
    inter=((rb-lt).clamp(0)).prod(-1); union=a1[:,None]+a2-inter
    return inter/union.clamp(1e-6), union

def cbr(ic,oc,k=3,s=1,p=1,g=1):
    return nn.Sequential(nn.Conv2d(ic,oc,k,s,p,groups=g,bias=False),nn.BatchNorm2d(oc),nn.ReLU(True))

class HGStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.s=nn.Sequential(cbr(3,24,3,2,1),cbr(24,48,3,2,1))
        self.l=nn.Sequential(cbr(48,96,1,1,0),cbr(96,96,3,2,1,96),cbr(96,96,1,1,0))
    def forward(self,x): s=self.s(x); return s,self.l(s)

class SinePE2D(nn.Module):
    def __init__(self,d,T=10000): super().__init__(); self.d=d;self.T=T
    def forward(self,x):
        B,C,H,W=x.shape; dev=x.device; dt=x.dtype
        yg=torch.arange(H,device=dev,dtype=dt)[:,None].expand(H,W)/H
        xg=torch.arange(W,device=dev,dtype=dt)[None,:].expand(H,W)/W
        q=self.d//4; dt2=torch.arange(q,device=dev,dtype=dt)
        om=self.T**(2*dt2/self.d)
        ax=2*math.pi*xg[...,None]/om; ay=2*math.pi*yg[...,None]/om
        px=torch.stack([ax.sin(),ax.cos()],-1).flatten(-2)
        py=torch.stack([ay.sin(),ay.cos()],-1).flatten(-2)
        p=torch.cat([px,py],-1)[None].expand(B,-1,-1,-1)
        return p.flatten(1,2)

class MHA(nn.Module):
    def __init__(self,d,h,dr=0):
        super().__init__(); self.h=h;self.dk=d//h;self.sc=math.sqrt(self.dk);self.d=d
        self.wq=nn.Linear(d,d);self.wk=nn.Linear(d,d);self.wv=nn.Linear(d,d);self.wo=nn.Linear(d,d)
        self.drop=nn.Dropout(dr)
    def _sh(self,x): B,T,_=x.shape; return x.reshape(B,T,self.h,self.dk).transpose(1,2)
    def forward(self,q,k,v,mask=None):
        B,Tq,_=q.shape; Q=self._sh(self.wq(q));K=self._sh(self.wk(k));V=self._sh(self.wv(v))
        s=Q@K.transpose(-2,-1)/self.sc
        if mask is not None: s=s.masked_fill(mask[:,None,None,:],float('-inf'))
        a=self.drop(s.softmax(-1)); c=(a@V).transpose(1,2).contiguous().reshape(B,Tq,self.d)
        return self.wo(c),a

class EncL(nn.Module):
    def __init__(self,d,h,df,dr=0):
        super().__init__(); self.a=MHA(d,h,dr);self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.l1=nn.Linear(d,df);self.l2=nn.Linear(df,d);self.act=nn.GELU();self.drop=nn.Dropout(dr)
    def forward(self,s,pos):
        n=self.n1(s); a,_=self.a(n+pos,n+pos,n); s=s+self.drop(a)
        n2=self.n2(s); return s+self.drop(self.l2(self.drop(self.act(self.l1(n2)))))

class DecL(nn.Module):
    def __init__(self,d,h,df,dr=0):
        super().__init__(); self.sa=MHA(d,h,dr);self.ca=MHA(d,h,dr)
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.n3=nn.LayerNorm(d)
        self.g1=nn.Linear(2*d,d);self.g2=nn.Linear(2*d,d)
        self.ff1=nn.Linear(d,df);self.ff2=nn.Linear(df,d);self.act=nn.GELU();self.drop=nn.Dropout(dr)
    def forward(self,t,m,qp,mp,msk=None):
        n1=self.n1(t); sa,_=self.sa(n1+qp,n1+qp,n1); t=t+self.drop(sa)
        n2=self.n2(t); co,_=self.ca(n2+qp,m+mp,m,mask=msk)
        gi=torch.cat([t,co],-1); t=torch.sigmoid(self.g1(gi))*t+torch.sigmoid(self.g2(gi))*co
        n3=self.n3(t); return t+self.drop(self.ff2(self.drop(self.act(self.ff1(n3)))))

class DFINEMini(nn.Module):
    def __init__(self,nc=10,nq=50,d=128,nel=2,ndl=3):
        super().__init__()
        self.d=d;self.nq=nq;self.nc=nc
        self.bb=HGStem(); self.proj=nn.Conv2d(96,d,1); self.pe=SinePE2D(d)
        self.enc=nn.ModuleList([EncL(d,4,d*4) for _ in range(nel)])
        self.dec=nn.ModuleList([DecL(d,4,d*4) for _ in range(ndl)])
        self.qe=nn.Embedding(nq,d); self.qp=nn.Embedding(nq,d)
        self.ch=nn.Linear(d,nc); self.bh=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,4))
    def forward(self,x):
        B=x.shape[0]; _,f=self.bb(x); fp=self.proj(f)
        src=fp.flatten(2).transpose(1,2); pos=self.pe(fp)
        m=src
        for enc in self.enc: m=enc(m,pos)
        t=self.qe.weight[None].expand(B,-1,-1); qp=self.qp.weight[None].expand(B,-1,-1)
        for dec in self.dec: t=dec(t,m,qp,pos)
        return {"pred_logits":self.ch(t),"pred_boxes":self.bh(t).sigmoid()}

import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self,wc=2,wb=5,wg=2,a=0.25,g=2.0):
        super().__init__(); self.wc=wc;self.wb=wb;self.wg=wg;self.a=a;self.g=g
    @torch.no_grad()
    def forward(self,outputs,targets):
        bs,nq=outputs["pred_logits"].shape[:2]
        op=F.sigmoid(outputs["pred_logits"].flatten(0,1)); ob=outputs["pred_boxes"].flatten(0,1)
        tids=torch.cat([v["labels"] for v in targets]); tb=torch.cat([v["boxes"] for v in targets])
        op2=op[:,tids]
        n=self.a*((1-op2)**self.g)*(-(op2+1e-8).log()); p=(1-self.a)*(op2**self.g)*(-(1-op2+1e-8).log())
        cc=n-p; cb=torch.cdist(ob,tb,p=1)
        cg=-giou(cxcywh_to_xyxy(ob),cxcywh_to_xyxy(tb))
        C=(self.wb*cb+self.wc*cc+self.wg*cg).nan_to_num(1.0).view(bs,nq,-1).cpu()
        sz=[len(v["boxes"]) for v in targets]
        ip=[linear_sum_assignment(c[i]) for i,c in enumerate(C.split(sz,-1))]
        return {"indices":[(torch.as_tensor(i,dtype=torch.int64),torch.as_tensor(j,dtype=torch.int64)) for i,j in ip]}

class SetCriterion(nn.Module):
    def __init__(self,nc,wv=1,wb=5,wg=2): super().__init__(); self.nc=nc;self.wv=wv;self.wb=wb;self.wg=wg
    def _idx(self,indices):
        bi=torch.cat([torch.full_like(s,i) for i,(s,_) in enumerate(indices)])
        si=torch.cat([s for s,_ in indices]); return bi,si
    def forward(self,outputs,targets,indices):
        pl=outputs["pred_logits"]; pb=outputs["pred_boxes"]; B,N,C=pl.shape
        nb=max(sum(len(t["labels"]) for t in targets),1); bi,si=self._idx(indices)
        ms=pb[bi,si]; mt=torch.cat([t["boxes"][j] for t,(_,j) in zip(targets,indices)])
        lbbox=F.l1_loss(ms,mt,reduction="sum")/nb
        sx=cxcywh_to_xyxy(ms); tx=cxcywh_to_xyxy(mt)
        gi_v=torch.diag(giou(sx,tx)); lgiou=(1-gi_v).sum()/nb
        tc=torch.full((B,N),self.nc,dtype=torch.int64,device=pl.device)
        tc[bi,si]=torch.cat([t["labels"][j] for t,(_,j) in zip(targets,indices)])
        gs=torch.zeros(B,N,device=pl.device)
        if ms.numel()>0:
            ious,_=box_iou(sx.detach(),tx); gs[bi,si]=torch.diag(ious).clamp(0).detach()
        p=torch.sigmoid(pl.flatten(0,1)); oh=F.one_hot(tc.flatten().clamp(0,C-1),C).float()
        tgt=oh*gs.flatten()[:,None]; w=0.75*p.pow(2)*(1-oh)+oh*gs.flatten()[:,None]
        vfl=F.binary_cross_entropy_with_logits(pl.flatten(0,1),tgt,weight=w.detach(),reduction="sum")/nb
        return {"loss_vfl":self.wv*vfl,"loss_bbox":self.wb*lbbox,"loss_giou":self.wg*lgiou}

# ───── Problem 12 Functions ─────────────────────────────────────────────────

def make_synthetic_batch(B, num_gt, num_classes=10, device=torch.device("cpu")):
    images = torch.randn(B, 3, 256, 256, device=device)
    targets = []
    for _ in range(B):
        cx = torch.rand(num_gt, device=device) * 0.6 + 0.2
        cy = torch.rand(num_gt, device=device) * 0.6 + 0.2
        w  = torch.rand(num_gt, device=device) * 0.25 + 0.05
        h  = torch.rand(num_gt, device=device) * 0.25 + 0.05
        # Clamp so boxes stay in [0,1]
        cx = cx.clamp(w/2+0.01, 1-w/2-0.01)
        cy = cy.clamp(h/2+0.01, 1-h/2-0.01)
        boxes  = torch.stack([cx, cy, w, h], -1)
        labels = torch.randint(0, num_classes, (num_gt,), device=device)
        targets.append({"labels": labels, "boxes": boxes})
    return images, targets


def train_one_epoch(model, criterion, matcher, optimizer, data_loader, device):
    model.train()
    totals = {"loss_vfl": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    for images, targets in data_loader:
        images  = images.to(device)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        optimizer.zero_grad()
        out = model(images)
        indices = matcher(out, targets)["indices"]
        losses  = criterion(out, targets, indices)
        total   = sum(losses.values())
        total.backward()
        optimizer.step()
        for k in totals:
            totals[k] += losses[k].item()
    n = max(len(data_loader), 1)
    return {f"avg_{k}": v / n for k, v in totals.items()}


def evaluate(model, data_loader, device):
    model.eval()
    all_ious = []
    with torch.no_grad():
        for images, targets in data_loader:
            images  = images.to(device)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            out     = model(images)
            pb      = out["pred_boxes"]  # (B, N, 4)
            for b, tgt in enumerate(targets):
                gt_boxes = tgt["boxes"]  # (T, 4)
                if gt_boxes.shape[0] == 0: continue
                pred_boxes = pb[b]  # (N, 4)
                # Compute IoU between all pred (N,) and all gt (T,)
                iou_mat, _ = box_iou(cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(gt_boxes))
                # For each GT, take max IoU across predictions
                best_iou_per_gt = iou_mat.max(0)[0]  # (T,)
                all_ious.append(best_iou_per_gt.mean().item())
    return {"mean_iou": sum(all_ious) / max(len(all_ious), 1)}
