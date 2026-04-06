# Problem 06 Theory - Transformer Decoder Layer

## Core Definitions
- Decoder refines object-query representations.
- It uses query self-attention and cross-attention against encoder memory.

## Variables and Shape Dictionary
- $N_{obj}$: number of object queries.
- $N_{src}$: encoder token count.
- $B$: batch size.
- $C$: hidden dimension.
- $T$: decoder query states $(N_{obj},B,C)$.
- $M$: encoder memory $(N_{src},B,C)$.
- $Q_e$: learned query embeddings $(N_{obj},B,C)$.
- $P$: encoder positional embeddings $(N_{src},B,C)$.

## Main Equations (LaTeX)
$$
T_1 = T + \text{MHA}(Q=T+Q_e, K=T+Q_e, V=T)
$$
$$
T_2 = T_1 + \text{MHA}(Q=T_1+Q_e, K=M+P, V=M)
$$
$$
Y = \text{Norm}\left(T_2 + \text{FFN}(\text{Norm}(T_2))\right)
$$

## Step-by-Step Derivation or Computation Flow
1. Self-attention among object queries reduces duplicate intent.
2. Cross-attention lets each query extract relevant spatial evidence from memory.
3. FFN updates each query feature independently in channel space.
4. Residuals and norms maintain gradient flow and stability.

## Tensor Shape Flow (Input -> Intermediate -> Output)
- Input query state: $(N_{obj},B,C)$.
- Self-attention output: $(N_{obj},B,C)$.
- Cross-attention output: $(N_{obj},B,C)$.
- FFN output: $(N_{obj},B,C)$.
- Layer output: $(N_{obj},B,C)$.

Worked mini-example:
- With $N_{obj}=100, N_{src}=400, B=2, C=256$:
- Cross-attention logits per head are proportional to $(B,h,100,400)$.

## Practical Interpretation
- Query embeddings act like learnable detection slots.
- Cross-attention is where object hypotheses connect to image evidence.
- Repeated decoder layers progressively refine object predictions.