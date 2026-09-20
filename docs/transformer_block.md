# Complete Transformer Block: Flow and Equations

## High-Level Architecture

A modern decoder-only Transformer block has two major sub-layers:

1. Multi-Head Attention
2. SwiGLU Feed Forward Network

Both follow the **Pre-Norm + Residual Connection** pattern.

The two main equations are:

$$
Y = X + MHA(RMSNorm(X))
$$

$$
Z = Y + FFN(RMSNorm(Y))
$$

Where:

- $X$ = input to the Transformer block
- $Y$ = output after Attention
- $Z$ = final output of the Transformer block

---

# Complete Flow Diagram

```text
                         INPUT X
                      [T, d_model]
                            │
                            ▼
                         RMSNorm
                            │
                            ▼
                  X_norm = RMSNorm(X)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
              Q             K             V
              │             │             │
              ▼             ▼             ▼
         X_norm W_Q    X_norm W_K    X_norm W_V
              │             │             │
              └──────┬──────┘             │
                     ▼                    │
                  QKᵀ                     │
                     │                    │
                     ▼                    │
              Divide by √d_head           │
                     │                    │
                     ▼                    │
                Causal Mask               │
                     │                    │
                     ▼                    │
                  Softmax                 │
                     │                    │
                     ▼                    │
             Attention Weights ───────────┘
                     │
                     ▼
              Attention Weights × V
                     │
                     ▼
                Head Outputs
                     │
                     ▼
                Combine Heads
                     │
                     ▼
            Output Projection W_O
                     │
                     ▼
            Attention Output A
                     │
                     ▼
                 X + A
                     │
                     ▼
                     Y
                     │
                     ▼
                 RMSNorm(Y)
                     │
              ┌──────┴──────┐
              ▼             ▼
           Up Path       Gate Path
              │             │
              ▼             ▼
        Y_norm W_up   Y_norm W_gate
              │             │
              │             ▼
              │            SiLU
              │             │
              └───────⊙─────┘
                      │
                      ▼
                   Hidden
                      │
                      ▼
               Down Projection
                      │
                      ▼
                FFN Output F
                      │
                      ▼
                   Y + F
                      │
                      ▼
                 OUTPUT Z

                 [T, d_model]
```

# Steps 1 to 17

A decoder-only Transformer block has two major parts:

1. Multi-Head Attention
2. SwiGLU Feed Forward Network

Both use:

- RMSNorm before computation
- Residual connection after computation

---

# 1. RMSNorm Before Attention

Normalize each token independently.

$$
RMS(X)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d} X_i^2
+
\epsilon
}
$$

$$
X_{norm}
=
\frac{X}{RMS(X)}
\odot
\gamma
$$

Output shape remains:

$$
[T,d_{model}]
$$

---

# 2. Create Query, Key and Value

Project the normalized input into three representations.

$$
Q=X_{norm}W_Q
$$

$$
K=X_{norm}W_K
$$

$$
V=X_{norm}W_V
$$

Initially:

$$
[T,d_{model}]
\rightarrow
[T,d_{model}]
$$

---

# 3. Split into Multiple Heads

Split the model dimension across attention heads.

$$
d_{model}
=
n_{heads}
\times
d_{head}
$$

Therefore:

$$
d_{head}
=
\frac{d_{model}}{n_{heads}}
$$

Example:

$$
d_{model}=4,\quad n_{heads}=2,\quad d_{head}=2
$$

Conceptually:

$$
[T,d_{model}]
\rightarrow
[n_{heads},T,d_{head}]
$$

---

# 4. Calculate Attention Scores

Compare every Query with every Key.

$$
Scores=QK^T
$$

For one head:

$$
[T,d_{head}]
\times
[d_{head},T]
=
[T,T]
$$

Each score measures how strongly a Query matches a Key.

---

# 5. Scale Attention Scores

Scale the scores by the square root of the head dimension.

$$
ScaledScores
=
\frac{QK^T}
{\sqrt{d_{head}}}
$$

This keeps the score magnitudes controlled before Softmax.

---

# 6. Apply Causal Mask

A token must not see future tokens.

Future positions are replaced with:

$$
-\infty
$$

Therefore:

$$
MaskedScores
=
ScaledScores
+
Mask
$$

After masking:

```text
Token 1 → Token 1

Token 2 → Token 1, Token 2

Token 3 → Token 1, Token 2, Token 3

Token 4 → Token 1, Token 2, Token 3, Token 4
```

---

# 7. Apply Softmax

Convert attention scores into probabilities.

$$
AttentionWeights
=
Softmax(MaskedScores)
$$

Each row satisfies:

$$
\sum_{j=1}^{T}
AttentionWeights_{ij}
=
1
$$

---

# 8. Collect Value Information

Use the attention weights to combine Value vectors.

$$
HeadOutput
=
AttentionWeights
\times
V
$$

For one head:

$$
[T,T]
\times
[T,d_{head}]
=
[T,d_{head}]
$$

Each token now contains contextual information.

---

# 9. Combine Heads

Concatenate outputs from all attention heads.

$$
[n_{heads},T,d_{head}]
$$

becomes:

$$
[T,d_{model}]
$$

because:

$$
d_{model}
=
n_{heads}
\times
d_{head}
$$

---

# 10. Output Projection

Mix information from all heads using a learned projection.

$$
AttentionOutput
=
CombinedHeads
\times
W_O
$$

Shape:

$$
[T,d_{model}]
\rightarrow
[T,d_{model}]
$$

---

# 11. Attention Residual Connection

Add the original input back.

$$
Y
=
X
+
AttentionOutput
$$

Therefore:

$$
\boxed{
Y
=
X
+
MHA(RMSNorm(X))
}
$$

This completes the Attention sub-layer.

---

# 12. RMSNorm Before FFN

Normalize the output from the Attention sub-layer.

$$
Y_{norm}
=
RMSNorm(Y)
$$

Shape remains:

$$
[T,d_{model}]
$$

---

# 13. Up Projection

Expand the representation into a larger FFN dimension.

$$
Up
=
Y_{norm}W_{up}
$$

Shape:

$$
[T,d_{model}]
\rightarrow
[T,d_{ff}]
$$

---

# 14. Gate Projection and SiLU

Create a second projection.

$$
GateRaw
=
Y_{norm}W_{gate}
$$

Apply SiLU:

$$
Gate
=
SiLU(GateRaw)
$$

Where:

$$
SiLU(x)
=
x
\cdot
Sigmoid(x)
$$

and:

$$
Sigmoid(x)
=
\frac{1}{1+e^{-x}}
$$

---

# 15. SwiGLU Gating

Multiply the Up and Gate paths element-wise.

$$
Hidden
=
Up
\odot
Gate
$$

Therefore:

$$
\boxed{
Hidden
=
(Y_{norm}W_{up})
\odot
SiLU(Y_{norm}W_{gate})
}
$$

Shape:

$$
[T,d_{ff}]
$$

---

# 16. Down Projection

Project the expanded representation back to the model dimension.

$$
FFNOutput
=
Hidden
\times
W_{down}
$$

Shape:

$$
[T,d_{ff}]
\rightarrow
[T,d_{model}]
$$

The complete FFN is:

$$
FFN(X)
=
\left[
(XW_{up})
\odot
SiLU(XW_{gate})
\right]
W_{down}
$$

---

# 17. FFN Residual Connection

Add the input of the FFN sub-layer back.

$$
Z
=
Y
+
FFNOutput
$$

Therefore:

$$
\boxed{
Z
=
Y
+
FFN(RMSNorm(Y))
}
$$

This completes one Transformer block.

---

# Complete Transformer Block

## Attention Sub-Layer

$$
\boxed{
Y
=
X
+
MHA(RMSNorm(X))
}
$$

where:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

and:

$$
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}
{\sqrt{d_{head}}}
+
Mask
\right)V
$$

---

## SwiGLU FFN Sub-Layer

$$
\boxed{
Z
=
Y
+
FFN(RMSNorm(Y))
}
$$

where:

$$
FFN(X)
=
\left[
(XW_{up})
\odot
SiLU(XW_{gate})
\right]
W_{down}
$$

---

# Final Flow

```text
X
│
├── RMSNorm
│
├── Q, K, V
│
├── Split Heads
│
├── QKᵀ / √d_head
│
├── Causal Mask
│
├── Softmax
│
├── Attention Weights × V
│
├── Combine Heads
│
├── Output Projection W_O
│
├── + X
│
▼
Y
│
├── RMSNorm
│
├── Up Projection
│
├── Gate Projection → SiLU
│
├── Up × Gate
│
├── Down Projection
│
├── + Y
│
▼
Z
```

# Shape

The Transformer block preserves the overall shape:

$$
\boxed{
[T,d_{model}]
\rightarrow
[T,d_{model}]
}
$$