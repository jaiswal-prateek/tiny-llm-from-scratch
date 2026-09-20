import torch
import torch.nn as nn
import math

from .rope import RoPE

class Attention(nn.Module):
    """
    Multi-head self-attention.

    Projects input into Q, K, and V,
    applies RoPE to Q and K,
    then performs causal self-attention.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias= False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias= False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias= False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias= False)

        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor, past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False):
        """
        Input X
        [batch, seq_len, hidden_size]
                ↓
        Q/K/V Projection
        [batch, seq_len, hidden_size]
                ↓
        view
        [batch, seq_len, num_heads, head_dim]
                ↓
        transpose(1, 2)
        [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        # q = self.rope(q)

        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)
        # k = self.rope(k)

        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)

        past_len = 0

        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2) # past_k = [batch, num_heads, past_len or sequence, head_dim]

        q = self.rope(q, position_offset= past_len)
        k = self.rope(k, position_offset= past_len)

        if past_key_value is not None:
            k = torch.cat([past_k, k], dim= 2)
            v = torch.cat([past_v, v], dim= 2)

        scores = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim))

        total_len = k.size(2)

        # for logging how kv cache works when run generate.py
        # if use_cache:
        #         print(
        #         f"Attention | input_tokens={seq_len} | "
        #         f"past_tokens={past_len} | "
        #         f"total_KV_tokens={total_len}"
        #         )

        query_positions = torch.arange(past_len, past_len + seq_len, device= x.device)
        key_positions = torch.arange(total_len, device= x.device)

        causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
        masked = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = masked.softmax(dim=-1)
        attn_output = attn_weights @ v
        output = self.o_proj(attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size))

        if use_cache:
            return output, (k, v)
        
        return output

if __name__ == "__main__":
    x = torch.randn(1, 4, 8)

    att = Attention(8, 2)
    output = att(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)