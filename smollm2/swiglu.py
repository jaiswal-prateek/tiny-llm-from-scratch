import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Projects the input through up and gate paths,
    applies SiLU to the gate, then projects back.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        hidden = up * gate
        output = self.down_proj(hidden)
        return output

if __name__ == "__main__":
    x = torch.randn(2,3,4)

    swiglu = SwiGLU(x.shape[-1], x.shape[-1] * 4)
    output = swiglu(x)
    print(f"SwiGLU output shape: {output.shape}")
    # print(f"Input: {x}")
    # print(f"Output: {output}")