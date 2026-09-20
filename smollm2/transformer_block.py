import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import Attention
from .swiglu import SwiGLU

class TransformerBlock(nn.Module):
    """
    Transformer decoder block.

    Applies:
    RMSNorm → Attention → Residual
    RMSNorm → SwiGLU → Residual
    """

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.input_rmsnorm = RMSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads)

        self.output_rmsnorm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False):
        residual = x
        x = self.input_rmsnorm(x)

        if use_cache:
            x, present_key_value = self.attention(x, past_key_value= past_key_value, use_cache=True)
        else:
            x = self.attention(x)
        
        x = residual + x

        residual = x
        x = self.output_rmsnorm(x)
        x = self.ffn(x)
        x = residual + x

        if use_cache:
            return x, present_key_value

        return x
        

if __name__ == "__main__":
    x = torch.randn(1, 4, 8)

    block = TransformerBlock(8, 2, 8*4)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
