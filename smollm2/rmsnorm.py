import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input across the hidden dimension
    and applies a learnable scaling parameter.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(
            (x ** 2).mean(-1, keepdim=True) + self.eps
        )
        return (x / rms) * self.gamma

if __name__ == "__main__":
    x = torch.tensor([
        [2.0, 4.0, 6.0, 8.0]
    ])
    # x = torch.tensor([
    #     [
    #         [2.0, 4.0, 6.0, 8.0],
    #         [1.0, 3.0, 5.0, 7.0],
    #         [4.0, 3.0, 2.0, 1.0]
    #     ],
    #     [
    #         [3.0, 6.0, 9.0, 12.0],
    #         [2.0, 5.0, 8.0, 11.0],
    #         [7.0, 4.0, 2.0, 1.0]
    #     ]
    # ])
    # print(x.shape)
    norm = RMSNorm(x.shape[-1])
    output = norm(x)

    print(output)
    # print(
    #     torch.sqrt(
    #         (output ** 2).mean(-1)
    #     )
    # )