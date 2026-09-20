import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock

class SmolLM2(nn.Module):
    """
    SmolLM2 language model.

    Consists of an embedding layer, multiple transformer blocks,
    and a final RMSNorm and linear projection.
    Token IDs → Embedding → Transformer Blocks → RMSNorm → LM Head → Logits
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, intermediate_size: int, num_layers: int):
        super().__init__()

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
                    [TransformerBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
                    )
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None = None, use_cache: bool = False):
        x = self.embed_tokens(input_ids)

        present_key_values = []

        for layer_idx, layer in enumerate(self.layers):
            past_key_value = None

            if past_key_values is not None:
                past_key_value = past_key_values[layer_idx]

            if use_cache:
                x, present_key_value = layer(x, past_key_value=past_key_value, use_cache=use_cache)
                present_key_values.append(present_key_value)
            else:
                x = layer(x)

        x = self.norm(x)
        x = self.lm_head(x)

        if use_cache:
            return x, tuple(present_key_values)

        return x

if __name__ == "__main__":

    torch.manual_seed(42)

    vocab_size = 100
    hidden_size = 8
    num_heads = 2
    intermediate_size = 32
    num_layers = 2

    model = SmolLM2(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers
    )

    model.eval()

    # Prompt: 4 tokens
    prompt_ids = torch.tensor([
        [10, 20, 30, 40]
    ])

    # Next token that we want to generate
    next_token_id = torch.tensor([
        [50]
    ])

    with torch.no_grad():

        # 1. PREFILL: process full prompt + build cache

        prompt_logits, past_key_values = model(
            prompt_ids,
            use_cache=True
        )

        print("Prompt logits shape:", prompt_logits.shape)

        print("\nKV cache after prompt:")

        for layer_idx, (k, v) in enumerate(past_key_values):
            print(
                f"Layer {layer_idx}: "
                f"K={k.shape}, V={v.shape}"
            )

        # 2. DECODE: process only the new token

        cached_logits, updated_key_values = model(
            next_token_id,
            past_key_values=past_key_values,
            use_cache=True
        )

        print("\nNew token logits shape:", cached_logits.shape)

        print("\nKV cache after new token:")

        for layer_idx, (k, v) in enumerate(updated_key_values):
            print(
                f"Layer {layer_idx}: "
                f"K={k.shape}, V={v.shape}"
            )

        # 3. NORMAL: process all 5 tokens again

        full_ids = torch.cat(
            [prompt_ids, next_token_id],
            dim=1
        )

        full_logits = model(full_ids)

        print("\nFull-sequence logits shape:", full_logits.shape)

        # 4. Compare prediction for token 5

        cached_last_logits = cached_logits[:, -1, :]
        full_last_logits = full_logits[:, -1, :]

        max_difference = (
            cached_last_logits - full_last_logits
        ).abs().max().item()

        print("Cached argmax:", cached_last_logits.argmax(dim=-1))
        print("Normal argmax:", full_last_logits.argmax(dim=-1))
        print("\nMax difference:", max_difference)

        print(
            "Cached and normal outputs match:",
            torch.allclose(
                cached_last_logits,
                full_last_logits,
                atol=1e-6
            )
        )
