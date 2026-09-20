import torch
import json

from tokenizer import Tokenizer
from smollm2.model import SmolLM2


# Load model configuration
with open("model_config.json", "r") as file:
    model_config = json.load(file)


# Load tokenizer
tokenizer = Tokenizer()
tokenizer.load("tokenizer.json")

eos_id = tokenizer.token_to_id["<eos>"]

# Recreate model
model = SmolLM2(
    vocab_size=model_config["vocab_size"],
    hidden_size=model_config["hidden_size"],
    num_heads=model_config["num_heads"],
    intermediate_size=model_config["intermediate_size"],
    num_layers=model_config["num_layers"]
)


# Load trained weights
model.load_state_dict(
    torch.load("smollm2_model.pt")
)

model.eval()

def apply_top_k(logits, top_k):
    if top_k is None:
        return logits

    top_k = min(top_k, logits.size(-1))

    top_k_values, _ = torch.topk(
        logits,
        top_k
    )

    kth_value = top_k_values[:, -1].unsqueeze(-1)

    return torch.where(
        logits < kth_value,
        torch.full_like(logits, float("-inf")),
        logits
    )

def apply_top_p(logits, top_p):
    if top_p is None:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )

    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    sorted_remove = cumulative_probs > top_p

    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False

    remove_mask = torch.zeros_like(
        sorted_remove,
        dtype=torch.bool
    )

    remove_mask.scatter_(
        1,
        sorted_indices,
        sorted_remove
    )

    return logits.masked_fill(
        remove_mask,
        float("-inf")
    )

@torch.no_grad()
def generate(
            prompt,
            max_new_tokens=10,
            temperature=1.0,
            top_k=None,
            top_p=0.9
    ):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")

    generated_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    # Prefill: process the complete prompt once
    logits, past_key_values = model(
        generated_ids,
        use_cache=True
    )

    for _ in range(max_new_tokens):

        next_token_logits = logits[:, -1, :]

        # Temperature
        next_token_logits = (
            next_token_logits / temperature
        )

        # Top-k
        next_token_logits = apply_top_k(
            next_token_logits,
            top_k
        )

        # Top-p
        next_token_logits = apply_top_p(
            next_token_logits,
            top_p
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # Sample next token
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Stop generation
        if next_token_id.item() == eos_id:
            break

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1
        )

        # Decode only the newly generated token
        logits, past_key_values = model(
            next_token_id,
            past_key_values=past_key_values,
            use_cache=True
        )

    return tokenizer.decode(
        generated_ids[0].tolist()
    )

# Generation settings
prompt = "did maya read the report"
max_new_tokens = 10
temperature = 1.0
top_k = None
top_p = 0.9

generated_text = generate(
    prompt=prompt,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p
)

print("\nPrompt:")
print(prompt)

print("\nTemperature:", temperature)
print("Top K:", top_k)
print("Top P:", top_p)

print("\nGenerated:")
print(generated_text)