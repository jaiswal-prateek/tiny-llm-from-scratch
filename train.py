import torch
import json

from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from smollm2.dataset import TextDataset
from smollm2.model import SmolLM2

# Configuration
from config import (
    HIDDEN_SIZE,
    NUM_HEADS,
    INTERMEDIATE_SIZE,
    NUM_LAYERS,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    SEQ_LEN
)

# Load training data
with open("data/sample.txt", "r") as file:
    text = file.read()

# Build tokenizer
tokenizer = Tokenizer()
tokenizer.build_vocab(text)
tokenizer.save("tokenizer.json")

eos_id = tokenizer.token_to_id["<eos>"]

token_ids = []

for line in text.splitlines():
    if line.strip():
        token_ids.extend(tokenizer.encode(line))
        token_ids.append(eos_id)

VOCAB_SIZE = len(tokenizer.token_to_id)

# Build dataset
dataset = TextDataset(
    token_ids=token_ids,
    seq_length=SEQ_LEN
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Build model
model = SmolLM2(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    num_layers=NUM_LAYERS
)

# Training setup
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Training
model.train()

for epoch in range(NUM_EPOCHS):

    total_loss = 0

    # Forward pass
    for batch in dataloader:

        # get batch
        input_ids = batch['input_ids']
        targets = batch['target_ids']

        # Forward pass
        logits = model(input_ids)

        # CrossEntropyLoss expects:
        # [batch, vocab_size, seq_len]
        logits = logits.transpose(1, 2)

        # Calculate loss
        loss = loss_fn(logits, targets)

        # Clear old gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Add batch loss
        total_loss += loss.item()

    # Average loss across all batches
    average_loss = total_loss / len(dataloader)

    # Print progress
    if epoch % 50 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Loss: {average_loss:.4f}"
        )

# Save Model
torch.save(
    model.state_dict(),
    "smollm2_model.pt"
)

# Save model configuration
model_config = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_heads": NUM_HEADS,
    "intermediate_size": INTERMEDIATE_SIZE,
    "num_layers": NUM_LAYERS
}

with open("model_config.json", "w") as file:
    json.dump(model_config, file, indent= 4)

print("\nModel and configuration saved successfully.")