import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.token_ids = token_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, idx):
        input_ids = self.token_ids[idx:idx+self.seq_length]
        target_ids = self.token_ids[idx+1:idx+1+self.seq_length]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

if __name__ == "__main__":

    # token_ids = [
    #     10, 20, 30, 40,
    #     50, 60, 70, 80
    # ]
    with open('data/sample.txt', 'r') as file:
        text = file.read()
    tokenizer = Tokenizer()
    tokenizer.build_vocab(text)
    token_ids = tokenizer.encode(text)

    print("Vocabulary size:", len(tokenizer.token_to_id))
    print("Total token IDs:", len(token_ids))

    dataset = TextDataset(
        token_ids=token_ids,
        seq_length=4
    )

    # print("Dataset length:", len(dataset))

    # print("\nFirst example:")
    # print(dataset[1])

    # print("\nSecond example:")
    # # print(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False
        )

    for batch in dataloader:
        print(batch)
        break