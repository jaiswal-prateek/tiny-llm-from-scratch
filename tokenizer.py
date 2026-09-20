import re
import json

class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        # Simple whitespace and punctuation tokenizer
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text)

        return tokens

    def build_vocab(self, text):
        tokens = self.tokenize(text)
        vocab = ['<pad>', '<unk>', '<eos>'] + sorted(set(tokens))
        self.token_to_id = {token:idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx:token for idx, token in enumerate(vocab)}

    def save(self, path):
        with open(path, "w") as file:
            json.dump(self.token_to_id, file)

    def load(self, path):
        with open(path, "r") as file:
            self.token_to_id = json.load(file)

        self.id_to_token = {
            int(idx): token
            for token, idx in self.token_to_id.items()
        }

    def encode(self, text):
        tokens = self.tokenize(text)

        unk_id = self.token_to_id['<unk>']

        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.id_to_token[idx] for idx in token_ids]

        return ' '.join(tokens).replace(' .', '.')

if __name__ == "__main__":
    tokenizer = Tokenizer()

    # with open('data/sample.txt', 'r') as file:
    #     text = file.read()
    text = "Hello world. This is a test."
    tokenizer.build_vocab(text)
    
    encoded = tokenizer.encode('hello world. this')
    decoded = tokenizer.decode(encoded)

    print(f"Text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")