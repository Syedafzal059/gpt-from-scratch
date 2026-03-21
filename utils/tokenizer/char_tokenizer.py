
class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids):
        return "".join(self.itos[i] for i in token_ids)




# ------------------ TEST ------------------

if __name__ == "__main__":
    sample_text = "hello world"

    tokenizer = CharTokenizer(sample_text)

    encoded = tokenizer.encode("hello")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    print("Vocab:", tokenizer.chars)
    print("Vocab size:", tokenizer.vocab_size)