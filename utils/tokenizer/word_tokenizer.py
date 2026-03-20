import re 


class WordTokenizer:
    def __init__(self, text):
        # Split text into words
        self.words = self._tokenize(text)

        # Unique vocabulary
        self.vocab = sorted(list(set(self.words)))
        self.vocab.append("<UNK>")
        self.vocab_size = len(self.vocab)

        # Mappings
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for i, word in enumerate(self.vocab)}

    def _tokenize(self, text):
        # Split words and keep punctuation(Hello World! -> ["Hello", "World", "!"]) 
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text):
        tokens = self._tokenize(text)
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokens]

    def decode(self, tokens):
        return " ".join([self.itos[token] for token in tokens])


# ------------------ TEST ------------------

if __name__ == "__main__":
    sample_text = "Hello Afzal! This is GPT."

    tokenizer = WordTokenizer(sample_text)

    # Add UNK token (important)
    tokenizer.stoi["<UNK>"] = len(tokenizer.stoi)
    tokenizer.itos[len(tokenizer.itos)] = "<UNK>"

    encoded = tokenizer.encode("Hello GPT!")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    print("Vocab:", tokenizer.vocab)
    print("Vocab size:", tokenizer.vocab_size)
