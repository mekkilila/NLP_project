import re
from collections import Counter, defaultdict

class CustomTokenizer:
    # the size of the vocabulary is 5000, and we skip the 50 most frequent tokens 
    def __init__(self, max_vocab_size=5000, skip_top=50):
        self.max_vocab_size = max_vocab_size
        self.skip_top = skip_top
        self.vocab = {}
        self.token_freqs = Counter() # for frequency
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1}

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"<br\s*/?>", " ", text)  # replace <br> or <br /> with space
        text = re.sub(r"<[^>]+>", " ", text)   # remove any remaining HTML tags
        text = re.sub(r'([!?.,:;()"])', r' \1 ', text)  # isolate punctuation 
        text = re.sub(r"\s+", " ", text)  # clean multiple spaces
        return text.strip().split()

    def build_vocab(self, texts):
        # count all tokens
        for text in texts:
            tokens = self.tokenize(text)
            self.token_freqs.update(tokens)

        # build vocab
        most_common = self.token_freqs.most_common(self.skip_top + self.max_vocab_size)
        filtered_tokens = most_common[self.skip_top:]  # to skip the skip_top first tokens 

        # start vocab with special tokens for padding and unknown
        self.vocab = dict(self.special_tokens)
        for idx, (token, _) in enumerate(filtered_tokens, start=len(self.vocab)):
            self.vocab[token] = idx 

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]