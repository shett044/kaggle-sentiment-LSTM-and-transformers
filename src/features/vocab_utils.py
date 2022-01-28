import torch
from torchtext.vocab import build_vocab_from_iterator


class VocabFactory:
    def __init__(self, specials: list, tokenizer=None):
        self.specials = specials if len(specials) > 0 else ['<unk>']
        if tokenizer is None:
            # self.tokenizer = get_tokenizer('basic_english')
            tokenizer = lambda x: x.split()
        self.tokenizer = tokenizer
        self.vocab = None

    def build_vocab(self, d_iter, save_vocab_file='models/vocab_obj.pth'):
        def yield_tokens(data_iter):
            for X, y in data_iter:
                yield self.tokenizer(X)

        vocab = build_vocab_from_iterator(yield_tokens(d_iter), specials=self.specials, min_freq=2)
        vocab.set_default_index(0)
        self.vocab = vocab
        self.save_vocab(save_vocab_file)

    def transform(self, X: str):
        if self.vocab is not None:
            return self.vocab(self.tokenizer(X))

    def save_vocab(self, vocab_file):
        torch.save(self.vocab, vocab_file)

    def get_vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def load_vocab(vocab_file, specials):
        vocab = torch.load(vocab_file)
        v = VocabFactory(specials)
        v.vocab = vocab
        return v
