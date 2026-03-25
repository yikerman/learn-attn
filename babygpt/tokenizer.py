"""Character-level tokenizer for TinyShakespeare."""


class CharTokenizer:
    """Maps individual characters to integers and back.

    This is the simplest possible tokenizer: the vocabulary is just the set of
    unique characters in the training text. Each character gets a unique integer
    ID. There are no subword units, no byte-pair encoding, no special tokens.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.char_to_index = {ch: i for i, ch in enumerate(chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_index)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_index[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.index_to_char[token] for token in tokens)
