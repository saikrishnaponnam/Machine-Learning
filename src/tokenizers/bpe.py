import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


def _merge_pair(pair: Tuple[str, str], word: List[str]) -> List[str]:
    """
    Merge all occurrences of a given pair in a word.
    Input:
        pair: A pair of tokens to merge, e.g., ('tes', 't').
        word: A list of tokens representing the word.
    Returns:
        A new list of tokens with the pair merged.
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


class BPETrainer:
    """
    Byte Pair Encoding (BPE) Trainer for building a BPE tokenizer.
    """

    def __init__(self, vocab_size: int = 10000, unk_token="<unk>", special_tokens=None):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.merges = []
        self.token2id = {}

    def _get_stats(self, vocab: Counter) -> Dict[Tuple[str, str], int]:
        """
        Get the frequency of all adjacent token pairs in the vocabulary.
        Input:
            vocab: A Counter object where keys are word tokens and values are their frequencies.
        Returns:
            A dictionary where keys are pairs of tokens (tuples) and values are their frequencies.
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Counter) -> Counter:
        """
        Merge all occurrences of a given pair in the vocabulary.
        Input:
            pair: A pair of tokens to merge, e.g., ('tes', 't').
            vocab: A Counter object where keys are word tokens and values are their frequencies.
        """
        new_vocab = Counter()
        for word, freq in vocab.items():
            new_word = _merge_pair(pair, word)
            new_vocab[tuple(new_word)] += freq
        return new_vocab

    def build_token_mappings(self, vocab: Counter):
        """
        Build token to ID mappings from the vocabulary.
        """
        tokens = set()
        for word in vocab:
            tokens.update(word)
        for merge in self.merges:
            tokens.add("".join(merge))
        tokens = list(tokens)
        if self.unk_token not in tokens:
            tokens.append(self.unk_token)
        if self.special_tokens:
            for token in self.special_tokens:
                if token not in tokens:
                    tokens.append(token)

        self.token2id = {tok: i for i, tok in enumerate(tokens)}

    def train(self, corpus: List[str]):
        """
        Train the BPE tokenizer on the provided corpus.
        Input:
            corpus: A list of strings, where each string is a line of text.
        """
        # Split corpus into character-level tokens
        tokens = [list(word) + ["</w>"] for line in corpus for word in line.strip().split()]

        # Initialize vocabulary with characters
        vocab = Counter([tuple(token) for token in tokens])

        # Perform BPE merges
        merges = []
        while len(merges) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            merges.append(best)

        self.merges = merges
        self.build_token_mappings(vocab)

    def save(self, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the vocabulary and merges to files.
        Input:
            filename_prefix: Optional prefix for the output files.
        """
        vocab_file = os.path.join((filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join((filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f)
        with open(merge_file, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for merge in self.merges:
                f.write(" ".join(merge) + "\n")

        return vocab_file, merge_file


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer for tokenizing text using a pre-trained BPE model.
    """

    def __init__(self, vocab_file, merges_file, unk_token: str = "<unk>", special_tokens=None):
        """
        Initialize the BPE tokenizer with vocabulary and merges files.
        Input:
            vocab_file: Path to the vocabulary file (JSON format).
            merges_file: Path to the merges file (text format).
            unk_token: Token to use for unknown words.
            special_tokens: List of special tokens to include in the vocabulary.
        """
        self.unk_token = unk_token
        self.special_tokens = special_tokens
        self.merges = []
        self.token2id = {}
        self.id2token = {}

        with open(vocab_file, encoding="utf-8") as fp:
            self.token2id = json.load(fp)
        self.id2token = {v: k for k, v in self.token2id.items()}
        with open(merges_file, encoding="utf-8") as fp:
            merges = fp.read().split("\n")[1:]
        self.merges = [tuple(line.split()) for line in merges]

    def get_vocab(self) -> Dict[str, int]:
        return self.token2id

    def tokenize(self, word: str) -> List[str]:
        """
        Tokenize a word using the BPE algorithm.
        Input:
            word: A string representing the word to tokenize.
        Returns:
            A list of tokens representing the word after applying BPE.
        """
        word = list(word) + ["</w>"]
        pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        while True:
            min_pair = None
            for merge in self.merges:
                if merge in pairs:
                    min_pair = merge
                    break
            if not min_pair:
                break
            word = _merge_pair(min_pair, word)
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        if word[-1] == "</w>":
            word = word[:-1]
        return word

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of token IDs using the BPE tokenizer.
        Input:
            text: A string representing the text to encode.
        Returns:
            A list of integers representing the token IDs.
        """
        words = text.strip().split()
        tokens = []
        for word in words:
            word_tokens = self.tokenize(word)
            tokens.extend([self.token2id.get(t, self.token2id[self.unk_token]) for t in word_tokens])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.
        Input:
            token_ids: A list of integers representing the token IDs to decode.
        Returns:
            A string representing the decoded text.
        """
        tokens = [self.id2token.get(i, self.unk_token) for i in token_ids]
        text = ""
        for t in tokens:
            text += t
        return text.replace("</w>", " ").strip()


class BPEHFTokenizer(PreTrainedTokenizer):
    """
    Hugging Face compatible BPE tokenizer that uses a pre-trained BPE model.
    """

    def __init__(self, vocab_file, merges_file, unk_token: str = "<unk>", special_tokens=None):
        super().__init__(unk_token=unk_token, pad_token=None, eos_token=None, bos_token=None)
        self.bpe_tokenizer = BPETokenizer(vocab_file, merges_file, unk_token, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.bpe_tokenizer.token2id)

    @property
    def get_vocab(self) -> Dict[str, int]:
        return self.bpe_tokenizer.token2id

    def _tokenize(self, text: str) -> List[str]:
        return self.bpe_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.bpe_tokenizer.token2id.get(token, self.bpe_tokenizer.token2id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.bpe_tokenizer.id2token.get(index, self.bpe_tokenizer.unk_token)
