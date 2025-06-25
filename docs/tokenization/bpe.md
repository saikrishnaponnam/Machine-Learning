# BPE Tokenization

[Code](https://github.com/saikrishnaponnam/Machine-Learning/blob/main/src/tokenizers/bpe.py)

Byte Pair Encoding (BPE) is a lossless data compression technique that was originally used for compressing text data.
The basic idea of BPE is to iteratively merge the most frequent adjacent pairs of bytes (or characters) in a corpus to form a new, longer tokens. This process continues until a predefined vocabulary size is reached or no more pairs can be merged.
It starts with a base vocabulary of individual characters and builds up to larger subword units by merging the most common pairs.

In the context of NLP, BPE tokenization is used to split text into subwords, which are contiguous sequences of characters within a word.
This allows for efficient handling of rare words and out-of-vocabulary terms by breaking them down into smaller, more manageable pieces. This enables models to capture meaningful patterns and relationships in text data.

It’s used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.

## Algorithm

1. **Initialization**: Start with a vocabulary of all unique characters in the training corpus.
2. **Count Pairs**: Count the frequency of all adjacent pairs of characters in the corpus.
3. **Merge**: Identify the most frequent pair of characters and merge them into a new token.
4. **Update Vocab**: Add the new token to the vocabulary and update the corpus by replacing all occurrences of the merged pair with the new token.
5. **Repeat**: Repeat steps 2-4 until the desired vocabulary size is reached or no more pairs can be merged.

The resulting vocabulary contains a mix of individual characters, subwords, and whole words. When tokenizing new text, the BPE algorithm splits the text into the most likely subwords based on the learned vocabulary.

## Advantages
- **Handling Rare Words**: BPE effectively reduces the out-of-vocabulary (OOV) problem by breaking down rare words into subwords, allowing models to handle previously unseen terms.
- **Smaller Vocab**: BPE reduces the vocabulary size compared to traditional word-based tokenization, which can lead to more efficient model training and inference.
- **Language Agnostic**: BPE can be applied to any language, making it a versatile tokenization method.

## Disadvantages
- **Suboptimal for some languages**: BPE can produce awkward splits in some scripts (e.g., Chinese).
- **Not linguistically informed**: Merges are frequency-based, not semantic.


## Best Practices
- **Vocabulary Size**: Choose an appropriate vocabulary size based on the dataset and model requirements. A larger vocabulary can capture more nuances but may lead to increased computational costs.
- **Preprocessing**: Preprocessing techniques, such as token normalization and stopword removal, can improve the effectiveness of BPE tokenization.

## Limitations
- **Out-of-Vocabulary (OOV) Handling**: While BPE reduces OOV issues, it may still struggle with extremely rare or domain-specific terms.
- **Morphological Variations**: BPE tokenization may not effectively capture morphological variations, such as prefixes and suffixes, which can be important in certain languages.
- **Language Dependence**: BPE tokenization may not be equally effective across different languages, especially those with complex morphology or rich inflectional systems.