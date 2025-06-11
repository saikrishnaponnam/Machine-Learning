# Tokenization

[BPE](https://arxiv.org/abs/1508.07909)
| [WordPiece](https://arxiv.org/abs/1609.08144) | [Fast WordPiece](https://arxiv.org/abs/1808.06226)
| [SentencePiece](https://arxiv.org/abs/1808.06226)

Any ML model that processes text data requires the text to be converted into a numerical format.
Tokenization is the process of breaking up text into smaller units called tokens. These tokens are then mapped to numerical representations that can be fed into machine learning models.
These tokens can be words, subwords, or even characters, depending on the task and the tokenizer used.
Tokenization is often the first step in NLP pipelines, as it allows models to work with manageable pieces of language.
Despite being often overlooked, tokenization decisions affect model accuracy and training efficiency.

This section dives into what tokenization is, common techniques, challenges, tools widely used in the NLP ecosystem and training a custom tokenizers.

## Why Tokenization?
Tokenization acts as a bridge between raw text and computational models. Without it, machines would not understand how to segment language into parts they can learn from.
It is essential for several reasons:

1. **Text Representation**: It converts raw text into a structured format that can be processed by machine learning models.
2. **Standardization**: Natural language is highly variable, with different ways to express the same meaning. Tokenization helps standardize this variability.
3. **Efficiency**: Tokenization reduces the dimensionality of text data, making it easier to process and analyze.


## Types of Tokenization
1. **Word Tokenization**: Splits text into words. This is the simplest form of tokenization, but it can struggle with out-of-vocabulary words and does not handle subword information.
   - Example: "I love NLP" → ["I", "love", "NLP"]

2. **Character Tokenization**: Treats each character as a token. This is useful for languages with complex morphology or when dealing with noisy text. In some languages, it fails to capture meaningful units.
    - Example: "NLP" → ["N", "L", "P"]

3. **Subword Tokenization**: Breaks words into smaller units, allowing the model to handle rare words and out-of-vocabulary terms more effectively.
    - Example: "unhappiness" → ["un", "happiness"]
    - Common algorithms include Byte-Pair Encoding (BPE), WordPiece and SentencePiece.

4. **Sentence Tokenization**: Splits text into sentences, which can be useful for tasks like summarization or question answering.
    - Example: "I love NLP. It's fascinating!" → ["I love NLP.", "It's fascinating!"]

## Tokenization in Practice
Tokenization is typically performed using libraries like NLTK, SpaCy, or Hugging Face's Transformers.
These libraries provide pre-built tokenizers that can handle various languages and tokenization strategies.
Some commonly used tokenizers include:

### BPE Tokenizer

Byte-Pair Encoding (BPE) is a subword tokenization technique that iteratively merges the most frequent pairs of characters or subwords in a corpus until a predefined vocabulary size is reached.

- **Pros**: Handles out-of-vocabulary words, reduces vocabulary size, and captures subword information.
- **Cons**: Can lead to longer sequences, and the merging process can be computationally expensive.
- Example: "unhappiness" might be tokenized as ["un", "happi", "ness"].


### WordPiece Tokenizer

WordPiece is another subword tokenization technique that splits words into subwords based on their frequency in the training corpus.

- **Pros**: Similar to BPE, it handles out-of-vocabulary words and captures subword information.
- **Cons**: Can also lead to longer sequences, and the vocabulary size is fixed during training.
- Example: "unhappiness" → ["un", "happiness"]

### SentencePiece Tokenizer
SentencePiece is a data-driven tokenizer that can be used for both word and subword tokenization. It treats the input text as a sequence of Unicode characters and learns a vocabulary based on the frequency of character sequences.

- **Pros**: Language-agnostic, can handle multiple languages, and does not require pre-tokenization.
- **Cons**: Requires training on a large corpus, and the resulting tokens may not align with natural language boundaries.
- Example: "unhappiness" → ["un", "happi", "ness"]

## Training Tokenizers
If a language model is not available in the language you are interested in, or if your corpus is very different from the one your language model was trained on, you will most likely want to retrain the model from scratch using a tokenizer adapted to your data.

We can use hugging face's `tokenizers` library to train a tokenizer on our own corpus. The process involves the following steps:

1. **Prepare the Corpus**: Collect a large text corpus in the target language or domain.
2. **Choose a Tokenization Algorithm**: Select a tokenization algorithm (e.g., BPE, WordPiece, SentencePiece) based on your requirements.
3. **Load the tokenizer**: We even though we are training a tokenizer from scratch, we can use the `tokenizers` library to load a pre-existing tokenizer configuration. This will keep the tokenizer compatible with the model and only vocabulary is changed.
4. **Train the Tokenizer**: Use the `train` method to train the tokenizer on the prepared corpus.


## Challenges in Tokenization
1. **Ambiguity**: Some words can be tokenized in multiple ways, leading to ambiguity. For example, "New York" can be treated as a single token or two separate tokens.
2. **Out-of-Vocabulary Words**: Tokenizers may struggle with words that are not in their vocabulary, leading to unknown tokens or subword splits.
3. **Language Variability**: Different languages have different tokenization rules, and a tokenizer that works well for one language may not perform well for another.
4. **No Whitespace Languages**: Some languages, like Chinese and Japanese, do not use whitespace to separate words, making tokenization more challenging.
5. **Punctuation and Special Characters**: Deciding how to handle punctuation and special characters can affect the tokenization process. Some tokenizers may remove punctuation, while others may keep it as separate tokens.
6. **Contextual Meaning**: Some words can have different meanings based on context, and tokenization does not capture this semantic information. For example, "bank" can refer to a financial institution or the side of a river.


## Best Practices
1. **Choose the Right Tokenizer**: Select a tokenizer that aligns with your task and data. For example, use subword tokenization for tasks involving rare words or out-of-vocabulary terms.
2. **Preprocess Text**: Clean and preprocess your text data before tokenization. This may include lowercasing, removing special characters, or handling contractions.
3. **Unknown tokens**: Tokens that are not in the tokenizer's vocabulary can be replaced with a special "unknown" token, represented as "\<unk\>" or "[UNK]".
4. **Batching**: When processing sequences, ensure that all sequences in a batch are of the same length. This can be achieved by padding shorter sequences with a special "padding" token, represented as "\<pad\>" or "[PAD]".

## Q&A


