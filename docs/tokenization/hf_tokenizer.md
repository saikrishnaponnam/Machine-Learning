# HF Tokenizer

The Hugging Face Tokenizer library is a powerful and fast tool for text tokenization, widely used in NLP pipelines. 
It is the backbone of many popular models in the Hugging Face Transformers ecosystem, providing efficient and customizable tokenization for a variety of languages and tasks.

## Why Use Hugging Face Tokenizers?
- **Speed**: Written in Rust, the library is extremely fast and efficient.
- **Flexibility**: Supports multiple tokenization algorithms (BPE, WordPiece, Unigram, etc.).
- **Customizability**: Allows users to train their own tokenizers or use pre-trained ones.
- **Integration**: Seamlessly integrates with Hugging Face Transformers and datasets.

## Key Features
- Pre-trained tokenizers for popular models (BERT, GPT, RoBERTa, etc.)
- Support for special tokens (CLS, SEP, PAD, etc.)
- Fast batch encoding/decoding

## Installation
You can install the library using pip:

```bash
pip install tokenizers
```

## Basic Usage Example
Here's a simple example using a pre-trained tokenizer from Hugging Face Transformers:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = "Hugging Face's Tokenizers are awesome!"

# Tokenize text
tokens = tokenizer.tokenize(text)
print(tokens) 
# Output: ['hugging', 'face', "'", 's', 'token', '##izer', '##s', 'are', 'awesome', '!']


# Encode text to input IDs
input_ids = tokenizer.encode(text)
print(input_ids) 
# Output:  [101, 17662, 2227, 1005, ... , 999, 102]


# Decode input IDs back to text
decoded = tokenizer.decode(input_ids)
print(decoded) 
# Output:  [CLS] hugging face ' s tokenizers are awesome! [SEP]
```

## Tokenization Pipeline
The tokenizer pipeline consists of the following components:


### Normalization
Normalization is the process of converting text into a standard format, such as lowercasing, removing accents, etc.
Normalization operations in the Tokenizers library are represented by a Normalizer. You can combine multiple operations using `normalizers.Sequence`.
```python
from tokenizers import normalizers
normalizer = normalizers.Sequence([
    normalizers.NFD(),  # Decompose characters
    normalizers.Lowercase(),  # Convert to lowercase
    normalizers.StripAccents()  # Remove accents
])
```

### Pre-tokenization
The pre-tokenizer is responsible for splitting the input text into words and then actual tokenization happens on these words.
You can combine any PreTokenizer together. For instance, here is a pre-tokenizer that will split on space, punctuation and digits, separating numbers in their individual digits:

```python
from tokenizers import pre_tokenizers
pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),  # Split on whitespace
    pre_tokenizers.Punctuation(),  # Split on punctuation
    pre_tokenizers.Digits()  # Split digits into individual tokens
])
```

### Model
The model defines how the tokens are represented in the vocabulary. The most common models are BPE (Byte Pair Encoding), WordPiece, and Unigram.
The role of the model is to split your “words” into tokens, using the rules it has learned. It also maps these tokens to their corresponding IDs in the vocabulary.

```python
from tokenizers import models
model = models.BPE()  # Using Byte Pair Encoding
```

### Post-processing
Post-processing is the final step where the tokenized output is converted into a format suitable for model input, such as adding special tokens, padding, and truncation.
```python
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)
```


All the components that can be used to build tokenizer are listed [here](https://huggingface.co/docs/tokenizers/components).


## Training a Custom Tokenizer
You can also train your own tokenizer using the `tokenizers` library:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Prepare training data
files = ["path/to/text.txt"]

# Train the tokenizer
trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("custom-tokenizer.json")
```

## Batch Encoding and Decoding
You can efficiently encode and decode batches of texts:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
texts = ["Hello world!", "Tokenization is fun."]

# Batch encode
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(batch['input_ids'])

# Batch decode
decoded = tokenizer.batch_decode(batch['input_ids'])
print(decoded)
```



## Resources
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
