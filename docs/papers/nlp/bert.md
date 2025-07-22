---
title: BERT
---
# BERT: Bidirectional Encoder Representations from Transformers

[Read the Paper](https://arxiv.org/abs/1810.04805) | [Google AI Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) | [Code (TBD)]()

BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary language model introduced by Google in 2018. Unlike earlier models that processed text in only one direction (left-to-right or right-to-left), BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. This enables BERT to capture richer language understanding and achieve state-of-the-art results on a wide range of NLP tasks, such as question answering, sentiment analysis, and named entity recognition.

## Introduction

Pre-training language models has proven highly effective for improving many NLP tasks, as demonstrated by works like [Dai and Le, 2015](https://arxiv.org/abs/1511.01432), [ELMo](elmo.md), [GPT 1](GPT.md), and [Howard and Ruder, 2018](https://arxiv.org/abs/1801.06146). These approaches benefit both sentence-level and token-level tasks. However, most previous models are unidirectional, processing text either left-to-right or right-to-left, which limits their ability to fully capture the context of each word.

There are two main ways to use pre-trained representations in downstream tasks:

- **Feature-based**: Use the pre-trained representations as additional features in a task-specific architecture (e.g., ELMo).
- **Fine-tuning**: Fine-tune the pre-trained model along with minimal task-specific parameters, allowing the model to adapt to the specific task.

Both approaches traditionally rely on unidirectional language models during pre-training, which can restrict the power of the learned representations. This limitation is especially problematic for fine-tuning approaches and for token-level tasks, where understanding context from both directions is crucial.

BERT overcomes these limitations by introducing a masked language model (MLM) objective. The MLM randomly masks some tokens in the input and trains the model to predict them using context from both directions. This allows BERT to learn deep bidirectional representations. Additionally, BERT uses a next sentence prediction (NSP) objective to help the model understand relationships between sentences, which is important for tasks like question answering and natural language inference.

---
## Architecture

BERT is built entirely on the Transformer encoder architecture. It consists of multiple layers of bidirectional Transformer encoders, enabling the model to capture context from both directions in a sentence. The original paper introduces two model sizes:

- **BERT-Base**: 12 layers (L=12), hidden size of 768 (H=768), and 12 self-attention heads (A=12).
- **BERT-Large**: 24 layers (L=24), hidden size of 1024 (H=1024), and 16 self-attention heads (A=16).

Notably, BERT-Base is designed to be comparable in size to GPT-1.

**Input and Output Representation:**  
To handle both single sentences and sentence pairs, BERT uses special tokens in its input sequence. The first token is always `[CLS]`, which is used for classification tasks. For sentence pairs, a `[SEP]` token separates the two sentences. The input representation is constructed by summing three types of embeddings: token embeddings, segment embeddings (to distinguish sentences), and positional embeddings (to encode word order). 

The combined embedding is denoted as $E$. The final hidden state of the `[CLS]` token is represented as $C$ and is typically used for classification tasks. The final hidden state of the $i^{th}$ token is denoted as $T_i$ and is used for token-level tasks.

---
## Training

BERT’s training process is divided into two main stages: pre-training and fine-tuning.

### Pre-training

During pre-training, BERT learns from large, unlabeled text corpora such as BooksCorpus and English Wikipedia. It uses two self-supervised objectives:

- **Masked Language Modeling (MLM):**  
  BERT randomly masks 15% of the input tokens and trains the model to predict these masked tokens using context from both directions. To reduce the mismatch between pre-training and fine-tuning (where the `[MASK]` token does not appear), BERT applies the following strategy when masking: 80% of the time, the selected token is replaced with `[MASK]`, 10% of the time, it is replaced with a random token, 10% of the time, it remains unchanged. This approach helps BERT learn to predict both masked and unmasked tokens, improving its robustness.

- **Next Sentence Prediction (NSP):**  
  BERT is also trained to understand relationships between sentences. Given a pair of sentences, the model predicts whether the second sentence follows the first in the original text. The final hidden state of the `[CLS]` token is used for this task.

### Fine-tuning

After pre-training, BERT can be fine-tuned on specific tasks by adding a simple output layer. The entire model is trained end-to-end on the downstream task with minimal architecture changes. Fine-tuning typically requires only a few epochs and achieves state-of-the-art results on many benchmarks.

At the output layer, the final hidden vector of the [CLS] token is used for classification tasks, while the final hidden vectors of all tokens are used for token-level tasks like named entity recognition.

---
## Experiments

BERT was evaluated on 11 NLP tasks, including:

- **GLUE Benchmark**: Both the base and large model outperformed all previous SOTA models.
- **SQuAD v1.1 & v2.0**: Set new records for question answering, surpassing human-level performance on SQuAD v1.1.
- **SWAG**: Outperforms the GPT baseline by 8.3%.

---
## Ablations

- **Pre-training**:
    - No NSP: Removing NSP hurts performance on tasks requiring sentence relationships, like QMNLI, MNLI and SQuAD.
    - LTR & No NSP: LTR model performs worse than MLM on all tasks, showing the importance of bidirectional context.
- **Model size**: Larger models consistently outperform smaller ones, with BERT-Large achieving state-of-the-art results on most tasks.
- **Feature-based Approach**: The feature-based approach using BERT representations were on par with previous models on CoNLL-2003 NER task, demonstrating thar BERT is effective for both fine-tuning and feature-based approaches.

These experiments highlight that both the bidirectional pre-training and the NSP objective are crucial for BERT’s success, and that scaling up the model size further enhances its capabilities.



## Impact and Extensions

BERT has had a transformative impact on NLP, inspiring numerous variants and extensions such as RoBERTa, ALBERT, DistilBERT, and multilingual BERT.
