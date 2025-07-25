---
title: LLaMA
---

# LLaMA: Open and Efficient Foundation Language Models

[Paper](https://arxiv.org/abs/2302.13971) | [Code](TODO) | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/LLaMA)

## Introduction

LLaMA is a family of large language models developed by Meta AI in 2023. These models are designed to be efficient, open, and accessible, making advanced language modeling capabilities available to researchers and practitioners. Unlike previous proprietary models, LLaMA is distributed under a non-commercial license, supporting academic and research use while promoting responsible AI development.

Large language models (LLMs) trained on vast datasets have demonstrated impressive abilities in natural language understanding and generation. While it is often assumed that increasing model size leads to better performance, [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) showed that, for a fixed budget, training smaller models on more data can outperform simply scaling up model size. However, the budget doesn't consider inference costs which is critical for serving LLMs at scale.

This work focuses on training a series of language models that optimize performance across different inference budgets by using more training tokens than what is typically used. All LLaMA models are trained exclusively on publicly available datasets.

## Approach

### Data

LLaMA was trained on a large, diverse dataset comprising publicly available sources. The dataset includes:

- English CommonCrawl
- C4
- Github: Projects under BSD and MIT licenses
- Wikipedia
- Gutenberg and Books3
- ArXiv
- Stack Exchange

The BPE algorithm is used for tokenization. The total training corpus contains approximately 1.4 trillion tokens. Rigorous data curation and deduplication ensure high-quality training data and minimize the inclusion of low-quality or harmful content.

### Architecture

LLaMA is based on the standard transformer architecture, with several modifications for improved efficiency:

- **Pre-Normalization**: Instead of normalizing the output, LLaMA normalizes the input of each sub-layer using RMSNorm, which replaces the traditional LayerNorm.
- **Activation Function**: LLaMA uses the SwiGLU activation function instead of ReLU, improving model performance.
- **Positional Embeddings**: Rotary positional embeddings are used instead of absolute positional embeddings, enhancing the model's ability to capture sequence information.

**Implementation Details**

- LLaMA leverages FlashAttention, a memory-efficient attention mechanism that reduces memory usage and accelerates training.
- Checkpointing is employed to reduce the activations computations during backward pass and further improve training efficiency.

## Results

LLaMA was evaluated in both zero-shot and few-shot settings across 20 benchmarks:

- **Common Sense Reasoning**: LLaMA-65B outperformed Chinchilla-70B and PaLM-540B on most benchmarks. LLaMA-13B also surpassed GPT-3 on most tasks, despite being 10x smaller.
- **Reading Comprehension**: LLaMA-65B is competitive with PaLM-540B, and LLaMA-13B outperformed GPT-3 by a few percentage points.
- **Code Generation**: LLaMA outperformed other general models such as PaLM and LaMDA.
- **Massive Multitask Language Understanding**: LLaMA-65B lags behind Chipmunk-70B and PaLM-540B, possibly because it was trained on only 177GB of books data, compared to nearly 2TB for other models. However, instruction fine-tuning significantly improves LLaMA-65B's performance.

## Bias, Toxicity, and Misinformation

- **Toxicity**: On the RealToxicityPrompts benchmark, LLaMA's toxicity scores are comparable to other models. Toxicity tends to increase with model size.
- **Bias**: LLaMA models exhibit more bias than GPT-3 and OPT-175B, particularly in the religion category, followed by age and gender.
- **Misinformation**: LLaMA models are likely to hallucinate incorrect answers. Compared to GPT-3, LLaMA scored higher on TruthfulQA benchmark.


## Limitations and Future Directions

