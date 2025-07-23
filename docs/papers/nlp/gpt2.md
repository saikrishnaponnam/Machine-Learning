---
title: GPT-2
---

# GPT-2: Language Models are Unsupervised Multitask Learners

[Paper](https://openai.com/research/publications/language-models-are-unsupervised-multitask-learners) | [Code](TODO) | [Hugging Face](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2)

## Introduction

GPT-2 (Generative Pretrained Transformer 2) is a language model developed by OpenAI in 2019. It demonstrated that large-scale unsupervised language models can achieve impressive results on a wide range of NLP tasks, often rivaling or surpassing supervised approaches. GPT-2 is based on the Transformer architecture and is trained to predict the next word in a sentence.

Traditional machine learning systems excel at tasks they are specifically trained for, using large datasets, high-capacity models, and supervised learning. However, these systems often perform poorly when the data distribution changes, making them "narrow" in scope. The goal of the GPT-2 paper is to move towards more general systems that can perform a variety of tasks and generalize to new ones without retraining.

The authors hypothesize that training on a single task and domain limits generalization. While multitask training can improve general performance, it is challenging to collect large, diverse datasets and design objectives that cover all possible tasks. This paper demonstrates that a single model, trained on a large and diverse dataset, can learn to perform downstream tasks in a zero-shot setting—without any parameter or architecture changes.

## Approach

GPT-2 is trained using a language modeling objective: predicting the next word in a sentence given the previous words. This unsupervised approach allows the model to learn a wide range of language patterns, structures, and semantics without explicit task-specific supervision.

$$p(x) = p(s_1, s_2, \ldots, s_n) = \prod_{i=1}^n p(s_i | s_1, \ldots, s_{i-1})$$

**Handling Multiple Tasks:**
A general language model should be able to perform many tasks, not just one. While a single task can be framed as predicting $p(\text{output}|\text{input})$, a general system should learn $p(\text{output}|\text{input}, \text{task})$. Task conditioning is often implemented by modifying the model architecture or using meta-learning. In GPT-2 (similar to T5), the task is specified by a prefix in the input text, allowing the model to adapt to different tasks without changing its architecture or parameters.

The authors suggest that a sufficiently large language model, trained on diverse data, can naturally learn to infer and perform tasks simply by predicting text well. They validate this by evaluating the model in zero-shot settings.

### Dataset

GPT-2 was trained on the WebText dataset, which contains over 8 million documents and 40GB of text data. The data was collected from outbound links on Reddit with at least 3 karma, and Wikipedia documents were excluded to avoid overlap with evaluation benchmarks.

### Input Representation

A general language model should be able to generate and compute probability for any string. While character-level or byte-level language models can handle arbitrary strings, they often underperform compared to word-level models. GPT-2 addresses this by using byte-level Byte Pair Encoding (BPE) for tokenization. Unlike standard BPE, which requires a large base vocabulary (about 130,000 tokens), byte-level BPE operates with a much smaller base vocabulary of 256 tokens. This approach efficiently handles rare and out-of-vocabulary words. Additionally, GPT-2’s BPE implementation avoids merging tokens across different character categories (e.g., letters and punctuation), to prevent bad merges and maintain meaningful token boundaries.

### Model

GPT-2 is built on a decoder-only Transformer architecture, with the largest version containing up to 1.5 billion parameters. Key architectural features include:

- Stacked self-attention layers (up to 48 in the largest model)
- Layer normalization applied to the input of each sub-block, with an additional normalization after the final self-attention block
- Residual connection weights initialized with a scaling factor of 1/$\sqrt{N}$, where $N$ is the number of residual layers


## Experiments and Results

GPT-2 was trained in four different sizes: 117M, 345M, 762M, and 1.5B parameters, with the largest model commonly referred to as GPT-2. These models were evaluated on a wide range of NLP benchmarks, including language modeling, reading comprehension, translation, question answering, and summarization.

**Key Results:**

- Achieved strong performance on language modeling tasks such as LAMBADA and WikiText-103
- Demonstrated impressive zero-shot capabilities on tasks like LAMBADA, CBT, summarization, and translation, without any task-specific fine-tuning

These results highlight GPT-2's ability to generalize across tasks using only its pretraining, supporting the hypothesis that large-scale language models can perform a variety of tasks in a zero-shot setting.

## Generalization vs. Memorization

A major concern for the large language models is whether they truly generalize or simply memorize their training data. The authors analyzed the overlap between the WebText training set and common language modeling test sets, finding that 1-6% of test data overlapped with training data. While this overlap provided a small, consistent benefit, the overall performance improvements were attributed to the model's generalization ability. Notably, both training and test performance improved together as model size increased, suggesting that GPT-2 was still underfitting the WebText dataset and had not simply memorized it.

## Discussion

While GPT-2 achieved strong empirical results, the authors note that it is still far from being practical for real-world applications. Further evaluation on more diverse benchmarks is needed to determine whether increased model capacity and training data can overcome the limitations of unidirectional representations, as highlighted by models like BERT. The work demonstrates the potential of large-scale unsupervised language models, but also points to areas for future research and improvement.
