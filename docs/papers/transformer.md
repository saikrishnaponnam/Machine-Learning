# Transformer

Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)| [Code]()

## Introduction
The Transformer is a deep learning model introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. It revolutionized natural language processing by relying entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. Transformers have since become the foundation for state-of-the-art models in NLP, such as BERT, GPT, and many others.

Prior to Transformers, sequence transduction models like RNNs, GRUs and LSTMs were state of the art approaches. However, these models struggled with long-range dependencies and parallelization.The RNNs have an inherently sequential nature which prevents parallelization within training examples. When dealing with large sequences, the memory constraints limit batching across examples. Attention mechanisms ([Additive Attention](additive_attention.md), [Dot Attention](dot_product_attention.md)) were introduced to address long-range dependencies, allowing models to focus on relevant parts of the input sequence. However, such attention mechanism are used in conjunction with RNNs. The Transformer architecture was introduced to remove RNNs and rely entirely on attention and takes this further by using self-attention as its core building block.

## Background




## Architecture

Similar to other sequence transduction models, The Transformer consists of an encoder and a decoder, each composed of multiple identical layers. The encoder processes the input sequence and passes its representation to the decoder, which generates the output sequence. Key components include:

- **Multi-Head Self-Attention:** Allows the model to attend to information from different representation subspaces at different positions.
- **Position-wise Feed-Forward Networks:** Each layer contains a fully connected feed-forward network applied to each position separately.
- **Positional Encoding:** Since the model lacks recurrence, positional encodings are added to input embeddings to provide information about the order of the sequence.
- **Residual Connections and Layer Normalization:** These help with training deep networks by mitigating vanishing gradients and stabilizing learning.

### Encoder-Decoder Stacks
**Encoder**: Encoder takes input and generates a rich context-sensitive representation of the input. The encoder is composed of a stack of $N = 6$ identical layers. Each containing two sub-layers: multi-head self-attention and position-wise fully connected feed-forward networks. Residual connections are used around each sub-layer, followed by layer normalization. To facilitate this residual connections all the sub-layers in the model as well as embeddings layer produce outputs of the same dimension $d_{model} = 512$.

**Decoder**: Decoder generates the output sequence one token at a time, attending to both the previously generated tokens and the encoder's output. The decoder is also composed of a stack of $N = 6$ identical layers. The decoder inserts a third sub-layer, in addition to the two sub-layers in the encoder, which performs multi-head attention over the encoder's output, allowing the decoder to focus on relevant parts of the input sequence while generating each token. The masking in the self-attention layer ensures that the predictions for a given token can only depend on known outputs at earlier positions.

### Attention

**Scaled Dot-Product Attention**:
The attention mechanism is the core of the Transformer architecture. It computes a weighted sum of values (V) based on the similarity between queries (Q) and keys (K). The attention scores are calculated using the dot product of Q and K, scaled by the square root of the dimension of K, followed by a softmax operation to obtain weights. The output is then computed as a weighted sum of V.

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Dot-product attention is much faster than additive attention, as it only requires a single matrix multiplication and a softmax operation. The scaling factor $\sqrt{d_k}$ is used to prevent the dot products from growing too large, which can lead to numerical instability in the softmax function.

### Multi-Head Attention

Multi-head attention extends the basic attention mechanism by allowing the model to jointly attend to information from different representation subspaces. Instead of performing a single attention function with $d_{model}$-dimensional keys, queries, keys and values are linearly projected h times with different, learned linear projections to $d_k, f_k, d_v$ dimensions. On each of these projected representations, the attention function is applied in parallel, yielding $d_v$ dimensional $h$ different outputs. These outputs are then concatenated and linearly transformed to produce the final output.

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O \\
\text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

In the original Transformer, $h = 8$ and $d_k = d_v = 64$, resulting in a total dimension of $d_{model} = h \cdot d_k = 512$.


### Position-wise Feed-Forward Networks

A fully connected feed-forward network is applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between. The first linear transformation projects the input from $d_{model}$ to $d_{ff}=2048$ dimensions, and the second projects it back to $d_{model}$ dimensions.

### Positional Encoding

Since the Transformer does not have recurrence or convolution, it uses positional encodings to inject information about the position of tokens in the sequence. The positional encodings are added(summed) to the input embeddings and are designed to allow the model to learn relative positions. The encoding is defined using sine and cosine functions of different frequencies

## Why Self-Attention

Self-attention enables the model to weigh the importance of different words in a sequence, regardless of their distance from each other. This allows for better modeling of long-range dependencies and parallel computation, making training faster and more efficient compared to RNNs.

The authors compare self-attention with RNNs and CNNs in three critical aspects: 

- **Computational Complexity**: Self-attention has a complexity of $O(n^2 * d)$ for a sequence of length $n$ and d-dimensional embeddings, while RNNs have $O(n * d^2)$ per step but require sequential processing. CNNs can also achieve $O(k * n * d^2)$ with fixed-size kernels, but they struggle with long-range dependencies.
- **Parallelization**: Self-attention allows for parallel computation across all positions in the sequence, making it highly efficient on modern hardware. RNNs require sequential processing, which limits parallelization. CNNs can be parallelized but are constrained by kernel sizes.
- **Long-Range Dependencies**: Self-attention can directly connect distant positions in the sequence, making it effective for modeling long-range relationships. RNNs struggle with long-range dependencies due to vanishing gradients, while CNNs require larger kernels to capture such relationships, which can lead to increased computational costs. The shorter the distance between any combinations of positions in the input and output sequences, the easier it is to learn long-range dependencies. The maximum path length in the self-attention is $O(1)$, while in RNNs it is $O(n)$, and in CNNs it is $O(log_k n)$.


## Training

The Transformer model was trained on the WMT 2014 English-to-German dataset, which contains 4.5 million sentence pairs. Sentences were encoded using Byte Pair Encoding (BPE), resulting in a shared source and target vocabulary of 37,000 tokens.

To optimize training efficiency, sentence pairs were batched by approximate sequence length, with each batch containing about 25,000 source tokens and 25,000 target tokens.

The model was trained using the Adam optimizer. The learning rate was set to $d_{model}^{-0.5}$, with a warm-up phase for the first 4,000 steps. After the warm-up, the learning rate decayed proportionally to the inverse square root of the step number.

## Results

The Transformer achieved state-of-the-art performance on machine translation tasks. It outperformed previous models in both translation accuracy and training speed, demonstrating the effectiveness of the self-attention mechanism and the overall architecture.
