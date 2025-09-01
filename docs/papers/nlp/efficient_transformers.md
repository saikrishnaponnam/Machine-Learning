---
title: Efficient Transformers
---

# Efficient Transformers: A Survey

[Paper Link](https://arxiv.org/abs/2009.06732)


## Introduction

Transformers have revolutionized natural language processing and other domains by enabling powerful sequence modeling and attention mechanisms. However, their quadratic complexity with respect to input sequence length poses significant computational and memory challenges, especially for long sequences. As applications scale, the need for more efficient transformer architectures has become critical. There has been a lot of research on the efficient variants of the model. This paper surveys the overview of the recent advances made primarily in modeling and architectural innovations to improve the efficiency of transformers. The paper also propose a taxonomy to categorize them by their core techniques and primary use cases.

## Background

The standard transformer architecture, introduced by Vaswani et al. (2017), relies on self-attention to capture dependencies between all tokens in a sequence. While highly effective, the self-attention mechanism scales quadratically (O(N^2)) with sequence length N, making it impractical for long documents, high-resolution images, or real-time applications. A non-trivial amount of compute also comes from the two layer feed-forward layers at every block, although the complexity of FFN is linear w.r.t sequence length, but is generally still costly.

### Transformer Modes
Transformers can primarily operate in three modes:

1. **Encoder-Only**: Used for tasks like text classification and named entity recognition (e.g., BERT).
2. **Decoder-Only**: Used for autoregressive tasks like language modeling and text generation (e.g., GPT).
3. **Encoder-Decoder**: Used for sequence-to-sequence tasks like machine translation.

The mode of usage depends on the specific application and task requirements. In encoder there is only self-attention, while in decoder there is both self-attention and cross-attention. This influence the design of the self-attention mechanism. The self-attention in decoder need to be causal whereas in encoder it can be non-causal. This requirement make designing efficient transformers challenging.



## Taxonomy of Efficient Transformers

Efficient transformers can be broadly categorized based on the techniques they use to address the bottlenecks of standard self-attention:

1. **Fixed patterns**: Instead of letting every token attend to every other token, limit attention to fixed windows or blocks.
   - Blockwise Patterns: Divide the sequence into blocks and restrict attention within blocks.
   - Strided Patterns: Allow attention to tokens at fixed intervals.
   - Compressed Patterns: Use techniques like pooling to reduce sequence length before attention. For example, Compressed Attention (Liu et al., 2018) uses strided convolution to reduce the sequence length.
2. **Combinations of Patterns**: Combine multiple approaches to improve coverage.
3. **Learned Patterns**: Unliked fixed patterns, models learn the access patterns during training. They determine the token relevance and then assign tokens to buckets or clusters.
4. **Neural Memory**: Uses a learnable side memory module that can access multiple tokens at once. The global tokens act as a form of model memory that learns to gather information from the entire sequence.
5. **Low-Rank Methods**: Approximate the attention matrix using low-rank factorization, reducing computation.
6. **Kernel Methods**: Re-express attention as kernel operations so you don’t explicitly compute the N x N matrix.
7. **Recurrent Memory**: An extension to blockwise method is to connect the blocks recurrently.
8. **Downsampling**: Shrink the sequence length (pooling, striding).
9. **Sparse Models**: Sparse models sparsely activate a subset of parameters. Typically, sparse models operate on an adaptive basis in which the sparsity is typically learned (via MOE like mechanisms).

## Key Models and Approaches


### Memory Compressed Transformer 
[Liu et al., 2018](https://arxiv.org/abs/1801.10198)

An earlier attempt to handle long sequences. It uses two techniques:

- Local Attention Span: Input sequence is divided into blocks and self-attention is computed within each block. For a block size of b, the computational and memory cost of self-attention is $O(b^2)$. Since there are N / b blocks, the overall cost is $O(N. b)$, which is linear.
- Memory-Compressed Attention: Strided Convolution is used to reduce the number of keys and values, while queries remain unchanged. This also allows model exchange information globally across the input sequence. Applying a kernel size and strides of k, reduces the cost of attention mechanism to $O(N . N/k)$. 

### Image Transformer
[Parmar et al., 2018](https://arxiv.org/abs/1802.05751)

Image Transformer restricts the self-attention to only local neighborhoods. Input is partitioned into query blocks and each query block attends to a memory block that contains the query block and its surrounding pixels. This allows the model to scale linearly with respect to the number of pixels. There are two variants for choosing query blocks:

- 1D Local Attention: The image is flattened into a 1D sequence in raster order and divided into non-overlapping query blocks of length $l_q$. Each query block attends to a memory block that contains the query block and a fixed number of pixels, $l_m$, generated before query pixel. 
- 2D Local Attention: The image is divided into non-overlapping query blocks of size $l_q = w_q x h_q$. Each query block attends to a memory block that contains the query block and a fixed number of pixels, $h_m$ and $w_m$, generated before the query block.

Overall complexity is O(N. m), where N = num of pixels, m = size of memory block.  
Although local attention can decrease the computational cost, it also loses global context. 

### Set Transformer
[Lee et al., 2019](https://arxiv.org/abs/1810.00825)

### Sparse Transformer
[Child et al., 2019](https://arxiv.org/abs/1904.10509)

Sparse Transformer uses fixed sparse patterns to reduce attention computation. It combines two types of attention patterns - Strided Attention and local Attention. Half of the heads are dedicated to strided attention and the other half to local attention. The idea is to reduce the dense attention matrix to sparse version by only computing attention on a sparse number of $q_i, k_j$ pairs.  
The complexity of sparse attention is $O(N. \sqrt{N})$.

But sparse attention requires custom GPU kernel implementation for efficient block-sparse variant matrix multiplication.

### Longformer
[Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)

Longformer is a variant of Sparse Transformer that uses a combination of dilated sliding window attention and global attention tokens. Each token attends to a fixed-size window of surrounding tokens (with gasps), while a few special tokens can attend to all tokens in the sequence(like [CLS] for classification, or question tokens in QA). This allows the model to capture both local and global context efficiently. Longformer also increases the receptive field as the model goes deeper. The complexity is $O(N.w)$, where w is the window size.

### Extended Transformer Construction (ETC)
[Ainslie et al., 2020](https://arxiv.org/abs/2004.08483)

The ETC is another variant of Sparse Transformer that introduces a new global-local attention mechanism. ETC introduces $n_g$ additional auxiliary tokens, called global tokens, as a prefix to the original input sequence. Attention is then split into four components: global-to-global, global-to-local, local-to-global, and local-to-local. The global tokens can attend to all tokens in the sequence, while local tokens can only attend to a fixed window of surrounding tokens. Local tokens capture local context efficiently. Global tokens serve as memory hubs that summarize and redistribute information. The complexity is $O(n_g^2 + n_g N )$.

ETC can't be used for autoregressive decoding. This is because we can't compute causal masks because of global attention.

### BigBird
[Zaheer et al., 2020](https://arxiv.org/abs/2007.14062)

BigBird is primarily built on top of ETC. BigBird model comprises several key components:

- Global attention: A subset of indices is selected as global tokens.
- Sliding window attention: Each query attends to $w/2$ tokens on left and right.
- Random attention: Each query attends to $r$ random tokens in the sequence.

Memory complexity is $O(N)$

Similar to ETC, BigBird can't be used for autoregressive decoding.

### Routing Transformer
[Roy et al., 2021](https://arxiv.org/abs/2003.05997)

Routing Transformer is a content based sparse attention mechanism. It learns attention sparsity in a data-driven approach using clustering. Q & K are projected into a routing matrix R, using a d x d orthonormal projection matrix. $R = QW_R, KW_R$. The routing matrix is then clustered using K-means. K-means is trained in an online fashion. Each query attends to keys in the same cluster. This allows the model to learn dynamic attention patterns based on content similarity, but keeps it sparse. The complexity is $O(N^{1.5})$


### Reformer
[Kitaev et al., 2020](https://arxiv.org/abs/2001.04451)

Reformer is based on locality-sensitive hashing (LSH) to reduce the complexity of self-attention. The key idea is to hash the queries and keys into buckets such that similar queries and keys fall into the same bucket. Attention is then computed only within each bucket, significantly reducing the number of comparisons. The complexity is $O(N.log(N))$.

**LSH Attention**: LSH introduces parameter sharing between queries and keys. The queries and keys are hashed into buckets using a random projection matrix $R \in R^{k x b/2}$. The hash function is defined as $hash(x) = argmax([xR; -xR])$, where x is the query or key vector. In order maintain causal masking, Reformer assign and maintain a position index for every query and key.

### Sinkhorn Transformer
[Tay et al., 2020](https://arxiv.org/abs/2002.11296)

Sinkhorn transformer is based on learned patterns. It learns sparse patterns by re-sorting the input key and values in a block-wise fashion and then applying local block-based attention. 

$$ A_{ij} = Q_i \psi_S(K)_j^T \text{ if } (\lfloor j/N \rfloor = \lfloor i/N \rfloor) \text{ else } 0 $$



### Linformer
[Wang et al., 2020](https://arxiv.org/abs/2006.04768)

### Performer
[Choromanski et al., 2020](https://arxiv.org/abs/2009.14794)

### Linear Transformer
[Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236)


### Transformer-XL
[Dai et al., 2019](https://arxiv.org/abs/1901.02860)



### Axial Transformer
[Ho et al., 2019](https://arxiv.org/abs/1912.12180)



## Techniques for Efficiency

- **Chunking and Windowing**: Process sequences in chunks or windows to limit attention scope.
- **Approximate Nearest Neighbors**: Use hashing or clustering to select relevant tokens for attention.
- **Custom Backpropagation**: Optimize gradient computation for memory savings.
- **Low-Precision Arithmetic**: Employ quantization and mixed-precision training.

## Applications

Efficient transformers have enabled breakthroughs in:
- Long document classification and summarization
- Genomics and protein sequence modeling
- Real-time speech and video processing
- Large-scale vision tasks (e.g., high-res images)

## Challenges and Future Directions

Despite significant progress, challenges remain:
- Balancing efficiency and accuracy
- Generalizing across domains and tasks
- Hardware optimization and deployment
- Interpretability and robustness

Future research will likely focus on hybrid models, adaptive attention mechanisms, and further integration with hardware accelerators.
