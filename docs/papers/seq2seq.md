# Seq2Seq

Paper: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)


## Introduction
DNNs are powerful ML models that achieve excellent performance on a wide range of tasks such as image recognition and speech recognition. 
Despite their flexibility and power, DNNs can only be applied to tasks where the input and output are fixed-length vectors. It is a major limitation, since many important tasks are best expressed with sequences whose lengths are not known in advance.
For example, speech recognition and machine translation are sequential problems.

This paper shows that LSTMs can solve general sequence-to-sequence problems.
They came up with the idea of using two LSTMs, one for the input sequence, to obtain large fixed-dimensional vector and one to get the output sequence from that vector.
LSTMs are chosen because of its ability to learn long-term dependencies in sequences.

## Model

Given an input sequence of vectors $(x_1, x_2, \ldots, x_T)$, a standard RNN produces a sequence of outputs $(y_1, y_2, \ldots, y_T)$. RNNs can map input to output sequences when their alignment is known. However, mapping sequences of different lengths or with complex relationships is challenging for standard RNNs.

To address this, the authors propose a sequence-to-sequence (seq2seq) model with two RNNs:

- **Encoder**: Converts the input sequence into a fixed-length vector.
- **Decoder**: Generates the output sequence from this vector.


Because RNNs are difficult to train, LSTMs are used for both encoder and decoder. The goal of LSTM is to estimate the conditional probability $p(y_1, y_2, \ldots, y_{T'} | x_1, x_2, \ldots, x_T)$.

$$p(y_1, y_2, \ldots, y_{T'} | x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t|v, y_1, \ldots, y_{t-1}) $$

where $v$ is the fixed-length vector (the last hidden state of the encoder LSTM).

Each $p(y_t|v, y_1, \ldots, y_{t-1})$ is computed using a softmax over the vocabulary. To allow variable-length outputs, a special end-of-sequence token "<EOS\>" is used.

Based on their experiments, the authors made two modifications in the actual model:

- They used deep LSTMs (4 layers with 1000 cells per layer), which outperformed shallower architectures.
- The input sequence is reversed (e.g., input: a, b, c → LSTM input: c, b, a → output: x, y, z). This brings corresponding words closer together, improving learning.

## Training

- Input vocabulary is 160,000 and output vocabulary is 80,000. Each element is vocabulary is represented by a 1000-dimensional embeddings.
- Initialized with uniform distribution in range [-0.08, 0.08].
- Trained for a total of 7.5 epochs, with a batch size of 128.
- Used SGD with momentum, with a fixed learning rate of 0.7. After 5 epochs, learning rate is halved every half epoch.
- The sequences in minibatch are sampled to be of similar lengths.
- Minibatches contain sequences of similar lengths.


The training objective is to maximize the log probability of the correct translation $T$ given the source sentence $S$:

$$ \frac{1}{|\mathcal{S}|} \sum_{(T,S) \in \mathcal{S}} \log p(T|S) $$

where $\mathcal{S}$ is the training set.

During inference, the most likely translation is $\hat{T} = \arg\max_T p(T|S)$. A simple left-to-right beam search decoder is used to find the best translation.

## Experiments/Results

**Dataset:**

- The model is evaluated on the WMT'14 English-to-French translation task.
- A subset of 12 million sentence pairs is used for training, containing 348 million French words and 304 million English words.

Translation quality is measured using the BLEU score. The best results are achieved with an ensemble of LSTMs that differ in initialization and the random order of minibatches. The system outperforms a phrase-based SMT baseline by a significant margin of 2.2 BLEU points, despite not handling out-of-vocabulary words.

Reversing the input sequence reduces LSTM perplexity from 5.8 to 4.7 and improves the best BLEU score from 29.6 to 30.6.
