# Regularization

## L1

### Q&A

## L2

### Q&A

## Dropout 
[Paper](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)  

Dropout is a regularization technique used to prevent overfitting in neural networks. 
It works by randomly setting a fraction of the input units to zero during training, which helps to prevent the model from becoming too reliant on any one feature.
During inference, dropout is turned off, and the full network is used.


**Why is Dropout needed?**  
Dropout helps address overfitting, especially in deep networks where the model might memorize the training data instead of learning general patterns.

Acts as an ensemble of subnetworks, improving robustness.


**Disadvantages:** 

- Slower convergence during training due to randomness.
- Incompatible with certain architectures like RNNS, LSTMs, CNNs.
- Can lead to underfitting if the dropout rate is too high.
- Can conflict with batch normalization, as it introduces noise that can affect the normalization process.


### Q&A

1. What happens to the network during inference if Dropout was used during training?
      - During inference, all neurons are active, and the outputs are typically scaled to account for the dropped units during training.

2. How does Dropout work in practice?
     - At each training step, each neuron's output is set to zero with probability p (dropout rate). The remaining outputs are scaled by 1 / (1 - p) to keep the expected sum of activations the same.

3. What is the variance of the activations in each hidden layer when dropout is and is not applied? Draw a plot to show how this quantity evolves over time for both models.
      - When dropout is applied, the variance of activations is reduced due to the random dropping of units. Without dropout, the variance tends to be higher as all units contribute to the output. 

4. What happens when dropout and weight decay are used at the same time? Are the results additive? Are there diminished returns (or worse)? Do they cancel each other out? 
      - When dropout and weight decay are used together, they can complement each other. 
   Dropout reduces overfitting by randomly dropping units, while weight decay penalizes large weights. 
   There can be diminished returns if both techniques are too aggressive, leading to underfitting. They do not cancel each other out; rather, they work together to improve generalization.

5. What happens if we apply dropout to the individual weights of the weight matrix rather than the activations?
      - This is called DropConnect. Zeroing weights directly can lead to more erratic training, unless handled with care. It is also computationally more expensive.







