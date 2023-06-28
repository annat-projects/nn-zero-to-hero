# Neural Networks: Zero to Hero
A course on neural networks from Andrej Karpathy that starts all the way at the basics.

# Building micrograd
Basics of backpropagation and training of neural networks from zero.

# Building makemore
Implemented a bigram character-level language model. The focus is on:
1) torch.Tensor and its subtleties and use in efficiently evaluating neural networks
2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).

# Building makemore: MLP
Implemented a multilayer perceptron (MLP) character-level language model. Introduced many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).

# Building makemore: Activations & Gradients, BatchNorm
Dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. Looked at the typical diagnostic tools and visualizations to understand the health of deep network. Introduced the first modern innovation that greatly simplifies the work: Batch Normalization.

# Building makemore: Building WaveNet
Made it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions.

# Building GPT
Built a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3.
