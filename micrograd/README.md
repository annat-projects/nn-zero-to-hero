# Micrograd
A tiny Autograd engine. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. The DAG only operates over scalar values, so e.g. chop up each neuron into all of its individual tiny adds and multiplies.

# Tracing / visualization
For added convenience, trace.py produces graphviz visualizations.

![image](https://github.com/annat-projects/nn-zero-to-hero/assets/19928756/0f119211-b4cd-4476-b28a-4cc561094053)
