# Makemore Tutorial

In this repository, we follow along with Andrej Karpathy's [building makemore tutorial](https://youtu.be/PaCmpygFfXo). To learn more about makemore, check out the [makemore repository](https://github.com/karpathy/makemore/tree/master).

- [mm_intro.ipynb](mm_intro.ipynb)

    Introduces the task and dataset. We explore the objective function of minimizing the negative log-likelihood of the data to learn the conditional distribution of the next character given the previous characters. A baseline is established using the bigram-counts matrix derived from the training data. We then train a simple one-layer neural network on the task and compare the results to the baseline.

- [mm_mlp.ipynb](mm_mlp.ipynb)
    We expand the context window to 3 characters and train a multi-layer perceptron (MLP) on the task. For educational purposes, the forward pass is implemented manually without the use of PyTorch's `nn` module. We analyze the saturation of the activation functions and pre-activation distributions of the hidden layers. In this context, we explore the effects of the weight initialization scheme and the motivation for batch normalization. Finally, we learn to analyze the distribution of the activations and the weights and how to assess the learning rate using the update-to-gradient ratio.

- [mm_mlp_manual_backprop.ipynb](mm_rnn.ipynb)
    The full backpropagation algorithm is implemented manually without the use of PyTorch's `autograd` module. We further derive optimized flows (gradients) through the softmax and batch normalization layers on paper to speed up the backpropagation process. This is then implemented in code. The full manual backpropagation algorithm is then compared to the automatic differentiation provided by PyTorch to ensure correctness.

- [mm_wavenet.ipynb](mm_wavenet.ipynb)
    Finally, we expand the context window to 8 characters and train a WaveNet-inspired model on the task. For educational purposes, we do so with our own implementations of the Embedding, Linear, BatchNorm1d, Tanh, Flatten(Consequtive) and Sequential modules, similar to PyTorch's `nn` module. 


