# Deep Learning for Natural Language Processing
This repo contains code for my experiments in Deep Learning for NLP. Where possible, I'll try to provide a direct link to a Colab Notebook. I'll be using PyTorch.

For feedback, feel free to reach me on: '{}@{}.com'.format('abhimanyu.talwar1', 'gmail')

# List of Experiments
1. [Simple LSTM based Language Model with Beam Search](#Simple-LSTM-based-Language-Model-with-Beam-Search) ([Colab Notebook](https://colab.research.google.com/drive/1qv9-LVdIe0boX2HxHLl-zov1NSErif-T "Language Model with Beam Search"))

2. [Visualization of Layer Activation Distribution with & without BatchNorm](#Visualization-of-Layer-Activation-Distribution-with-and-without-BatchNorm) ([Colab Notebook](https://colab.research.google.com/drive/1_glfmBfFWiqKbXAcjXEMngIZC8af8UtC "Visualize Activation Distributions with and without BatchNorm"))


# Experiment Details
## Simple LSTM based Language Model with Beam Search
* My implementation of an LSTM based language model with Beam Search (for sentence generation).
* [Click this link for Colab Notebook](https://colab.research.google.com/drive/1nD2s4r7XrYP95gxfBoTr3Ajy9QW4YjUQ "Language Model with Beam Search")

## Visualization of Layer Activation Distribution with and without BatchNorm
* Inspired by the paper [_How Does Batch Normalization Help Optimization?_](https://arxiv.org/abs/1805.11604), this notebook trains a VGG-11 architecture on CIFAR-10 and compares layer activation distributions with and without BatchNorm. I observe activations of Layer 10 of the original VGG11 network (the one which does not use BatchNorm).

* **Distributional stability of activations is similar across epochs, BN or no BN**: The plots below show disitribution of activations of Layer 10, with and without using BatchNorm (see notebook for code). I do observe a slight shift in the distribution (without BatchNorm) between epochs 0-10, however it does not seem severe (sorry for not being precise!).

  Santurkar et. al do note in the paper that the "difference in disitributional stability [...] seems to be marginal."

	<p float="center">
		<img src="./images/vgg11-layer10-noBN.png" width="40%" />
		<img src="./images/vgg11-layer10-BN.png" width="40%" />
	</p>
* **BN speeds up training even if distribution stabilization effect is suppressed:** Borrowing from the authors' experiments, I injected random Gaussian noise *after* the first BatchNorm layer which precedes Layer 10, with the intent of destabilizing the activation distribution again. The plots below show the impact of this noise on distribution of Layer 10 activations (for networks with and without BatchNorm).

	<p float="center">
		<img src="./images/noisyBN-vgg11-layer10-noBN.png" width="40%" />
		<img src="./images/noisyBN-vgg11-layer10-BN.png" width="40%" />
	</p>
  The plot below shows training performance (loss and accuracy for train/validation sets). If the hypothesis is that BatchNorm improves training by stabilizing activation distributions, then that seems somewhat dispelled by this plot. The blue lines plot the regime where I introduced noise *after* the BatchNorm layer, to destabilize activations. We can observe that even after introducing noise, training accuracy improved at a faster pace, than in the case of the black line, which plots the vanilla regime (no noise and no BatchNorm), at least for the first 15 epochs or so.

  	<p float="center">
		<img src="./images/bn-comparison.png" width="90%" />
	</p>
* **Does mere whitening of a layer's outputs suffice for stabilizing activation distribution?** Authors Ioffee and Szegedy note in their  original BatchNorm paper that mere whitening of a layer's activations does not work. They describe a toy example in which a single layer adds an intercept (which is learned) to its inputs, and then normalizes the output by subtracting the mean over the outputs. They explain that if the gradient calculation does not take this normalization into account, then the layer's output and correspondingly the loss would never change even as the intercept value continues to increase. I have reproduced that toy example in my Colab Notebook. The plot below shows the result and confirms that the loss doesn't change even as the intercept continues to increase.
	<p float="center">
		<img src="./images/whiten.png" width="50%" />
	</p>
