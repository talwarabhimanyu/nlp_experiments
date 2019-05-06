# Deep Learning for Natural Language Processing
This repo contains code for my experiments in Deep Learning for NLP. Where possible, I'll try to provide a direct link to a Colab Notebook. I'll be using PyTorch.

# List of Experiments
1. [Simple LSTM based Language Model with Beam Search](#Simple-LSTM-based-Language-Model-with-Beam-Search) ([Colab Notebook](https://colab.research.google.com/drive/1nD2s4r7XrYP95gxfBoTr3Ajy9QW4YjUQ "Language Model with Beam Search"))
)
2. [Visualization of Layer Activation Distribution with & without BatchNorm](#Visualization-of-Layer-Activation-Distribution-with-and-without-BatchNorm)



# Experiment Details
### Simple LSTM based Language Model with Beam Search
* My implementation of an LSTM based language model with Beam Search (for sentence generation).
* [Click this link for Colab Notebook](https://colab.research.google.com/drive/1nD2s4r7XrYP95gxfBoTr3Ajy9QW4YjUQ "Language Model with Beam Search")

### Visualization of Layer Activation Distribution with and without BatchNorm
* Inspired by the paper [_How Does Batch Normalization Help Optimization?_](https://arxiv.org/abs/1805.11604), this notebook trains a VGG-11 architecture on CIFAR-10 and compares layer activation distributions with and without BatchNorm. 

* The plots below show disitribution of activations of Layer 10 (of the VGG11 network without BatchNorm) with and without using BatchNorm (see notebook for code). I do observe a slight shift in the distribution (without BatchNorm) between epochs 0-10, however it does not seem severe (sorry for not being precise!).

  Santurkar et. al do note in the paper that the "difference in disitributional stability [...] seems to be marginal."

	<p float="middle">
		<img src="./images/vgg11-layer10-noBN.png" width="35%" />
		<img src="./images/vgg11-layer10-BN.png" width="35%" />
	</p>
* Colab notebook coming soon.


