# Deep Learning for Natural Language Processing
This repo contains code for my experiments in Deep Learning for NLP. Where possible, I'll try to provide a direct link to a Colab Notebook. I'll be using PyTorch.

## List of Experiments
### Simple LSTM based Language Model with Beam Search
* My implementation of an LSTM based language model with Beam Search (for sentence generation).
* [Click this link for Colab Notebook](https://colab.research.google.com/drive/1nD2s4r7XrYP95gxfBoTr3Ajy9QW4YjUQ "Language Model with Beam Search")

### Visualization of Layer Activation Distribution with and without BatchNorm
* Inspired by the paper [_How Does Batch Normalization Help Optimization?_](https://arxiv.org/abs/1805.11604), this notebook trains a VGG-11 architecture om CIFAR-10 and compares layer activation distributions with and without BatchNorm:
[comment]![without-bn][without-bn] ![with-bn][with-bn]
<p float="center">
	<img src=" https://github.com/talwarabhimanyu/nlp_experiments/blob/master/images/vgg11-layer10-noBN.png">
	<img src=" https://github.com/talwarabhimanyu/nlp_experiments/blob/master/images/vgg11-layer10-BN.png">
</p>
* Colab notebook coming soon.

[comment][without-bn]: https://github.com/talwarabhimanyu/nlp_experiments/blob/master/images/vgg11-layer10-noBN.png
[[comment]with-bn]: https://github.com/talwarabhimanyu/nlp_experiments/blob/master/images/vgg11-layer10-BN.png

