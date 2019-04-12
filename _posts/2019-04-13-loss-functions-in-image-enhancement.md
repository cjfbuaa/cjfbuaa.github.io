---
layout: single
title:  "Loss Functions in Image Enhancements"
---

# What is the Loss function?

According to the [wiki](<https://en.wikipedia.org/wiki/Loss_function>) and [blogs](<https://isaacchanghau.github.io/post/loss_functions/>), the loss function or cost function is a function that maps events or values of variables onto a real number to represent **cost**. *In the artificial neural networks (NN), we use loss function to measure the inconsistency between predicted values and real values*. Most of our research in NN is seeking to minimize a loss function.

The most widely used loss function in image enhancement is *Mean Squared Error* (MSE), it formulated as follow:
$$
MSE = \sum_{i=1}^n (\hat {X_{i}} - X_{i})^{2},
$$
where $X$ is vector of observed values and $\hat X$ is vector of generated values. We usually use MSE to measure the inconsistency of high-quality images and enhanced images from low-quality. 