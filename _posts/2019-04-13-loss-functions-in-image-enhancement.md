---
layout: single
title:  "Loss Functions in Image Enhancements"
---

## What is the Loss function?

According to the [wiki](<https://en.wikipedia.org/wiki/Loss_function>) and [blogs](<https://isaacchanghau.github.io/post/loss_functions/>), the loss function or cost function is a function that maps events or values of variables onto a real number to represent **cost**. **In the artificial neural networks (NN), we use loss function to measure the inconsistency between predicted values and real values**. Most of our research in NN is seeking to minimize a loss function.

The most widely used loss function in image enhancement is *Mean Squared Error (MSE)*, it formulated as follow:

$$
\begin{align}
MSE = \sum_{i=1}^n (\hat {X_{i}} - X_{i})^{2},
\end{align}
$$

where $X$ is vector of observed values and $\hat X​$ is vector of generated values. We usually use *MSE* to measure the inconsistency of high-quality images and enhanced images from low-quality. 

## Which Loss function is good ?

Many loss function is suitable to be used in Image Enhancement tasks.



**The Total Variation (TV) Loss** is widely used in image generation tasks[1, 2]. It encourages spatial smoothness in the generated images. People usually use it in noise removal as the [wiki](<https://en.wikipedia.org/wiki/Total_variation_denoising#cite_note-strong-2>) shows. TV denoising is remarkably effective at simultaneously preserving edges whilst smoothing away noise in flat regions, even at low signal-to-noise ratios.

![Denoising Example From Wikis](https://upload.wikimedia.org/wikipedia/en/e/e8/ROF_Denoising_Example.png)

When the input is a image, we can formulated the TV Loss as:
$$
\begin{align}
V_{\text{aniso}}(y)=\sum _{i,j}{\sqrt {|y_{i+1,j}-y_{i,j}|^{2}}}+{\sqrt {|y_{i,j+1}-y_{i,j}|^{2}}}=\sum _{i,j}|y_{i+1,j}-y_{i,j}|+|y_{i,j+1}-y_{i,j}|,
\end{align}
$$
where $i, j$ represent the pixel position.



## References

[1] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2414-2423).

[2] Mahendran, A., & Vedaldi, A. (2015). Understanding deep image representations by inverting them. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 5188-5196).