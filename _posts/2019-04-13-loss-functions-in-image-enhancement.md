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

where $X$ is vector of observed values and $\hat X$ is vector of generated values. We usually use *MSE* to measure the inconsistency of high-quality images and enhanced images from low-quality.  You can use it by call `nn.MSELoss()` as simple as possible.

## Which Loss function is good ?

Many loss functions are suitable to be used in Image Enhancement tasks.

**The Total Variation (TV) Loss** is widely used in image generation tasks[1, 2]. It encourages spatial smoothness in the generated images. People usually use it in noise removal as the [wiki](<https://en.wikipedia.org/wiki/Total_variation_denoising#cite_note-strong-2>) shows. TV Denoising is remarkably effective at simultaneously preserving edges whilst smoothing away noise in flat regions, even at low signal-to-noise ratios.

![Denoising Example From Wikis](https://upload.wikimedia.org/wikipedia/en/e/e8/ROF_Denoising_Example.png)

When the input is a image, we can formulated the TV Loss as:
$$
\begin{align}
V_{\text{aniso}}(y) &=\sum _{i,j}{\sqrt {|y_{i+1,j}-y_{i,j}|^{2}}}+{\sqrt {|y_{i,j+1}-y_{i,j}|^{2}}} \\
&=\sum _{i,j}|y_{i+1,j}-y_{i,j}|+|y_{i,j+1}-y_{i,j}|,
\end{align}
$$
where $i, j​$ represent the pixel position. We can use PyTorch to implement it as:

```python
class TVLoss(nn.Module):
    def __init__(self, tvloss_weight=1):
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight

    def forward(self, generated):
        b, c, h, w = generated.size()
        h_tv = torch.pow((generated[:, :, 1:, :] - generated[:, :, :(h - 1), :]), 2).sum()
        w_tv = torch.pow((generated[:, :, :, 1:] - generated[:, :, :, :(w - 1)]), 2).sum()
        return self.tvloss_weight * (h_tv + w_tv) / (b * c * h * w)
```



**Content Loss** or **Perceptual Loss** is widely used in style transfer[3] and image super-resolution tasks [4, 5]. It extract features using pre-trained a VGG network. Using results from this [blog](<https://medium.com/lets-enhance-stories/content-and-style-loss-using-vgg-network-e810a7afe5fc>), we can show the effects by using it as a loss function:

![vgg-effects.png](https://i.loli.net/2019/04/14/5cb2ba7cd0f2a.png)

The deeper layer in VGG you use, the more high-level features like object patterns you can combine into final results, in contrast, shallowed layer brings more low-level features. In addition, Wang [5] proposed that features before activation is more better since most features are becoming inactive after activation:

![Activation Features](https://i.loli.net/2019/04/14/5cb2bcf95d121.png)

We can formulated the Content Loss as below:
$$
\begin{align}
l_{VGG/x} =
	\frac{1}{W\times H} & \sum_{i=1}^W \sum_{j=1}^H (\phi_x(\hat Y)_{i,j} - \phi_x(Y_{i,j}))^2,	
\end{align}
$$
where $x, y$ represent the pixel position, and the $\phi_{i, j}$ denotes the features obtained by the j-th convolution after the activation layer or before the activation layer. We can use PyTorch to implement it as:

```python
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, layers, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice = torch.nn.Sequential()
        for x in range(layers):
            self.slice.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        f = self.slice(x)
        return f


class ContentLoss(torch.nn.Module):
    def __init__(self, vgg19_model, layer, criterion):
        super(ContentLoss, self).__init__()
        self.feature_extractor = nn.Sequential(*list(vgg19_model.module.features.children())[:layer])
        self.criterion = criterion

    def forward(self, generated, groundtruth):
        generated_vgg = self.feature_extractor(generated)
        groundtruth_vgg = self.feature_extractor(groundtruth)
        groundtruth_vgg_no_grad = groundtruth_vgg.detach()
```



**Adversarial Color Loss** is proposed by WESPE [6], which is measured by an adversarial discriminator $D_c$. They trained $D_c$ to differentiate between the blurred versions of enhanced $\hat Y$ and input $Y$ images. They define the  Gaussian
blur as $G_{k,l} = A \exp\bigl(-\frac{(k - \mu_x)^2}{2\sigma_x} -\frac{(l - \mu_y)^2}{2\sigma_y}\bigr)$ with with $A=0.053$, $\mu_{x,y}=0$, and $\sigma_{x,y}=3$ set empirically.  In order to be used in GAN training, the loss function is formulated as:
$$
\begin{align}
\mathcal{L}_{\text{color}} = -\sum_{i} \log{D_c(G(x)_b)}.
\end{align}
$$
where the $D_c$ is the discriminator network to be trained. We can implement gaussian filtering as below:

```python
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)

```





You can find codes for this post in this [Repo](<https://github.com/MKFMIKU/Enhancing-Loss.pytorch>)

------



## References

[1] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2414-2423).

[2] Mahendran, A., & Vedaldi, A. (2015). Understanding deep image representations by inverting them. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 5188-5196).

[3] Johnson, J., Alahi, A., & Fei-Fei, L. (2016, October). Perceptual losses for real-time style transfer and super-resolution. In *European conference on computer vision* (pp. 694-711). Springer, Cham.

[4] Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4681-4690).

[5] Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 0-0).

[6] Ignatov, A., Kobyshev, N., Timofte, R., Vanhoey, K., & Van Gool, L. (2018). WESPE: weakly supervised photo enhancer for digital cameras. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops* (pp. 691-700).