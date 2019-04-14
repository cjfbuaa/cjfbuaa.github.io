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
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
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
import torchvision.models.vgg as vgg

class ContentLoss(torch.nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output
```

------



## References

[1] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2414-2423).

[2] Mahendran, A., & Vedaldi, A. (2015). Understanding deep image representations by inverting them. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 5188-5196).

[3] Johnson, J., Alahi, A., & Fei-Fei, L. (2016, October). Perceptual losses for real-time style transfer and super-resolution. In *European conference on computer vision* (pp. 694-711). Springer, Cham.

[4] Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4681-4690).

[5] Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 0-0).