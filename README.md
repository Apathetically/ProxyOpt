# ProxyOpt
Unofficial implement of denoising SIDD dataset reference the article Hyperparameter Optimization in Black-box Image Processing using Differentiable Proxies

We use 10 instances of SIDD dataset that meet the conditions in the paper. Each instances we crop 300 512x512 image pair (noisy and gt).

### step1
The first step is to train a net to fit the function of bm3d noise reduction. Inputs are noisy images cat with parameter layers, labels are the images dealed by bm3d.

### step2
The second step is to find the optimal bm3d parameters suitable for this dataset. We fixed the net parameter and update the parameter layer which cat input noisy images. Inputs are noisy images cat with random initialize parameter layer, labels are ground truth images.

<p align="center">
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/figure_1.png" width="425"/>
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/figure_2.png" width="425"/>
</p>

### loss of step1

<p align="center">
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/train_step1.png" width="625"/>
</p>

However, we find that the net after step1 can not simulate the situation of poor parameter initialization. It will improve images with poor noise reduction due to poor initialization, as shown in the fourth row.

Because of this, it may be difficult to optimize the parameters in step2.

<p align="center">
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/psnr.png" width="320"/>
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/262.png" width="425"/>
</p>


### optimize parameter in step2

Temporarily vacant

### results

Here we give the results of network simulation. Following image depict the psnr after step1 and step2. And more results given behind. The final psnr is about 34.71 for step1 and 34.75.

note: top left [gt], top right [noisy], bottom left [bm3d], bottom right [net].

<p align="center">
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/figure_5.png" width="420"/>
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/182.png" width="420"/>
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/24.png" width="420"/>
  <img src="https://github.com/Apathetically/ProxyOpt/blob/master/results/readme/84.png" width="420"/>
</p>

### apply optimized parameter into bm3d

We optimize parameter layer several times and does not optimize the same parameters. We use one to bm3d and it can improve psnr from 35.+ to 37.+.


