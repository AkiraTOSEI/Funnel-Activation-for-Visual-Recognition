# Funnel-Activation-for-Visual-Recognition
this repository is for check FReLU([arXiv:2007.11824](https://arxiv.org/abs/2007.11824)) on CIFAR10.
I have tested ReLU, Swish and FReLU 3times using ResNet18. 
The result is shown as following.

![frelu](https://github.com/AkiraTOSEI/Funnel-Activation-for-Visual-Recognition/blob/master/frelu.png)

|Activation Function|minimum validation loss|
|---|---|
|ReLU|0.764 ± 0.009|
|Swish|0.763 ± 0.008|
|__FReLU__|__0.743 ± 0.006__|

