# MHBNN-PyTorch

A resnet and alexnet implementation of Multi-view Harmonized Bilinear Netowrk for 3D Object Detection(MHBN) inpsired by [Tan Yu et al](http://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Multi-View_Harmonized_Bilinear_CVPR_2018_paper.html). 

In this paper, the 3D object recognition problem is converted to multi-view 2D image classification problem. For each 3D object, there are multiple images taken from different views. 

![](https://github.com/LiyuanLacfo/MHBNN-PyTorch/blob/master/mhbn.png)

### Dependecies

* torch 0.4.1
* torchvision
* numpy

### Dataset

* ModelNet CAD data can be found at [Princeton](http://modelnet.cs.princeton.edu/)
* ModelNet40 12-view png images can be downloaded at [google drive](https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view?usp=sharing)
* You can also create your own png dataset with [Blend](https://github.com/WeiTang114/BlenderPhong)

### Train the model

```
python main.py --data <path to your png data>
```

### Special Thanks

I refered to [RBirkeland](https://github.com/RBirkeland/MVCNN-PyTorch) for some code.

