# EvolutionaryGAN
We provided Theano implementations for [Evolutionary Generative Adversarial Networks (E-GAN)](https://arxiv.org/abs/1803.00657). Meanwhile, we are working on new [Pytorch-based implementations](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch).

## Getting started

- Clone this repo:
```bash
git clone git@github.com:WANG-Chaoyue/EvolutionaryGAN.git
cd EvolutionaryGAN
```

- Install [Theano 1.0.0+](http://deeplearning.net/software/theano/install.html), [lassagne 0.2+](https://lasagne.readthedocs.io/en/latest/user/installation.html) and other dependencies ([requirements.txt](https://github.com/WANG-Chaoyue/EvolutionaryGAN/blob/master/requirements.txt)).

### Datasets
The proposed E-GAN was trained on two synthesis dataset and three real-world datasets. Among them, the two mixture Gaussians datasets are adopted from [here](https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py). For three real-world datasets, you should download them first from [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html), [LSUN bedroom](http://lsun.cs.princeton.edu/2017/), and [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and then moving them into [datasets](https://github.com/WANG-Chaoyue/EvolutionaryGAN/tree/master/dataset). Then,
```bash
cd dataset
python dataset.py
```
The download CelebA or LSUN bedroom datasets can be converted to 'hdf5' files.

### Training
- Train a model (take cifar-10 as an example)
```bash
cd cifar10
python train_cifar10.py
```
**Note** that related hpyer-parameters can be configured within 'train_cifar10.py'
```bash
vim train_cifar10.py
```

## Citation
If you use this code for your research, please cite our paper.
```
@article{wang2018evolutionary,
  title={Evolutionary Generative Adversarial Networks},
  author={Wang, Chaoyue and Xu, Chang and Yao, Xin and Tao, Dacheng},
  journal={arXiv preprint arXiv:1803.00657},
  year={2018}
}
```

## Related links
[Evolving Generative Adversarial Networks | Two Minute Papers #242](https://www.youtube.com/watch?v=ni6P5KU3SDU&vl=en)
