# Deep_Graph_Representation_via_Adaptive_Homotopy_Contrastive_Learning

Deep_Graph_Representation_via_Adaptive_Homotopy_Contrastive_Learning
---

### 目录

1. [Abstract](#Abstract)
2. [Environment](#Environment)
3. [Attention](#Attention)
4. [How2train](#How2train)
5. [Reference](#Reference)

### Abstract

Homotopy model is an excellent tool exploited by diverse research works in the field of machine learning. However, its
flexibility is limited due to lack of adaptiveness, i.e., manual fixing or tuning the appropriate homotopy coefficients.
To address the problem above, we propose a novel adaptive homotopy framework (AH) in which the Maclaurin duality is
employed, such that the homotopy parameters can be adaptively obtained. Accordingly, the proposed AH can be widely
utilized to enhance the homotopy-based algorithm. In particular, in this paper, we apply AH to contrastive learning (
AHCL) such that it can be effectively transferred from weak-supervised learning (given label priori) to unsupervised
learning, where soft labels of contrastive learning are directly and adaptively learned. Accordingly, AHCL has the
adaptive ability to extract deep features without any sort of prior information. Consequently, the affinity matrix
formulated by the related adaptive labels can be constructed as the deep Laplacian graph that incorporates the topology
of deep representations for the inputs. Eventually, extensive experiments on benchmark datasets validate the superiority
of our method.

### Environment

that codes of all the experiments are implemented under the PyTorch-1.4.0 and python-3.7.9 on an Ubuntu 18.04.2 LTS
server with an NVIDIA GeForce GTX 1080Ti GPU.

#### pip requirements.txt

torchkeras==2.1.2\
numpy==1.19.2\
torch==1.4.0\
~~ipy==1.5.2\
torchvision==0.5.0\
tqdm==4.51.0\
munkres==1.1.1\
imageio==2.9.0\
matplotlib==3.3.2\
Pillow==8.2.0\
scikit_learn==0.24.2\
scipy==1.6.3\

#### conda env.yaml

- python=3.7.9=h60c2a47_0
- tqdm=4.51.0=pyhd3eb1b0_0

### Attention

**We have adapted two kinds of training methods of datasets!**

- .mat : formed by Matlab. Loaded by `./datautils/load_mat.py `

```python
    mat = io.loadmat(config.train_path)
data = mat['X']
label = mat['Y']
```

- .png(img) : normal image format. Loaded by `datautils/dataloader_adjacent.py `

### How2train

- 1>Firstly, configure the hyperparameters in `config.py`. As for two hyper-parameter, we set self.margin(m) → 0.6 ∼ 0.8
  and self.Mul_MSN(γ) → 0.01 ∼ 0.001. And The larger the number of clusters, the smaller M should be.
- 2>run train.py as `python train.py`
