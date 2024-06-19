
<p align="center">
  <img src="https://github.com/Anonymous-for-double-blind-review/EF-IDD/blob/main/figures/setting.png" width=75% height=75%>
</p>

### Introduction
This repo contains the official PyTorch implementation of the paper: Exemplar-Free Incremental Deepfake Detection. (Ongoing...)

### Quick Start

- Get Code
```shell
 git clone https://github.com/anonymous-repository71/ForgeryCLIP](https://github.com/Anonymous-for-double-blind-review/EF-IDD.git
```
- Build Environment
```shell
cd EF-IDD
conda env create -f environment.yml
conda activate efidd
```


### Datasets

- Download the datasets: [FF++](https://github.com/ondyari/FaceForensics), [FFIW](https://github.com/tfzhou/FFIW), [DFDC-P](https://ai.meta.com/datasets/dfdc/), [CDF2](https://github.com/yuezunli/celeb-deepfakeforensics)
- Data preprocessing. For the all datasets, we use [RetainFace](https://github.com/biubug6/Pytorch_Retinaface) to align the faces. Use the code in directory ./preprocessing to get the preprocessed data.
