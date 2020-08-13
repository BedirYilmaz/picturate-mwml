# picturate
Based on https://github.com/suetAndTie/cycle-image-gan
Paper https://arxiv.org/abs/2003.12137

### A MadeWithML 2020 Summer Incubator Camp Project
* BERT encoder
* Cycle-GAN
* Image2Text encoder

### Try it on Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YkrdEw34m6Pfx2axJvDNsfzfcOJBOx-5?usp=sharing)


### Download Data
1. Download AttnGAN preprocessed data and captions [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ)
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`

### Instructions
* pretrain STREAM
```
python pretrain_STREAM.py --cfg cfg/STREAM/bird.yaml --gpu 0
```
* train CycleGAN
```
python main.py --cfg cfg/bird_cycle.yaml --gpu 0
```
