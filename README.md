# Koopa


This is the official codebase for the paper: [Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors](https://arxiv.org/pdf/2305.18803.pdf), Neurips 2023. 

## Introduction

Koopa is a **lightweight**, **MLP-based**, and **theory-inspired** model for efficient time series forecasting. 

- Compared with the advanced but painstakingly trained deep forecasters, Koopa achieves state-of-the-art performance while saving **77.3%** training time and **76.0%** memory footprint.

<p align="center">
<img src="./figures/efficiency.png" height = "240" alt="" align=center />
</p>

- Focus on portraying ubiquitous **non-stationary** time series, Koopa shows **enhanced model capacity** empowered by the modern Koopman theory that naturally addresses the nonlinear evolution of real-world time series.
  
<p align="center">
<img src="./figures/motivation.png" height = "180" alt="" align=center />
</p>

- Koopa differs from the canonical Koopman Autoencoder without the reconstruction loss function to achieve **end-to-end predictive training**.
  
<p align="center">
<img src="./figures/architecture.png" height = "360" alt="" align=center />
</p>


## Preparation

1. Install Python 3.7 and the necessary dependencies.
```
pip install -r requirements.txt
```
2. All the six benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).

## Training scripts

We provide the Koopa experiment scripts and hyperparameters of all benchmark datasets under the folder `./scripts`.

```bash
bash ./scripts/ECL_script/Koopa.sh
bash ./scripts/Traffic_script/Koopa.sh
bash ./scripts/Weather_script/Koopa.sh
bash ./scripts/ILI_script/Koopa.sh
bash ./scripts/Exchange_script/Koopa.sh
bash ./scripts/ETT_script/Koopa.sh
```

## Applicable for Rolling Forecast

- By adapting the operator on the incoming time series during rolling forecast, the proposed model can achieve more accurate performance via adapting to continuous distribution shift.

- The naïve implementation of operator adaptation is based on the DMD algorithm. We propose an iterative algorithm with reduced complexity. The details can be found in the Appendix of our paper.

<p align="center">
<img src="./figures/algorithm.png" height = "480" alt="" align=center />
</p>

## Discussions

There are already serveral discussions about our paper, we appreciate the authors a lot for their valuable efforts and comments.

* [Openreview] https://openreview.net/forum?id=jsanMaAxZE
* [Zhihu] https://www.zhihu.com/question/24189178/answer/3064876852

## Citation

If you find this repo useful, please cite our paper. 

```
@article{liu2023koopa,
  title={Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors},
  author={Liu, Yong and Li, Chenyu and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2305.18803},
  year={2023}
}
```

## Contact

If you have any questions or want to use the code, please contact:
* liuyong21@mails.tsinghua.edu.cn
* lichenyu20@mails.tsinghua.edu.cn