# REPULSIVE LATENT SCORE DISTILLATION (RLSD)

This is the README file for the implementation of Repulsive Latent Score Distillation (RLSD), a method introduced in https://arxiv.org/abs/2406.16683.
It provides an overview of the project, instructions for installation and usage, and other relevant information.

Our proposed method for inverse problems is summarized in the following figure

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/scheme_main.png" width="800" height="425">

Our method is inspired in variational inference (score distillation), and is adapted to solve inverse problems in the latent space.

In a nutshell, we incorporate a diversity-seeking regularizer in score distillation sampling; this term is a repulsion term, which is added to the score matching regularization.
Furtermore, our formulation relies on half-quadratic splitting; we described in more details below.

An example for image inpaiting half face is shown below

### Original image and noisy measurement

<div align="center">
<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/bip_large_00055_input_deg-1.png" width="450" height="225">
</div>

### Reconstuction

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/bip_large_00055_names.png" width="800" height="425">

More results can be found in this link:
https://drive.google.com/drive/folders/1D2Oh7imsXl0xZN65vwcJYHSqQaXhxvIN?usp=sharing

## Installation

```
cd <root>
git clone https://github.com/nzilberstein/Repulsive-score-distillation-RSD-.git
```

## Description

In this part we solve inverse problem using the augmentation and the repulsion term. 
For the configuration, we use Omega and Hydra.
Therefore, you need to change the root path in _configs/exp/default.

##  Installation.

To install the packages, you have two optinos

1) Via conda. You should run

```
conda env create -f environment.yaml
conda activate stable-dif
```


2) Via pip. First you create an empty env and then you install via pip using the requirements.txt

```
pip install -r requirements.txt
```

We recommend to create a new environment to avoid issues with versions.


## Datasets. 

We ran experiments using FFHQ, and also ImageNet 512.
For simplicity, you can download the .zip file in this link https://drive.google.com/drive/u/1/folders/1fnjqWu8gaPZfS0niVmtBoNuncM_O302-.



## Checkpoints.

We use hugginface for stablediffusion.
Please make sure you're logged in with huggingface-cli login, in order to download the Stable Diffusion model.

## Usage

We have a set of bash files to run different experiments. The most general is solve_inverse_stable_single.sh, where the user can use an arbitrary image (of 512 or 256) as input: you need to specify 

```
exp.img_path="PATH" \
exp.img_id="IMG.png" \
```

Then there are other bash files to run large scale experiments on FFHQ.

To specify the degradation, you need to specify 

```
deg='inp_random' (see utils/degradations for different types. The options are [sr4, deblur, deblur_gauss, deblur_uni, phase_retrieval, deblur_nl, hdr]).
```

To use the repulsion, you have to set 

```
algo.dino_flag = True
```

Another important parameter for repulsion is sigma_break, which controls in how many noise levels there is repulsion: 999 means in all noise levels (if the total number is 200, 999 is automatically rescaled to 199)

An example with inpainting with a particular image can be obtain by running

```
sh runs/solve_inverse_stable_single.sh
```

The results will be saved in the _exp folder.


For large scale experiments, there are two notebooks in the folder datasets that can be used compute the PSNR and LPIPS, as well diversity.

Regarding the hyperparamters, there are some experiments that might need a fine-tunning of the hyperparametrs.
The relationship between the parameters of the bash file and the paper is as follow:

```
rho_t corresponds to \tilde{\rho}
w_t corresponds to lambda
```

For other methods, we use the same logic from RED-diff repository.


## Citations

If you find our work interesting, please consider citing as

```
@article{zilberstein2024repulsive,
  title={Repulsive Latent Score Distillation for Solving Inverse Problems},
  author={Zilberstein, Nicolas and Mardani, Morteza and Segarra, Santiago},
  journal={arXiv preprint arXiv:2406.16683},
  year={2024}
}
```

## Contact

For any inquiries, issues with running the code or suggestions to improve reproducibility, please contact nzilberstein@rice.edu or add an issue in the repository.

## Acknowledgments

Credits go to RED-diff (https://github.com/NVlabs/RED-diff/tree/master), which we used as template.
Also, for the guidance term, we inspired in particle guidance (https://github.com/gcorso/particle-guidance/tree/main).
