# REPULSIVE SCORE DISTILLATION (RSD)

This is the README file for the implementaiton of repulsive score distillation method in the paper ..
It provides an overview of the project, instructions for installation and usage, and other relevant information.

Given that our framework was applied to unconstrained case (text-to-image, text-to-3D) and constrained (inverse problems), there are two different folders, one for each case.

Our proposed method (for constrained sampling) is summarized in the following figure

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/scheme2-1.png" width="800" height="425">

In a nutshell, we incorporate a diversity-seeking regularizer in score distillation sampling. 
We apply this formulation to unconstrained sampling, in the sense that we generate a scene or an image from a prompt, and constrained sampling, where we aim to compute an estimation given a noisy measurement. 
This term is a repulsion term, which is added to the score matching regularization.

We include experiments for both text-to-image and text-to-3D experiments, and for inverse problems.
Notice that our formulation for inverse problems relies on half-quadratic splitting; we described in more details below.

Examples of the results for different experiments are shown below


### Diversity for text-to-3D



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/2492ebef-842e-43a8-988d-522619a8a9fe



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/eaf5870b-511f-4014-a476-663c0c0db12d



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/deb57841-e73b-4230-b2ff-c827445d6b2b



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/c1f03cfb-820a-4fb4-9f25-6a6cde64ed1f


### Diversity for inverse problems

Image inpaiting half face:

Original image and noisy measurement

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/bip_large_00055_input_deg-1.png" width="800" height="425">


Reconstuction (top is RSD without repulsion and bottom is RSD with repulsion)

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/_00055_repo-1.png" width="800" height="425">



More results can be found in this link:
https://drive.google.com/drive/folders/1D2Oh7imsXl0xZN65vwcJYHSqQaXhxvIN?usp=sharing

## Installation

```
cd <root>
git clone https://github.com/nzilberstein/Repulsive-score-distillation-RSD-.git
```

## Unconstrained sampling

For unconstrained sampling, we include:

1. ProlificDreamer2D for text-to-image generation (see below in Acknowledgment). For the requirements, you can check in their repo to install the conda environment
2. DreamFusion
3. Noise-free score distillation (for this case we couldn't replicate the results from the original paper)
4. Threestudio implementation for text-to-3D (see below in Acknowledgment). The only difference wrt the implementation from threestudio is in the file Repulsive-score-distillation-RSD-/unconstrained_sampling/threestudio/threestudio/models/guidance/deep_floyd_guidance.py, where we have the repulsion term that we added. The flag to activate and desactivate the repulsion is an attribute of the class (it is hard-coded, we will change soon).


## Constrained sampling


#### Description


In this part we solve inverse problem using the augmentation and the repulsion term. 
For the configuration, we use Omega and Hydra.
Therefore, you need to change the root path in _configs/exp/default.

####  Installation.

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


#### Datasets. We ran experiments using FFHQ.

 
For simplicity, you can download the .zip file in this link https://drive.google.com/drive/u/1/folders/1fnjqWu8gaPZfS0niVmtBoNuncM_O302-.


#### Checkpoints.

We use hugginface for stablediffusion.
Please make sure you're logged in with huggingface-cli login, in order to download the Stable Diffusion model.

#### Usage

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

An example with inpainting with a particular image can be obtain by running

```
sh solve_inverse_stable_single.sh
```

The results will be saved in the _exp folder.
For large scale experiments, there are two notebooks in the folder datasets that can be used compute the PSNR and LPIPS, as well diversity.

Regarding the hyperparamters, there are some expeirments that might need a fine-tunning of the hyperparametrs.
The relationship between the parameters of the bash file and the paper is as follow:

```
rho_t corresponds to 1/rho^2
w_t corresponds to lambda
```

For other methods, we use the same logic from RED-diff repository.

## Some more examples of text-to-3D



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/876e5096-e172-486d-aacb-5c5d9761f7db



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/6f2db4f2-378e-4b17-a509-f7117e4b9ffb



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/8c90ba05-599b-4c2a-b918-5493fde71ad3



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/dec6ea5a-c503-461e-8fd4-ae6553f94561


## Citations

If you find our work interesting, please consider citing

## Contact

For any inquiries, please contact nzilberstein@rice.edu

## Acknowledgments

Credits go to a few repos that we used as templates: particle guidance (https://github.com/gcorso/particle-guidance/tree/main), prolific dreamear 2d (https://github.com/yuanzhi-zhu/prolific_dreamer2d) and RED-diff (https://github.com/NVlabs/RED-diff/tree/master).
For text-to-3D, we used the threestudio framework (https://github.com/threestudio-project/threestudio).
