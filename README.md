# REPULSIVE SCORE DISTILLATION

This is the README file for the implementaiton of repulsive score distillation method in the paper https://arxiv.org/abs/2305.04391. 
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



https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/assets/37513044/bb6c6cf1-2347-4a9c-a9ff-f28248ceff60



<img src="https://github.com/nzilberstein/higher-order-langevin/blob/main/figures/discretization_methods.png" width="500" height="425">


### Diversity for inverse problems

Image inpaiting half face:

Original image and noisy measurement

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/bip_large_00055_input_deg-1.png" width="800" height="425">


Reconstuction (top is RSD without repulsion and bottom is RSD with repulsion)

<img src="https://github.com/nzilberstein/Repulsive-score-distillation-RSD-/blob/main/figures/_00055_repo-1.png" width="800" height="425">


## Installation

cd <root>
git clone https://gitlab-master.nvidia.com/mmardani/red-diff.git .


## Unconstrained sampling

For unconstrained sampling, we include:

1. ProlificDreamer2D for text-to-image generation (see below in Acknowledgment). For the requirements, you can check in their repo to install the conda environment
2. DreamFusion
3. Noise-free score distillation (for this case we couldn't replicate the results from the original paper)
4. Threestudio implementation for text-to-3D (see below in Acknowledgment) 




## Constrained sampling


#### Description


In this part we solve inverse problem using the augmentation and the repulsion term. 
For the configuration, we use Omega and Hydra.

####  Installation.

To install the packages, you should run


```
pip install -r requirements.txt
```

We recommend to create a new environment to avoid issues with versions.


#### Datasets. We ran experiments using FFHQ.

 
For simplicity, you can download the .zip file in this link 


#### Checkpoints.

We use hugginface for stablediffusion.
Please make sure you're logged in with huggingface-cli login , in order to download the Stable Diffusion-1.5 model.

#### Usage

We have a set of bash files to run different experiments. The most general is solve_inverse_stable_single.sh, where the user can use an arbitrary image (of 512 or 256) as input: you need to specify 

- exp.img_path="PATH" \
- exp.img_id="IMG.png" \

Then there are other bash files to run large scale experiments on FFHQ.

To specify the degradation, you need to specify 

- deg='inp_random' (see utils/degradations for different types. The options are [sr4, deblur, deblur_gauss, deblur_uni, phase_retrieval, deblur_nl, hdr])

Regarding the hyperparamters, there are some expeirments that might need a fine-tunning of the hyperparametrs.
For other methods, we use the same logic from RED-diff repository.

## Citations and acknowledgments

Credits go to a few repos that we used: particle guidance (https://github.com/gcorso/particle-guidance/tree/main), prolific dreamear 2d (https://github.com/yuanzhi-zhu/prolific_dreamer2d) and RED-diff (https://github.com/NVlabs/RED-diff/tree/master).
For text-to-3D, we used the threestudio framework (https://github.com/threestudio-project/threestudio).
