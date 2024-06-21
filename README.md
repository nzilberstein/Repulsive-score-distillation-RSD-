# REPULSIVE SCORE DISTILLATION

This is the README file for the implementaiton of repulsive score distillation method in the paper https://arxiv.org/abs/2305.04391. 
It provides an overview of the project, instructions for installation and usage, and other relevant information.

Given that our framework was applied to unconstrained case (text-to-image) and constrained (inverse problems), there are two different folders, one for each case.

EXPERIMENTS 

## Unconstrained sampling



## Constrained sampling


Description

Installation

1. Hydra config tutorial



2. Datasets (where to download)



3. Checkpoints (where to download)

Usage

We have a set of bash files to run different experiments. The most general is solve_inverse_stable_single.sh, where the user can use an arbitrary image (of 512 or 256) as input: you need to specify 

- exp.img_path="PATH" \
- exp.img_id="IMG.png" \

Then there are other bash files to run large scale experiments on FFHQ.

To specify the degradation, you need to specify 

- deg='inp_random' (see utils/degradations for different types. The options are)

Regarding the hyperparamters, there are some expeirments that might need a fine-tunning of the hyperparametrs.


Citations and acknowledgments

Credits go to a few repos that we used: particle guidance, prolific dreamear 2d and RED-diff. We use also threestudio for the text-to-3D experiments.