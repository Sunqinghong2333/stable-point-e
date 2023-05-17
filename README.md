# Stable Point-E

We implemented the training code based on [OpenAI/Point-E](https://github.com/openai/point-e). In terms of data, we provide colored point cloud data from the ShapeNet dataset along with descriptive text. As for the model, we offer a checkpoint trained on the aforementioned training data.

## Data Preparation

You need to download pcl.tar from [GoogleDrive](https://drive.google.com/drive/folders/1r3R_p3AY5JajpCXZpPpVXw0MFWt7frac?usp=sharing) and extract it to the [data/shapenet](data/shapenet) folder.

## Usage

Install with `pip install -e .`.

To get started with aforementioned training data:

 * [train.sh](point_e/train.sh) - training shell based on slurm, you can run as following: `sh train.sh 32 train {your_slurm_partition} texts 5 point_e_save point_e/configs/shapenet.json`

To visualize:
* [vis.sh](point_e/vis.sh) - sample point clouds, conditioned on text prompts, you can run as following: `sh vis.sh {your_slurm_partition}`


For P-FID and P-IS evaluation scripts, see:

 * [evaluate_pfid.py](point_e/evals/scripts/evaluate_pfid.py)
 * [evaluate_pis.py](point_e/evals/scripts/evaluate_pis.py)


## Acknowledgement

* Thanks [OpenAI](https://arxiv.org/abs/2212.08751) for implementing the [Point-E](https://github.com/openai/point-e) codebase.

* Thanks [Yongqiang Yao](https://github.com/yqyao) for the implementation of [Distributed PyTorch](https://github.com/ModelTC/United-Perception).
