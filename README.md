# Molecular Attributes Transfer from Non-Parallel Data
The code was built based on [MOSES](https://github.com/molecularsets/moses). Thanks a lot for their code sharing!

# Requirements
+ Python 3.6
+ PyTorch 1.6
+ CUDA 10.2
+ RDKit

# Quick start
## Installation
`python setup.py install`

## Dataset preparation
Unzip file `zinc_all.zip`, and run code `python build_dataset.py --path PATH` to preprocess the active molecules.

## Running MolSty
To train model using MolSty, use:

`python ./scripts/train.py MolSty --target TARGET --model_save ./checkpoints --n_batch 16 --n_ins 2`

where `<TARGET>` is the name of dataset.

To generate molecules using MolSty, use:

`sh generate.sh PATH_CHECKPOINT`

where, `<PATH_CHECKPOINT>` is the checkpoint path, for example, `./checkpoints/ckpt_100.pt`
