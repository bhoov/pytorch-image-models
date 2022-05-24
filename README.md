## Setup

```
conda env create -f environment-dev.yml
conda env update -f environment-dcs.yml
conda activate timm
pip install -e .
pip install einops
pip install tqdm
pip install pytorch-lightning
```

Then we need to manually setup pytorch for the cluster.
```
pip install ~/scratch/pytorch1.9/torch-1.9.0-cp39-cp39-linux_ppc64le.whl
pip install ~/scratch/pytorch1.9/torchvision-0.10.0-cp39-cp39-linux_ppc64le.whl
```

And add the following lines to our .bashrc:

```
export LD_LIBRARY_PATH=~/scratch/pytorch1.9/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH
```
