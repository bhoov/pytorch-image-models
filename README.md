## Setup

**WARNING: Cluster launching code is dependent on this repository being cloned into `~/Projects`.**

```
conda env create -f environment-dev.yml
conda env update -f environment-dcs.yml
conda activate timm
pip install -r requirements.txt
pip install -e .
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

## TIMM Modifications

We extend timm algorithms by doing the following:

- Adding our attention algorithm to `timm/models/energy_transformer.py`
- Adding an ALBERT version of the vision transformer to `timm/models/vision_transformer_albert.py`
- Exposing both of these in `timm/models/__init__.py` to allow creation the normal way, through `timm.create_model("*")`

Model configurations can be registered according to normal TIMM convention. Note that both of the model files contain many extraneous flags that were necessary for testing the model.

## Launching Training Code

We wrap SLURM behavior into the following new files:

- `timm/utils/sbatch_tools.py` :: The SBATCH command that launches arbitrary python code
- `timm/tools/distributed_train.sh` :: **NOTE** This file must be located at `~/Projects/timm/tools/distributed_train.sh`. A very simple sh script that exposes the necessary environment variables for pytorch lightning

Example launch scripts are included in `tools/submit_TAVT**.py`. This will handle the request for a certain number of nodes/GPUs and flags to the model. They must be invoked from within the `tools` folder as the scripts rely on relative paths, e.g.,:

```
cd tools
python submit_TAVT01.py
```
