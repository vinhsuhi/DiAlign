# DiGress: Discrete Denoising diffusion models for graph generation


Warning: The code has been updated after experiments were run for the paper. If you don't manage to reproduce the 
paper results, please write to us so that we can investigate the issue.

For the conditional generation experiments, check the `guidance` branch. 

## Environment installation
  - Download anaconda/miniconda if needed
  - Install graph-tool (optional) (https://graph-tool.skewed.de/): `conda install -c conda-forge graph-tool` 
  - Install the nvcc drivers for your cuda version. For example, `conda install -c "nvidia/label/cuda-11.3.1" cuda-nvcc`
  - Install pytorch 1.10 or 1.11 (https://pytorch.org/)
  - Install pytorch-geometric. Your version should match the pytorch version that is installed (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - Install other packages using the requirement file: `pip install -r requirements.txt`
  - Install mini-moses: `pip install git+https://github.com/igor-krawczuk/mini-moses`
  - Run `pip install -e .`



## Run the code
  
  before running the code, please change the path of datadir in configs/dataset/pascalvoc.yaml to your local machine path

  - All code is currently launched through `python src/align.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
    