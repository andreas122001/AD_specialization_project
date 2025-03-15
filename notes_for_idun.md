# Notes for Carla Garage

## Notes for running TF++ on IDUN HPC

### SLURM Modules

These modules are needed (I don't remember what libjpeg is for):

```bash
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0
```

### Conda environment

When I tried installing the provided conda environment directly, some weird errors with importing torch in python occur: `ImportError: libnccl.so.2: cannot open shared object file: No such file or directory`. They (original authors) fix it by just loading the shared lib in a script from another module on the system, but I don't know where I could find that on IDUN, so here is the fix:

(1) First of all, we need to use the pytorch and nvidia channels to install PyTorch, as conda-forge does not ship with cuda (I think it's deprecated as pytorch wants you to use pip with venv-s). So we need to specify using `pytorch` and `nvidia` channels *before* `conda-forge`.

(2) Now, installing PyTorch via the requirements.txt file did not work, but adding `pytorch=2.5.0` to the conda dependencies worked. We could then also remove the line `torch==2.5.0` from requirements.txt, but it doesn't really matter, it will jsut say "already installed".

So this would be the diff:

```diff
- name: garage_2
+ name: lb2  # I also renamed it here because I wanted to
channels:
+  - pytorch
+  - nvidia
  - conda-forge
  - defaults
dependencies:
  - ca-certificates
  - freetype
  - libpng
  - python=3.10.15
+  - pytorch=2.5.0
  - pip
  - pip:
    - -r team_code/requirements.txt
```

Now, hopefully it will install correctly, try it:
```sh
conda clean -py  # sometimes doesn't work without this
conda config --add channels defaults  # "defaults" might not already be added
conda env create -f environment.yml -y  # create env
conda activate lb2  # activate env
python -c "import torch; print(torch.cuda.is_available())"  # test CUDA
```

It should print **True** now.


### Changes in evaluate_routes_slurm_tfpp.py

At the very start of the script, they include the fix for the conda stuff above. We fixed it, and can just remove that part:
```diff
- # Our centOS is missing some c libraries.
- # Usually miniconda has them, so we tell the linker to look there as well.
- newlib = '/mnt/lustre/work/geiger/bjaeger25/home/miniconda3/lib'
- if not newlib in os.environ['LD_LIBRARY_PATH']:
-   os.environ['LD_LIBRARY_PATH'] += ':' + newlib
``` 

In the SBATCH part of the script we need to add the account (since we don't want to use up the IDI quota). Additionally, the CUDA version we are using doesn't support the Tesla P100-GPUs for some reason, so we need to add a GPU constraint. Add this:
```diff
+ #SBATCH --account=share-ie-idi
+ #SBATCH --constraint=(a100|h100)
```

Update the parameter parser defaults to reflect your workflow, or just set them when you run the program, idc.

Hint: make the default value of the epochs parameter a list instead of a tuple, because else it is going to treat the name you specify as a string and you will get some weird errors when it creates a symlink like "file does not exist" even though the file verifiably *does* exist, it is just a broken symlink because it linked to a file "m.pth" when it should be "model_0030_0.pth" because that is the first index of the "list" that is actually string and you might legitimately and understandably go crazy.

```diff
- default=("model_0030"),
+ default=["model_0030"],
```


## Possible cleanups

In `model.py`:
1. Separate forward into two methods, and select in __init__ by which backbone is used to avoid the control flow. There is more control flow other places though.

