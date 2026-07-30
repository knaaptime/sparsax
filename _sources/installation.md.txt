# Installation

sparsax supports Python >= 3.12. We recommend using [miniforge] or [pixi].

## Installing a released version

`sparsax` is available on PyPI and can be installed with:

```bash
pip install sparsax
```


## Installing from source

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/knaaptime/sparsax.git
cd sparsax
conda env create -f environment.yml
conda activate sparsax
pip install -e . --no-deps
```

## Verifying the installation

```python
import sparsax
print(sparsax.__version__)
```

[miniforge]: https://github.com/conda-forge/miniforge
[pixi]: https://pixi.sh