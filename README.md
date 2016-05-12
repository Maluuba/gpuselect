GPUSelect
---------

if multiple GPUs are in one server, it is sometimes tricky to select the best
GPU. Especially, since the GPU ordering in nvidia-smi and the one used by the
driver do not match.

This module can be imported before theano, and will select the GPU with minimal
utilization.


Installation
------------

```bash
pip install git+https://github.com/temporaer/gpuselect@master#egg=gpuselect
```


Usage
-----

```python
import gpuselect
import theano
```

Configuration
-------------

1. ensure that `THEANO_FLAGS` or `.theanorc` has `device=gpu`
2. if necessary, set `GPUSELECT_GPU_WEIGHT` and/or `GPUSELECT_MEM_WEIGHT`
   environment variables. Default is 2 and 1, respectively.

