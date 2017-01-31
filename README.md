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
pip install git+https://github.com/Maluuba/gpuselect@master#egg=gpuselect
```

For python3, ensure that your `libboost_python` is compiled for Python3.
If you are using conda, you can ensure this by doing `conda install boost` in
your Python3 environment.


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


Background
----------

The NVidia driver uses an ordering which seems to place faster GPUs first,
while nvidia-smi and NVML use an ordering based on bus-ID.
GPU-Utilization is best acquired using the NVML API.

Thus, the strategy employed by this script is:

1. Get the busID of every device in driver-ordering using nvml
2. Get gpu/memory utilization of device via nvml for given bus id
3. Weight gpu/memory utilization
4. Select the least utilized GPU, `X`
5. Append `device=gpuX` to `THEANO_FLAGS`.
6. You load theano and everything is good.
