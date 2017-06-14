from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import re
import time
import numpy as np
from . import _gpuselect as gs
import logging
if sys.version_info[0] >= 3:
    from configparser import ConfigParser
    from py3nvml import nvidia_smi
else:
    # python2
    from ConfigParser import ConfigParser
    import nvidia_smi
logger = logging.getLogger("gpuselect")


def get_gpu(gpu_weight, mem_weight):
    n_gpus = gs.n_gpus()
    G, M = [], []
    for i in range(n_gpus):
        bus_id = gs.bus_id(i)
        h = nvidia_smi.nvmlDeviceGetHandleByPciBusId(bus_id.encode('ascii'))
        memutil, gpuutil = [], []
        for k in range(100):
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(h)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(h)
            memutil.append(float(memory.used) / memory.total)
            gpuutil.append(float(util.gpu) / 100)
            time.sleep(.01)
        G.append(np.mean(gpuutil))
        M.append(np.mean(memutil))
    print("GPU Utilization:", G)
    print("Mem Utilization:", M)
    print("GPU Weight     :", gpu_weight)
    print("Mem Weight     :", mem_weight)
    return np.argmin(gpu_weight*np.array(G) + mem_weight*np.array(M))


def get_default_device():
    dev = 'cpu'
    fn = os.path.expanduser("~/.theanorc")
    if os.path.exists(fn):
        config = ConfigParser()
        with open(fn) as f:
            config.readfp(f)
            cfg_dev = config.get("global", "device")
            if cfg_dev is not None:
                dev = cfg_dev
    if 'THEANO_FLAGS' in os.environ:
        res = re.match(r'.*\bdevice=(\w+)', os.environ['THEANO_FLAGS'])
        if res:
            dev = res.group(1)
    return dev


device = get_default_device()
if device in ('gpu', 'cuda'):
    nvidia_smi.nvmlInit()
    print("default is", device)

    gpu_weight = float(os.environ.get('GPUSELECT_GPU_WEIGHT', 2))
    mem_weight = float(os.environ.get('GPUSELECT_MEM_WEIGHT', 1))
    gpu = get_gpu(gpu_weight, mem_weight)
    if 'THEANO_FLAGS' in os.environ:
        flags = os.environ['THEANO_FLAGS']
    else:
        flags = ""
    os.environ['THEANO_FLAGS'] = flags + ",device=%s%d" % (device, gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print("Using device gpu", gpu)

if __name__ == "__main__":
    import theano
    print(theano.config.device)
