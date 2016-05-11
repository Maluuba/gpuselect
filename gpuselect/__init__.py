import os
import re
import time
import numpy as np
import _gpuselect as gs
import nvidia_smi
import logging
import ConfigParser
logger = logging.getLogger("gpuselect")

def get_gpu(gpu_weight, mem_weight):
    n_gpus = gs.n_gpus()
    G, M = [], []
    for i in xrange(n_gpus):
        bus_id = gs.bus_id(i)
        h = nvidia_smi.nvmlDeviceGetHandleByPciBusId("0000:%02X:00.0" % bus_id)
        memutil, gpuutil = [], []
        for k in xrange(100):
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(h)
            procs = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(h)
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(h)
            memutil.append(float(memory.used) / memory.total)
            gpuutil.append(float(util.gpu) / 100)
            time.sleep(.01)
        G.append(np.mean(gpuutil))
        M.append(np.mean(memutil))
    print "GPU Utilization:", G
    print "Mem Utilization:", M
    print "GPU Weight     :", gpu_weight
    print "Mem Weight     :", mem_weight
    return np.argmin(gpu_weight*np.array(G) + mem_weight*np.array(M))

def get_default_device():
    dev = 'cpu'
    fn = os.path.expanduser("~/.theanorc")
    if os.path.exists(fn):
        config = ConfigParser.ConfigParser()
        with open(fn) as f:
            config.readfp(f)
            cfg_dev = config.get("global", "device")
            if cfg_dev is not None:
                dev = cfg_dev
    if 'THEANO_FLAGS' in os.environ:
        res = re.match(r'device=(\w+)', os.environ['THEANO_FLAGS'])
        if res:
            dev = res.group(1)
    return dev


device = get_default_device()
if device == 'gpu':
    nvidia_smi.nvmlInit()
    print "default is", device
    if device == 'gpu':
        gpu_weight = float(os.environ.get('GPUSELECT_GPU_WEIGHT', 2))
        mem_weight = float(os.environ.get('GPUSELECT_MEM_WEIGHT', 1))
        gpu = get_gpu(gpu_weight, mem_weight)
        if 'THEANO_FLAGS' in os.environ:
            flags = os.environ['THEANO_FLAGS']
        else:
            flags = ""
        os.environ['THEANO_FLAGS'] = flags + ",device=gpu%d" % gpu
        print "Using device gpu", gpu

if __name__ == "__main__":
    import theano
    print theano.config.device
