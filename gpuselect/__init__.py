import os
import re
import time
import numpy as np
import _gpuselect as gs
import nvidia_smi
import logging
import ConfigParser
logger = logging.getLogger("gpuselect")

def get_gpu():
    n_gpus = gs.n_gpus()
    G, M = [], []
    for i in xrange(n_gpus):
        bus_id = gs.bus_id(i)
        h = nvidia_smi.nvmlDeviceGetHandleByPciBusId("0000:%d:00.0" % bus_id)

        memutil, gpuutil = [], []
        for k in xrange(100):
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(h)
            memutil.append(util.memory)
            gpuutil.append(util.gpu)
            time.sleep(.01)
        G.append(np.mean(gpuutil))
        M.append(np.mean(memutil))
    print "GPU Utilization:", G
    print "Mem Utilization:", M
    return np.argmin(2*np.array(G) + np.array(M))

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
        gpu = get_gpu()
        if 'THEANO_FLAGS' in os.environ:
            flags = os.environ['THEANO_FLAGS']
        else:
            flags = ""
        os.environ['THEANO_FLAGS'] = flags + ",device=gpu%d" % gpu
        print "Using device gpu", gpu

if __name__ == "__main__":
    import theano
    print theano.config.device
