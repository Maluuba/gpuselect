GPUSelect

import this before theano, if your .theanorc or your THEANO_FLAGS environment
variable specify device=gpu, it will 

1. find the device with  lowest utilization
2. set the THEANO_FLAGS environment variable so that theano chooses this device

Installation

pip install git+ssh://git@github.com/temporaer/gpuselect@master#egg=gpuselect
