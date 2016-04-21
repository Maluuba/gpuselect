#include <cuda_runtime.h>
#include <cuda.h>

#include <boost/python.hpp>

int n_gpus(){
    int cnt;
    cudaGetDeviceCount(&cnt);
    return cnt;
}
void reset(){
    cudaDeviceReset();
}
int bus_id(int deviceId){
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, deviceId);
    return dp.pciBusID;
}
double mem_utilization(int deviceId){
    cudaSetDevice(deviceId);
    size_t total,free;
    cudaMemGetInfo(&total, &free);
    double mem_util = free/(double)total;
    return mem_util;
}

BOOST_PYTHON_MODULE(_gpuselect) {
    using namespace boost::python;
    def("n_gpus", n_gpus);
    def("reset", reset);
    def("bus_id", bus_id);
    def("mem_utilization", mem_utilization);
}
