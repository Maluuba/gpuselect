PYV=$(shell python -c 'import sys; print("%d%d" % (sys.version_info[:2]))')
PYVDOT=$(shell python -c 'import sys; print("%d.%d" % (sys.version_info[:2]))')
CUDA_ROOT ?= "/usr/local/cuda"
all: _gpuselect.so

gpuselect.o: gpuselect.cpp
	${CUDA_ROOT}/bin/nvcc -Xcompiler -fPIC -c -I${CUDA_ROOT}/include -I/usr/include/python${PYVDOT} -o gpuselect.o gpuselect.cpp

_gpuselect.so: gpuselect.o
	g++ -fPIC -shared gpuselect.o -L${CUDA_ROOT}/lib64 -lcudart -lcuda -lboost_python-py${PYV} -lpython${PYVDOT} -o _gpuselect.so

clean:
	rm -f gpuselect.o _gpuselect.so
