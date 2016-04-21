all: _gpuselect.so

gpuselect.o: gpuselect.cpp
	/usr/local/cuda-7.5/bin/nvcc -Xcompiler -fPIC -c -I/usr/local/cuda/include -I/usr/include/python2.7 -o gpuselect.o gpuselect.cpp

_gpuselect.so: gpuselect.o
	g++ -fPIC -shared gpuselect.o -L/usr/local/cuda/lib64 -lcudart -lcuda -lboost_python-py27 -lpython2.7 -o _gpuselect.so

clean:
	rm -f gpuselect.o _gpuselect.so
