all: HysterisisMM.so HysterisisMM.cpython-36m-x86_64-linux-gnu.so

HysterisisMM.so: HysterisisMM.cu HysterisisMM.h
	/usr/local/cuda-9.1/bin/nvcc --compiler-options '-fPIC' -shared -o libHysterisisMM.so HysterisisMM.cu -std=c++11 -lcurand -lcublas
	
HysterisisMM.cpython-36m-x86_64-linux-gnu.so: SetupHysterisisMM.py HysterisisMM.pyx
	python SetupHysterisisMM.py build_ext --inplace
	
clean:
	rm -f HysterisisMM.cpython-36m-x86_64-linux-gnu.so libHysterisisMM.so HysterisisMM.cpp
	rm -r build
