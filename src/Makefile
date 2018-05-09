SHELL := /bin/bash # Use bash syntax

CC = sm_20

NVCC	= nvcc -Wno-deprecated-gpu-targets -arch=$(CC)
KERNEL	= Babis_Kernel
ARGS	= -ptx

all: kernel

kernel:
	$(NVCC) $(ARGS) $(KERNEL).cu

clean:
	rm -rf *~ *.ptx
