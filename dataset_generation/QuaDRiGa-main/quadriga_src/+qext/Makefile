# This Makefile is for Linux / GCC environments

# Set path to your CUDA installation (required):
# You can get CUDA from: https://developer.nvidia.com/cuda-toolkit
CUDA_PATH = /usr/local/cuda-11.7

# If needed, adjust the NVIDIA compute capability (50 should run on most modern GPUs). 
# Adjusting the value to match your GPUs capability may improve performance and load times, but is not required.
# Minimum supported capability for CUDA-11 is 35 and for CUDA-12 it is 50.
# For more info: https://developer.nvidia.com/cuda-gpus
COMPUTE_CAPABILITY = 50

# Set path to your MATLAB installation (optional):
# Leave this empty if you don't want to use MATLAB (you can still use Octave).
MATLAB_PATH = /usr/local/MATLAB/R2021a

# The following sections should not require editing.

# Conditional compilation of MATLAB targets (check if MATLAB path is set)
ifeq ($(MATLAB_PATH),)
	MATLAB_TARGETS =
else
	MATLAB_TARGETS = mex_matlab
endif

# Check if Octave is installed by trying to run mkoctfile
OCTAVE_VERSION := $(shell mkoctfile -v 2>/dev/null)

# Conditional compilation of Octave targets
ifeq ($(OCTAVE_VERSION),)
	OCTAVE_TARGETS =
else
	OCTAVE_TARGETS = mex_octave
endif

# Compilers
CC   = g++
NVCC = $(CUDA_PATH)/bin/nvcc
MEX  = $(MATLAB_PATH)/bin/mex
OCT  = mkoctfile

# NVIDIA libraries and includes
NV_LIB     = $(CUDA_PATH)/lib64
NV_INCLUDE = $(CUDA_PATH)/include

# Compiler flags
CCFLAGS	  = -fPIC
NVCCFLAGS = -ccbin=$(CC) --gpu-architecture=compute_$(COMPUTE_CAPABILITY) --gpu-code=compute_$(COMPUTE_CAPABILITY) \
            -Wno-deprecated-gpu-targets -Xcompiler '-fPIC' -I$(NV_INCLUDE)

# Compilations targets
.PHONY: dirs
all:
	@$(MAKE) dirs
	@$(MAKE) $(MATLAB_TARGETS)   $(OCTAVE_TARGETS)

mex = $(wildcard mex/*.cpp)
mex_matlab:  dirs $(mex:mex/%.cpp=%.mexa64)
mex_octave:  dirs $(mex:mex/%.cpp=%.mex)

dirs:
	mkdir -p build

# Object files
build/qd_mesh_functions.o:   src/qd_mesh_functions.cpp   src/qd_mesh_functions.h
	$(CC) $(CCFLAGS) -c $< -o $@

build/ray_mesh_intersect.o:   src/ray_mesh_intersect.cu   src/ray_mesh_intersect.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/test_gpu_access.o:   src/test_gpu_access.cu   src/test_gpu_access.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/ray_mesh_intersect_link.o:   build/ray_mesh_intersect.o
	$(NVCC) -dlink $< -o $@ $(NVLIBS)

build/qext.a:   build/qd_mesh_functions.o   build/ray_mesh_intersect.o   build/ray_mesh_intersect_link.o \
                build/test_gpu_access.o
	ar rcs $@ $^

# MEX MATLAB interface
%.mexa64:   mex/%.cpp   build/qext.a
	$(MEX) CXXFLAGS="$(CCFLAGS)" $^ -L$(NV_LIB) -lcudart

# MEX Octave interface
%.mex:   mex/%.cpp   build/qext.a
	CXXFLAGS="$(CCFLAGS)" $(OCT) --mex -o $@ $^ -L$(NV_LIB) -lcudart

clean:
	rm -rf build/*
	- rmdir build
	rm -rf *.log

tidy:   clean
	rm -rf *.mex*
