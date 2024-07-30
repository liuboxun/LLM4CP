%TEST_GPU_ACCESS Tests if there is a suitable NVIDIA GPU for accelerated computations
%
% Description:
%   This functions tries to access the NVIDIA-GPU through the CUDA libraries and perform a simple
%   calculation on the GPU. Upon success, the compute capability of the GPU is returned (See:
%   "https://developer.nvidia.com/cuda-gpus"). If the CUDA libraries are installed, but no GPU
%   is found, a value of 0 is returned. However, if the CUDA-libraries are not installed (which
%   would be the default case on a system without a NVIDIA-GPU), an error is returned stating that
%   the shared libraries cannot be found.
%
% Output:
%   cc
%   The compute capability of the detected GPU, returns 0 if no GPU was found.
%
%
% QuaDRiGa Copyright (C) 2011-2021
% Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
% Fraunhofer Heinrich Hertz Institute, Einsteinufer 37, 10587 Berlin, Germany
% All rights reserved.
%
% e-mail: quadriga@hhi.fraunhofer.de
%
% This file is part of QuaDRiGa.
%
% The Quadriga software is provided by Fraunhofer on behalf of the copyright holders and
% contributors "AS IS" and WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, including but not limited to
% the implied warranties of merchantability and fitness for a particular purpose.
%
% You can redistribute it and/or modify QuaDRiGa under the terms of the Software License for 
% The QuaDRiGa Channel Model. You should have received a copy of the Software License for The
% QuaDRiGa Channel Model along with QuaDRiGa. If not, see <http://quadriga-channel-model.de/>.
