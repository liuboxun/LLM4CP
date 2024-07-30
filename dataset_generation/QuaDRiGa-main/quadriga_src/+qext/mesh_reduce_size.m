%MESH_REDUCE_SIZE Reduces the mesh size
%
% Description:
%   The input mesh is provides as a list of triangle vertices. However, the mesh size is often much
%   lager than the volume covered by the ray origin and destination coordinates. Hence, a lot of
%   computation time can be saved by only considering the relevant part of the mesh. This function
%   computes the maximum and minimum x, y, and z coordinates of the covered volume and returns a
%   reduced mesh, containing only the no_red faces that are fall within the volume covered by the
%   ray start and end points.
%
% Input:
%   mesh
%   Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
%   points in 3D-space: [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; single precision;
%   Dimensions: [ no_mesh x 9 ]
%
%   orig
%   Ray origin in global Cartesian coordinates using units of [m]; optional; single precision;
%   Dimensions: [ no_ray x 3 ]
%
%   dest
%   Ray destination in global Cartesian coordinates using units of [m]; optional; single precision;
%   Dimensions: [ no_ray x 3 ]
%
% Output:
%   mesh_r
%   Reduced mesh; single precision; Dimensions: [ no_red x 9 ]
%
%   mesh_ind
%   A logical list indicating which mesh elements are used (true) of removed (false); 
%   Dimensions: [ no_mesh x 1]
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
