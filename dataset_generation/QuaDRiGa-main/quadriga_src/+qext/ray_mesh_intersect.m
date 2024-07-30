%RAY_MESH_INTERSECT Calculates the intersection points of rays with a triangle mesh in three dimensions
%
% Calling object:
%   None (static method)
%
% Description:
%   Rays are defined by their origin orig and their destination dest. This function calculates the
%   intersection points of those rays with a triangular mesh. It implements the Möller–Trumbore
%   ray-triangle intersection algorithm, named after its inventors Tomas Möller and Ben Trumbore, a
%   fast method for calculating the intersection of a ray and a triangle in three dimensions
%   without needing precomputation of the plane equation of the plane containing the triangle. This
%   function requires a NIVIDIA GPU with a compute capability of 3.5 or higher.
%
% Input:
%   orig
%   Ray origin in global Cartesian coordinates using units of [m]; single precision; 
%   Dimensions: [ no_ray x 3 ]
%
%   dest
%   Ray destination in global Cartesian coordinates using units of [m]; single precision;
%   Dimensions: [ no_ray x 3 ]
%
%   mesh
%   Vertices of the triangular mesh in global Cartesian coordinates. Each face (triangle) is
%   described by 3 points in 3D-space. Hence, a face requires 9 values in the order [ v1x, v1y,
%   v1z, v2x, v2y, v2z, v3x, v3y, v3z ]. Values are in single precision; 
%   Dimensions: [ no_mesh x 9 ]
%
%   no_hit_W
%   Maximum number of interaction points to be returned in output 'W'. This variable has no effect
%   on the outputs no_trans, fbs or lbs. Default: 1
%
%   verbose
%   Progress report: (0) Disable, (1) Bar plot, default, (2) Full Report
%
% Output:
%   no_trans
%   Number of interaction of a ray with the mesh; uint32; Dimensions: [ no_ray x 1 ]
%
%   fbs
%   First interaction point of a ray with the mesh; single; Dimensions: [no_ray x 3 ]
%
%   lbs
%   Last interaction point of a ray with the mesh; single; Dimensions: [ no_ray x 3 ]
%
%   iFBS
%   Index of the first mesh element that was hit by the ray; uint32;  Dimensions: [ no_ray x 1 ]
%
%   W
%   Normalized intersection points on the line between orig and dest. A value of 0 corresponds to
%   the origin, a value of 1 corresponds to the destination. Values are returned in the order in
%   which they are found. Dimensions: [ no_hit_W x no_ray ]
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
