%MESH_SUBDIVIDE Subdivides the faces of a triangle mesh into smaller faces
%
% Calling object:
%   None (static method)
%
% Description:
%   This function splits the triangles of a mesh into smaller triangles by subdividing the edges
%   into no_div sub-edges. This creates no_div^2 sub-faces per face.
%
% Input:
%   mesh
%   Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
%   points in 3D-space: [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; single precision;
%   Dimensions: [ no_mesh x 9 ]
%
%   no_div
%   Number of divisions per edge of the input mesh. The resulting number of faces is equal to
%   no_face = no_mesh · no_div^2
%
% Output:
%   mesh_d
%   Vertices of the sub-divided mesh in global Cartesian coordinates; single precision;
%   Dimensions: [ no_face x 9 ]
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
