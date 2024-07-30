%ICOSPHERE Constructs a geodesic polyhedron (icosphere), a convex polyhedron made from triangles
%
% Description:
%   An icosphere is constructed by subdividing faces of an icosahedron, a polyhedron with 20 faces,
%   12 vertices and 30 edges, and then projecting the new vertices onto the surface of a sphere.
%   The resulting mesh has 6 triangles at each vertex, except for 12 vertices which have 5
%   triangles. The approximate equilateral triangles have roughly the same edge length and surface
%   area.
%
% Input:
%   no_div
%   Number of divisions per edge of the generating icosahedron. The resulting number of faces is
%   equal to no_face = 20 · no_div^2
%
% Output:
%   dest
%   Position of the center point of each triangle; single precision;  Dimensions: [ no_face x 3 ]
%
%   length
%   Length of the vector pointing from the origin to the center point. This number is smaller than
%   1 since the triangles are located inside the unit sphere; single precision; 
%   Dimensions: [ no_face x 1 ]
%
%   trivec
%   The 3 vectors pointing from the center point to the vertices of each triangle; the values are
%   in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; single precision; 
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
