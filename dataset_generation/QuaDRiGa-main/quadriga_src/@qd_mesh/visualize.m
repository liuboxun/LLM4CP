function han = visualize( h_mesh, obj_id, create_new_figure )
%VISUALIZE Plots the polygon mesh
%
% Calling object:
%   Single object
%
% Description:
%   This method visualizes the mesh by plotting all faces into a 3D plot. Optionally, it is
%   possible to reuse an existing plot, e.g., a plot created by 'qd_layout.visualize', and plot the
%   3D mesh on top of it.
%
% Input:
%   obj_id
%   A vector containing the object indices that should be shown. Default: All
%
%   create_new_figure
%   If set to 0, no new figure is created, but the layout is plotted in the currently active
%   figure. Default value: 1 (create new figure)
%
% Output:
%   han
%   The figure handle
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

if numel( h_mesh ) > 1
    error('QuaDRiGa:qd_mesh:visualize','Visualize is not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = true(size(h_mesh.obj_index));
else
    obj_index = uint32( h_mesh.obj_index );
    ii = false( size( obj_index ) );
    for n = 1 : numel( obj_id )
        ii = ii | obj_index == uint32( obj_id(n) );
    end
    obj_id = ii;
end

if exist('create_new_figure','var') && ~isempty( create_new_figure )
    if ~( all(size(create_new_figure) == [1,1]) && isnumeric(create_new_figure) )
        error('QuaDRiGa:qd_mesh:visualize','??? "create_new_figure" must be scalar and numeric.')
    end
else
    create_new_figure = true;
end

% Create a new figure
if create_new_figure
    han = figure('Position',[ 100 , 100 , 1000 , 700]);
end

mtl_index = uint32(h_mesh.mtl_index);
n_mtl = numel( h_mesh.mtl_name );
for i_mtl = 1 : n_mtl
    ii_mtl = mtl_index == uint32(i_mtl) & obj_id;
    if any( ii_mtl )
        C = h_mesh.mtl_color(:,i_mtl)';
        patch ('Faces', h_mesh.face(:,ii_mtl)', 'Vertices', h_mesh.vert', 'FaceColor', C, 'EdgeColor', C/2  );
    end
end

if ~create_new_figure
    axis equal
end

end
