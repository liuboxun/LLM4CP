function shift( h_mesh, dist, axis, obj_id )
%SHIFT Moves objects along the principal coordinate axes
%
% Calling object:
%   Single object
%
% Input:
%   dist
%   Distance in [m]
%
%   axis
%   String describing the axis (x, y, z)
%
%   obj_id
%   A vector containing the object indices that should be simplified. Default: All
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
    error('QuaDRiGa:qd_mesh:shift','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if exist('axis','var')
    if ~( ischar(axis) && ...
            ( strcmp(axis,'x') || strcmp(axis,'y') || strcmp(axis,'z') ) )
        error('QuaDRiGa:qd_mesh:shift','??? "axis" can only be x,y or z.')
    end
else
    error('QuaDRiGa:qd_mesh:shift','??? "axis" is not given.')
end

% Read for faster access
vert = h_mesh.vert;

% Select faces that that should be moved
if ~exist('obj_id','var') || isempty( obj_id )
    ip = true(1,size(vert,2));
else
    i_face = ismember( uint32( h_mesh.obj_index ), uint32( obj_id ) );
    face = uint32( h_mesh.face );
    
    % List of vertices belonging to the selected objects
    ip = face(:,i_face);
    ip = sort(ip(:));
    ip = ip([true;diff(ip)~=0]);
    
    % List of vertices not belonging to the selected objects
    in = face(:,~i_face);
    in = sort(in(:));
    in = in([true;diff(in)~=0]);
    
    % List of vertices belonging to both groups
    ix = intersect(in,ip);
    
    % Copy duplicate vertices, if needed
    if ~isempty( ix )
        iy = uint32(1:numel(ix))' + uint32( size(vert,2) );
        vert = [ vert, vert(:,ix )];
        i_face = repmat( i_face,3,1 );
        for n = 1 : numel(ix)
            face( face == ix(n) & i_face ) = iy(n);
            ip( ip == ix(n) ) = iy(n);
        end
        h_mesh.face = face;
    end
end

% Move vertices
switch axis
    case 'x'
        vert(1,ip) = vert(1,ip) + dist;
    case 'y'
        vert(2,ip) = vert(2,ip) + dist;
    case 'z'
        vert(3,ip) = vert(3,ip) + dist;
end
h_mesh.vert = vert;

end

