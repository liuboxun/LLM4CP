function simplify( h_mesh, obj_id, tolerance, verbose )
%SIMPLIFY Removes co-located vertices from the mesh
%
% Calling object:
%   Single object
%
% Description:
%   This function iterates through all vertices and removes those vertices from the list that have
%   identical coordinates. The face IDs are updated accordingly. Faces that collapse to a line or
%   point are removed from the qd_mesh object. This is useful when, e.g., reading data from a PAR
%   or STL file, where faces are defined by the coordinated of the 3 vertices.
%
% Input:
%   obj_id
%   A vector containing the object indices that should be simplified. Default: All
%
%   tolerance
%   The maximum distance between vertices at which they are considered separate.
%
%   verbose
%   Enables (1, default) or disables (0) the progress bar.
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
    error('QuaDRiGa:qd_mesh:simplify','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = 1 : numel(h_mesh.obj_name);
end
obj_id = uint32(obj_id);

if ~exist('tolerance','var') || isempty( tolerance )
    tolerance = 0.01;
end

obj_index = uint32( h_mesh.obj_index );
no_obj = uint32( numel(h_mesh.obj_name) );
if max(obj_index) > no_obj
    error('QuaDRiGa:qd_mesh:simplify','Object indices do not match the numer of objects.'); 
end

if ~exist('verbose','var') || isempty( verbose )
    if numel(obj_index) < 5e4
        verbose = false;
    else
        verbose = true;
    end
end

if verbose
    fprintf('Simplify     [');
    vb_dots = 50;
    tStart = clock;
    m0=0;
end

io = 0;
vert = single( h_mesh.vert );
face = uint32( h_mesh.face );
mtl_index = uint32( h_mesh.mtl_index );

% Remove faces collapsing to a line or point
ix = any(diff(sort(face)) == 0);
face = face(:,~ix);
obj_index = obj_index(:,~ix);
mtl_index = mtl_index(:,~ix);

vert_out = {};
face_out = {};
n_face = numel(obj_index);
j_face = 0;
for n = 1 : numel( obj_id )
    i_obj = obj_index == obj_id(n);                         % Find mesh entries for the current object
    if any( i_obj )                                         % Only process if there are any vertices
        io = io + 1;
        
        f = face( :,i_obj );
        i_vert = sort(f(:));
        i_vert = i_vert([true;diff(i_vert)~=0]);
        v = vert(:,i_vert);
        
        m_face = size(f,2);
        m_vert = size(v,2);
        
        iv = 0;
        id = true( 1,m_vert );
        vc = nan(  3,m_vert,'single' );
        while any( id )
            iv = iv + 1;
            ic = find(id,1);
            vv = v(:,ic);
            vc(:,iv) = vv;
            
            % Find co-located vertices
            ix = id;
            ix(ix) = abs( v(1,ix) - vv(1)) > tolerance;
            ix = ~ix & id;
            ix(ix) = sum( ( v(:,ix) - repmat( vv, 1, sum(ix) ) ).^2 ) < tolerance^2;
            
            % Update list of faces
            f( f == i_vert( ic ) ) = iv;
            if sum( ix ) > 1
                ix = find( ix );
                for m = 2 : numel( ix )
                    f( f == i_vert(ix(m)) ) = iv;
                end
            end
            id(ix) = false;
            
            % Update progress bar
            i_face = sum( ~id )/m_vert * m_face;
            if verbose; m1=ceil((j_face+i_face)/n_face*vb_dots); if m1>m0
                    for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
        end
        vert_out{io} = vc(:,1:iv);
        face_out{io} = f;
    end
    j_face = j_face + sum(i_obj);
end
        
% Write output
h_mesh.Pvert = cat(2,vert_out{:});
i_face = uint32(1);
i_vert = uint32(0);
for n = 1 : io
    j_face = i_face + uint32( size(face_out{n},2) ) - 1;
    face( :,i_face:j_face ) = face_out{n}+i_vert;
    i_face = j_face + 1;
    i_vert = i_vert + uint32( size(vert_out{n},2) );
end

% Remove faces collapsing to a line or point
ix = any(diff(sort(face)) == 0);
h_mesh.face = face(:,~ix);
h_mesh.obj_index = obj_index(:,~ix);
h_mesh.mtl_index = mtl_index(:,~ix);

if verbose
    fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
end

end
