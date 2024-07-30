function split_objects( h_mesh, obj_id, verbose )
%SPLIT_OBJECTS Splits objects that have no common vertices
%
% Calling object:
%   Single object
%
% Description:
%   This function iterates through all faces of an object and detects which faces share common
%   vertices. All faces that are connected in this way are bundled into a new object. In this way,
%   a large list of faces can be categorized into sub-objects. Hence, the (large) object is split
%   into smaller objects with a reduced number of faces.
%
% Input:
%   obj_id
%   A vector containing the object indices that should be split. Default: All
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
    error('QuaDRiGa:qd_mesh:split_objects','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

obj_name = h_mesh.obj_name;
obj_index = uint32( h_mesh.obj_index );
no_obj = uint32( numel(obj_name) );

if max(obj_index) > no_obj
    error('QuaDRiGa:qd_mesh:split_objects','Object indices do not match the numer of objects.');
end
if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = 1 : no_obj;
end
if ~exist('verbose','var') || isempty( verbose )
    if numel(obj_index) < 5e4
        verbose = false;
    else
        verbose = true;
    end
end

if verbose
    fprintf('Split Obj.   [');
    vb_dots = 50;
    tStart = clock;
    m0=0;
end

n_face = numel(obj_index);

j_face = 0;
for n = 1 : numel(obj_id)
    io = obj_index == obj_id(n);
    if sum( io ) > 1
        face = uint32( h_mesh.face(:,io) );
        obj  = zeros( 1,size(face,2),'uint32' );
        cnn  = false( size(face) );
        sn   = sum(cnn);
        i_obj = uint32( 0 );
        while any( sn < 3 )
            i_face = find( sn == 2 | sn == 1 ,1);   % Search already connected vertices
            if isempty( i_face )
                i_face = find(sn == 0,1);
            end
            
            cnn = cnn | face == face(1,i_face) | face == face(2,i_face) | face == face(3,i_face);
            sn  = sum(cnn);
            if all( sn == 3 | sn == 0 )
                i_obj = i_obj + 1;
                obj( obj == 0 & sn == 3 ) = i_obj;
            end
            
            % Update progress bar
            if verbose; m1=ceil((j_face+i_face)/(n_face)*vb_dots); if m1>m0
                    for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
        end
        if i_obj > 1
            io = find(io);
            obj_name_base = obj_name{obj_id(n)};
            for m = 2 : i_obj
                no_obj = no_obj + 1;
                obj_name{no_obj} = [obj_name_base,'.',num2str(m-1,'%03d')];
                obj_index( io(obj==m) ) = no_obj;
            end
        end
        j_face = j_face + numel(obj);
    end
    
end

h_mesh.obj_name = obj_name;
h_mesh.obj_index = obj_index;

if verbose
    fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
end

end
