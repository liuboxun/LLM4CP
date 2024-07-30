function write_obj( h_mesh, fname, obj_id )
%WRITE_OBJ Writes mesh data from Wavefront .obj file format
%
% Calling object:
%   Single object
%
% Description:
%   The OBJ file format is a simple data-format that represents 3D geometry - namely, the position
%   of each vertex and the faces that make each polygon defined as a list of vertices. This method
%   writes the mesh data to a OBJ file.
%
% Input:
%   fname
%   Path to the OBJ File
%
% Input:
%   obj_id
%   A vector containing the object indices that should be shown. Default: All
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
    error('QuaDRiGa:qd_mesh:write_obj','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist( 'fname','var' ) || isempty( fname )
   error('QuaDRiGa:qd_mesh:write_obj','Filename is not given.'); 
end
if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = 1 : numel(h_mesh.obj_name);
end

% Parse filnames
[fdir,fname_mtl] = fileparts(fname);
fname_mtl = [fname_mtl,'.mtl'];
if numel( fname ) < 4 || ~strcmp( fname(end-3:end), '.obj' )
    fname = [fname,'.obj'];
end

% Init output
vert = {};
face = {};
mtl_id = [];
obj_name = {};

% Read obj data
m_face = uint32( h_mesh.face );
m_vert = single( h_mesh.vert );
obj_index = uint32( h_mesh.obj_index );
mtl_index = uint32( h_mesh.mtl_index );

io = 0;
for n = 1 : numel( obj_id )
    i_obj = obj_index == uint32(obj_id(n));                 % Find mesh entries for the current object
    if any( i_obj )
        mtl = sort( mtl_index( i_obj ) );                       % Does the object have different materials?
        mtl = mtl( [true,diff( mtl ) ~= 0] );
        for m = 1 : numel(mtl)
            io  = io + 1;
            i_face = i_obj & mtl_index == mtl(m);
            i_vert = m_face(:,i_face);
            i_vert = sort(i_vert(:));
            i_vert = i_vert([true;diff(i_vert)~=0]);
            vert{io} = m_vert(:,i_vert);
            
            % Write faces
            if all( diff( i_vert ) == 1 )
                face{io} = m_face(:,i_face) - (i_vert(1) - 1);
            else
                f1 = m_face(:,i_face);
                for o = 1 : numel( i_vert )
                    f1( f1 == i_vert(o) ) = o;
                end
                face{io} = f1;
            end
            
            mtl_id(1,io) = mtl(m);
            if numel(mtl) == 1
                obj_name{io} = h_mesh.obj_name{ obj_id(n)};
            else
                obj_name{io} = [h_mesh.obj_name{ obj_id(n)},'.',h_mesh.mtl_name{mtl(m)}];
            end
        end
    end
end

% Write OBJ File
fid = fopen(fname, 'w');
fprintf(fid, '# QuaDRiGa v%s OBJ File: ''%s''\r\n',qd_simulation_parameters.version, fname);
fprintf(fid, '# quadriga-channel-model.de\r\n');
fprintf(fid, 'mtllib %s\r\n', fname_mtl);
vert_cnt = uint32(0);
for n = 1 : numel(obj_name)
    fprintf(fid, 'o %s\r\n', obj_name{n} );
    fprintf(fid, 'v %g %g %g\r\n', vert{n} );
    fprintf(fid, 'usemtl %s\r\n', h_mesh.mtl_name{ mtl_id(n) } );
    fprintf(fid, 'f %d %d %d\r\n', face{n}+vert_cnt );
    vert_cnt = vert_cnt + uint32( size(vert{n},2) );
end
fclose( fid );

% Write MTL File
fid = fopen(fullfile(fdir,fname_mtl), 'w');
fprintf(fid, '# QuaDRiGa v%s MTL File: ''%s''\r\n',qd_simulation_parameters.version, fname_mtl);
fprintf(fid, '# Material Count: %d\r\n',numel(h_mesh.mtl_name));
for n = 1 : numel( h_mesh.mtl_name )
    fprintf(fid, '\r\nnewmtl %s\r\n', h_mesh.mtl_name{n} );
    fprintf(fid, 'Kd %g %g %g\r\n', h_mesh.mtl_color(:,n) );
    fprintf(fid, 'Ni %g\r\n', sqrt( h_mesh.mtl_prop(1,n)*h_mesh.mtl_prop(3,n) ) );
    fprintf(fid, 'd 1.000000\r\n' );
    fprintf(fid, 'illum 2\r\n' );
end
fclose( fid );

end

