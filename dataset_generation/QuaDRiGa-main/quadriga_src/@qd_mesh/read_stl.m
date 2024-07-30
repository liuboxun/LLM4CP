function read_stl( h_mesh, fname, mat_id )
%READ_STL Reads mesh data from a STL-File
%
% Calling object:
%   Single object
%
% Description:
%   STL files describe only the surface geometry of a three-dimensional object without any
%   representation of color, texture or other common CAD model attributes. The STL format specifies
%   both ASCII and binary representations. Binary files are more common, since they are more
%   compact. This method reads the mesh data to a STL file (ASCII or binary format).
%
% Input:
%   fname
%   Path to the STL File
%
%   mat_id
%   Material index in the existing 'qd_mesh' object
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
    error('QuaDRiGa:qd_mesh:read_stl','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist( 'fname','var' ) || isempty( fname )
   error('QuaDRiGa:qd_mesh:read_stl','Filename is not given.'); 
end

if ~exist( 'mat_id','var' ) || isempty( mat_id )
    mat_id = 1;
elseif mat_id > numel( h_mesh.mtl_name )
    error('QuaDRiGa:qd_mesh:read_stl',['Meterial ID ',num2str(mat_id),' is not defined in material list.']);
end

fileID = fopen(fname, 'r');                         % Open File
solid = fread(fileID,5,'*uint8');                   % Read first 5 bytes

mesh = single([]);                                  % Empty mesh

if all( solid == uint8( [115;111;108;105;100] ) )   % ASCII File
    
    fgetl(fileID);                                  % Read header
    chunksize = 12000000;                           % has to be a multiple of 3
    nn = chunksize*3;
    while nn == chunksize*3
        [XYZ, nn] = fscanf(fileID,...
            'facet normal %*f %*f %*f\nouter loop\nvertex %f %f %f\nvertex %f %f %f\nvertex %f %f %f\nendloop\nendfacet\n',...
            [3, chunksize]);
        mesh = [ mesh ; reshape( single(XYZ), 9, nn/9 )' ]; %#ok
    end
    
else % Binary format
    
    fread(fileID,75,'*uint8');
    no_face = fread(fileID,1,'*uint32');
    data = fread(fileID,[25,no_face],'*uint16');
    data = data(7:24,:);
    mesh = reshape( typecast(data(:),'single'), [], no_face )';
    
end
fclose( fileID ); 


% Set default object index
h_mesh.obj_name = [ h_mesh.obj_name, fname ];
obj_mesh_index = uint32( h_mesh.no_obj );

% Calculate vertices and faces
vert = reshape(mesh',3,[]);
face = reshape( uint32(size(h_mesh.vert,2)) + uint32(1:size(vert,2)), 3,[] );

% Write to object
i_face = h_mesh.no_face + 1;
h_mesh.vert = [ single(h_mesh.vert), vert ];
h_mesh.face = [ single(h_mesh.face), face ];
h_mesh.obj_index(i_face:end) = obj_mesh_index;
h_mesh.mtl_index(i_face:end) = uint32(mat_id);
   
end
