function read_par( h_mesh, fname_par, fname_lib )
%READ_PAR Reads mesh data from a PAR-File
%
% Calling object:
%   Single object
%
% Description:
%   The PAR file format is a very simple data-format that represents 3D geometry - namely, the
%   positions of the vertices that make each polygon. In addition to the PAR-File, a LIB-File
%   contains the material properties. PAR-Files are used by the RTOpterix RT-Engine. This method
%   reads the mesh data to a PAR file and material properties from a LIB file.
%
% Input:
%   fname_par
%   Path to the PAR File
%
%   fname_lib
%   Path to the LIB File
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
    error('QuaDRiGa:qd_mesh:read_par','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist( 'fname_par','var' ) || isempty( fname_par )
    error('QuaDRiGa:qd_mesh:read_par','Filename is not given.');
end
if ~exist( 'fname_lib','var' ) || isempty( fname_lib )
    error('QuaDRiGa:qd_mesh:read_par','Filename of Material library is not given.');
end

% Parse material library
fid = fopen(fname_lib, 'r');
tline = fgetl(fid);
if strcmp(tline, '#PARAY')
    while ~feof(fid)
        tline = fgetl(fid);
        if numel(tline) > 5 && strcmp(tline(1:5), 'MLW, ')
            mlw = regexp( tline, ', ','split' );
            if numel(mlw) > 3
                h_mesh.mtl_name(1,end+1) = mlw(2);
                color = fscanf(fid, ['$VRML ',h_mesh.mtl_name{end},' diffuseColor %f %f %f\n'] );
                if isempty(color)
                    h_mesh.mtl_color(:,end) = h_mesh.mtl_color(:,1);      % Use default color
                else
                    h_mesh.mtl_color(:,end) = color;
                end
                im = numel( h_mesh.mtl_name );                          % Current material
                for il = 1 : str2double( mlw{3} )
                    prop = regexp( mlw{3+il}, ' ','split');
                    prop = cellfun(@str2double,prop);
                    h_mesh.mtl_thickness( il,im ) = prop(1);
                    h_mesh.mtl_prop( :,im,il ) = prop(2:end)';
                end
            end
        end
    end
end
fclose( fid );

% Parse vertices
fid = fopen(fname_par, 'r');

% Position file pointer at the beginning of the data fields
tline = fgetl(fid);
if strcmp(tline, '#PARAY')
    while numel( tline ) < 4 || ~strcmp( tline(1:3), 'P, ' )
        ind = ftell(fid);
        tline = fgetl(fid);
    end
end

% Read Mesh
fseek( fid,ind,'bof' );
mesh = single([]);
chunksize = 4000000;
nn = chunksize*9;
while nn == chunksize*9
    [xyz,nn] = fscanf(fid, 'P, %*[a-zA-Z], 0, 3, %f %f %f, %f %f %f, %f %f %f\n',[9,chunksize] );
    mesh = [ mesh ; single(xyz') ]; %#ok
end

% Read Material Names
fseek( fid,ind,'bof' );
mtl = fscanf(fid, 'P, %[a-zA-Z,] 0, 3, %*f %*f %*f, %*f %*f %*f, %*f %*f %*f\n' );
mtl = char(mtl');
mtl = regexp(mtl,',','split');

fclose( fid );

% Assign materials to faces
if size(mesh,1) ~= numel(mtl)-1
   error('QuaDRiGa:qd_mesh:read_par','Ooops, something went wrong.');
end
mtl_mesh_index = ones(1,size(mesh,1),'uint32');
for im = 1 : numel( h_mesh.mtl_name )
    ind = strcmp( h_mesh.mtl_name{im}, mtl );
    mtl_mesh_index(1,ind(1:end-1)) = uint32( im );
end

% Set default object index
obj_mesh_index = uint32( numel( h_mesh.obj_name ) + 1 );
h_mesh.obj_name = [ h_mesh.obj_name, fname_par ];

% Calculate vertices and faces
vert = reshape(mesh',3,[]);
face = reshape( uint32(size(h_mesh.vert,2)) + uint32(1:size(vert,2)), 3,[] );

% Write to object
i_face = h_mesh.no_face + 1;
h_mesh.vert = [ single(h_mesh.vert), vert ];
h_mesh.face = [ single(h_mesh.face), face ];
h_mesh.mtl_index(i_face:end) = mtl_mesh_index;
h_mesh.obj_index(i_face:end) = repmat( obj_mesh_index,1,size(face,2));

end
