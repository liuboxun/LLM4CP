function write_par( h_mesh, fname_par, fname_lib, obj_id )
%WRITE_PAR Writes mesh data to a PAR-File
%
% Calling object:
%   Single object
%
% Description:
%   The PAR file format is a very simple data-format that represents 3D geometry - namely, the
%   positions of the vertices that make each polygon. In addition to the PAR-File, a LIB-File
%   contains the material properties. PAR-Files are used by the RTOpterix RT-Engine. This method
%   writes the mesh data to a PAR file and material properties to a LIB file.
%
% Input:
%   fname_par
%   Path to the PAR File
%
%   fname_lib
%   Path to the LIB File (optional)
%
%   obj_id
%   A vector containing the object indices that should be written. Default: All
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
    error('QuaDRiGa:qd_mesh:write_par','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist( 'fname_par','var' ) || isempty( fname_par )
    error('QuaDRiGa:qd_mesh:write_par','Filename is not given.');
else
    ind = regexp( fname_par,'.par','once' );
    if isempty( ind )
        fname_par = [fname_par,'.par'];
    end
end

if ~exist( 'fname_lib','var' ) || isempty( fname_lib )
    fname_lib = [fname_par(1:end-4),'.lib'];
end

if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = true(1,numel(h_mesh.Pobj_index));
else
    obj_index = uint32( h_mesh.obj_index );
    ii = false( size( obj_index ) );
    for n = 1 : numel( obj_id )
        ii = ii | obj_index == uint32( obj_id(n) );
    end
    obj_id = ii;
end


% Material library index list
mtl_mesh_index = h_mesh.mtl_index(obj_id);

% Write material library
fid = fopen(fname_lib, 'w');
fprintf(fid, '#PARAY\r\n');
fprintf(fid, '#sourcefile: %s\r\n', fname_par);
fprintf(fid, '#created: %s\r\n', datestr(now) );
for im = 1 : numel( h_mesh.mtl_name )
    if any( mtl_mesh_index == uint32(im) )
        no_layers = sum( h_mesh.mtl_thickness(:,im) > 0 );
        fprintf(fid,'\nMLW, %s, %d,',h_mesh.mtl_name{im}, no_layers );
        for il = 1 : no_layers
            prop = [ h_mesh.mtl_thickness(il,im), h_mesh.mtl_prop( :,im,il )' ];
            fprintf(fid,' %g %g %g %g %g',prop);
            if il ~= no_layers
                fprintf(fid,',');
            end
        end
        fprintf(fid,'\n$VRML %s diffuseColor %g %g %g\n', h_mesh.mtl_name{im}, h_mesh.mtl_color(:,im)' );
    end
end
fclose( fid );

% Write faces
mesh = h_mesh.mesh(obj_id,:);
fid = fopen(fname_par, 'w');
fprintf(fid, '#PARAY\r\n');
fprintf(fid, '#sourcefile: %s\r\n', fname_par);
fprintf(fid, '#created: %s\r\n', datestr(now) );
for im = 1 : numel( h_mesh.mtl_name )
    if any( mtl_mesh_index == uint32(im) )
        fprintf(fid, ['P, ',h_mesh.mtl_name{im},', 0, 3, %.10g %.10g %.10g, %.10g %.10g %.10g, %.10g %.10g %.10g\n'],...
            mesh(mtl_mesh_index == uint32(im), : )');
    end
end
fclose( fid );

end
