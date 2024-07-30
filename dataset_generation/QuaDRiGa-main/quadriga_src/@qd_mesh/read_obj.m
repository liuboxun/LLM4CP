function read_obj( h_mesh, fname )
%READ_OBJ Reads mesh data from Wavefront .obj file format
%
% Calling object:
%   Single object
%
% Description:
%   The OBJ file format is a simple data-format that represents 3D geometry - namely, the position
%   of each vertex and the faces that make each polygon defined as a list of vertices. This method
%   parses the OBJ file and reads the relevant mesh data into the calling 'qd_mesh' object. The
%   linked material library file name (mtllib) is read from the OBJ file and parsed separately. The
%   following conversions are made:
%
%   * Vertices (v) are stored as 'qd_mesh.vert'
%   * Face ids (f) are stored as 'qd_mesh.face'; texture coordinates and vertex normal are ignored
%   * Object names (o) are stored as 'qd_mesh.obj_name'
%   * The corresponding face ids belonging to this object as are stored as 'qd_mesh.obj_index'
%   * Materials (usemtl) are allocated to all faces belonging to an object
%   * The diffuse color (Kd) is read from the MTL file an stored as 'qd_mesh.mtl_color'
%   * The refraction index (Ni) is read from the MTL file and its squared value is used for the
%     relative permittivity. Conductivity is set to 0 and relative permeability is set to 1. 
%   * Material thickness is set to 0.1
%
% Input:
%   fname
%   Path to the OBJ File
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
    error('QuaDRiGa:qd_mesh:read_obj','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

use_octave = isempty( strfind( version,'R20' ) ); %#ok

if ~exist( 'fname','var' ) || isempty( fname )
   error('QuaDRiGa:qd_mesh:read_obj','Filename is not given.'); 
end

fid = fopen(fname, 'r');        % Open File

% Estimate the required space
fseek(fid,0,'eof');
file_size = ftell(fid);

% Read all vertices
fseek(fid,0,'bof');

no_vert = ceil(file_size/20/2);                     % 20 chars per line, half the lineas are vertices
vert = zeros(3,no_vert,'single');
i_vert = uint32(1);

face_format = true;                                 % Switch for face format
face = {};
obj_mtl_name = {};
obj_name = {};

i_face = uint32(1);
i_obj  = uint32(0);

obj_face_index = zeros(1,no_vert,'uint32');

mtl_file_name = '';
while ~feof(fid)
    l = fgets(fid);
    
    if l(1) == 'v'                                  % Look for vertice
        if l(2) == ' '                              % Only parse vertices, discart nodes and textures
            x = sscanf( l,'v %f %f %f' );           % Read vertice from line
            if ~isempty( x )                        % Found a vertice
                vert(:,i_vert) = single(x);         % Store it
                x = fscanf(fid,'v %f %f %f\n');     % Following lines are ofter vertices too
                if ~isempty( x )                    % Read them as a bunch
                    tmp = uint32(numel(x)/3);
                    vert(:,i_vert+(1:tmp)) = reshape(single(x),3,[]);
                    i_vert = i_vert + numel(x)/3 + 1;
                else
                    i_vert = i_vert+1;
                end
            end
        end
        
    elseif l(1) == 'f'                              % Look for face
        if face_format
            x = sscanf( l(3:end-1),'%d' );          % Try first face fromat
            if numel(x) == 1                        % The smallest face is a triangle - wrong format
                x = sscanf( l(3:end-1),'%d%*[/0-9]' );
                face_format = false;                % Switch format
            end
        else
            x = sscanf( l(3:end-1),'%d%*[/0-9]' );  % Try second face format (containing "/")
            if numel(x) == 1
                x = sscanf( l(3:end-1),'%d' );      % Try first face fromat
                face_format = frue;                 % Switch format
            end
        end
        face{i_face} = uint32(x); %#ok              % Save face
        obj_face_index( i_face ) = i_obj;           % Store object ID
        i_face = i_face + 1;
        
    elseif l(1) == 'o'                              % Look for object 
        i_obj = i_obj + 1;
        obj_name{i_obj} = sscanf( l(3:end),'%s' ); %#ok
        
    elseif l(1) == 'm' && numel(l) > 7 && strcmp( l(1:6),'mtllib' ) % Look for mtllib
        mtl_file_name = sscanf( l(8:end),'%s' );
        
    elseif l(1) == 'u' && numel(l) > 7 && strcmp( l(1:6),'usemtl' ) % Look for material
        if i_obj == 0
            i_obj = i_obj + 1;
            obj_name{i_obj} = 'Default'; %#ok
        end
        obj_mtl_name{i_obj} = sscanf( l(8:end),'%s' ); %#ok
    end
end
fclose( fid ); 

no_vert     = i_vert - 1;
no_face     = i_face - 1;
vert        = vert(:,1:no_vert);
obj_face_index = obj_face_index(1:no_face);


% Load material properties
if isempty( mtl_file_name ) 
    mtl_index  = ones(1,no_face,'uint32');  % Set default material for each face
else
    fid = fopen(fullfile( fileparts(fname),mtl_file_name), 'r');            % Open MTL File
    i_mtl = 0;
    mtl_name = {};
    mtl_color = zeros(3,1);
    mtl_Ni = 1.45;
    while ~feof(fid)
        l = fgets(fid);
        if l(1) == 'n' && numel(l) > 7 && strcmp( l(1:6),'newmtl' ) % Look for mtllib
            i_mtl = i_mtl + 1;
            mtl_name{1,i_mtl} = sscanf( l(8:end),'%s' ); %#ok
            mtl_color(:,i_mtl) = [0.8;0.8;0.8];
            mtl_Ni(i_mtl) = 1.45; %#ok
        elseif l(1) == 'K' && l(2) == 'd'
            mtl_color(:,i_mtl) = sscanf( l,'Kd %f %f %f' );
        elseif l(1) == 'N' && l(2) == 'i'
            mtl_Ni(i_mtl) = sscanf( l,'Ni %f' ); %#ok
        end
    end
    fclose( fid );
    
    % Write materials to h_mesh
    no_existing_mtl = h_mesh.no_mtl;
    if i_mtl ~= 0
        h_mesh.mtl_name = cat(2, h_mesh.mtl_name, mtl_name );
        h_mesh.mtl_color(:,no_existing_mtl+1:end) = mtl_color;
        h_mesh.mtl_prop(:,no_existing_mtl+1:end) = [ mtl_Ni.^2; zeros(1,i_mtl); ones(1,i_mtl); zeros(1,i_mtl) ];
        h_mesh.mtl_thickness(:,no_existing_mtl+1:end) = ones(1,i_mtl) * 0.1;
    end
    
    % Link materials to objects
    mtl_index  = zeros(1,no_face,'uint32');
    for n = 1 : i_obj
        ind = strcmp( mtl_name, obj_mtl_name{n} );
        if any( ind )
            mtl_index( obj_face_index == uint32(n) ) = uint32( find( ind, 1 ) );
        end
    end
    ind = mtl_index == 0;
    mtl_index = mtl_index + no_existing_mtl;
    mtl_index(ind) = 1;
end

% Triangularize all Polygons
face_shape = cellfun(@numel,face);
i_triangle = face_shape == 3;
i_polygon  = face_shape > 3;
if any( i_polygon ) 
    
    i_polygon = find( i_polygon );
    for n = 1 : numel( i_polygon )
        
        v = vert(:,face{i_polygon(n)});                     % Vertices of the current polygon
        scale = sum(v,2)./size(v,2);                        % Normalizaion factors
        v = v - repmat(scale,[1,size(v,2)]);                % Normalize
        
        % Calculate covariance 
        xx = sum(v(1,:).^2);
        xy = v(1,:) * v(2,:)';
        xz = v(1,:) * v(3,:)';
        yy = sum(v(2,:).^2);
        yz = v(2,:) * v(3,:)';
        zz = sum(v(3,:).^2);
        
        % Calculate determinants
        det_x = yy*zz-yz^2;
        det_y = xx*zz-xz^2;
        det_z = xx*yy-xy^2;
        det_max = max([det_x,det_y,det_z]);
        
        % Calculate normal vector components
        if det_x == det_max
            nx = det_x;
            ny = xz*yz - xy*zz;
            nz = xy*yz - xz*yy;
        elseif det_y == det_max
            nx = xz*yz - xy*zz;
            ny = det_y;
            nz = xy*xz - yz*xx;
        else
            nx = xy*yz - xz*yy;
            ny = xy*xz - yz*xx;
            nz = det_z;
        end
        
        % Normalize normal vector
        nn = 1/sqrt( nx^2 + ny^2 + nz^2 );          	
        nx = nx*nn;
        ny = ny*nn;
        nz = nz*nn;
        
        % Calculate rotation matrix to align normal vector with z-axis
        c  = 1/(1+nz);
        R = [ 1-c*nx^2, c*nx*ny, nx ; -c*nx*ny, 1-c*ny^2, ny ; -nx, -ny, 1-c*nx^2-c*ny^2 ];
        
        % Transform vertices so that all points lie in the x-y plane
        v = R'*v;
        
        if all( abs( v(3,:) ) < 0.1 )        % Points are in one plane (10 cm tolernace)
            if use_octave
                t = delaunay( v(1:2,:)' ,'Qz')';        % Triangularize Octave
            else
                t = delaunay( double(v(1:2,:))')';      % Triangularize Matlab
            end
            f = face{i_polygon(n)}(t');                 % Transformed faces
        else
            t = delaunay( double(v)' )';                % Tetrahedon
            f = face{i_polygon(n)}([ t([1,2,3],:), t([1,2,4],:), t([1,3,4],:), t([2,3,4],:) ]');
        end
               
        % Update facees
        face{i_polygon(n)} = f'; %#ok
        
        i_triangle(i_polygon(n)) = true;
    end
end

% Process triangles
if any( i_triangle )
    % Read object indices
    face_shape = cellfun(@numel,face);
    face_shape( ~i_triangle ) = 0;
    if any( face_shape ~= 3 )
        face_shape = round( face_shape/3 );
        fun = @(a,b) ones(1,a,'uint32').*b;
        obj_index = arrayfun( fun, face_shape, obj_face_index,'UniformOutput', false );
        obj_index = cat(2,obj_index{:});
        mtl_index = arrayfun( fun, face_shape, mtl_index,'UniformOutput', false );
        mtl_index = cat(2,mtl_index{:});
    else
        obj_index = obj_face_index;
    end
    
    % Write mesh
    no_exisiting_vert = uint32( h_mesh.no_vert );
    no_existing_face  = uint32( h_mesh.no_face );
    no_existing_obj   = uint32( h_mesh.no_obj );
    
    h_mesh.vert = [ h_mesh.vert, vert ];
    h_mesh.face = [ h_mesh.face,  cat( 2,face{i_triangle}  ) + no_exisiting_vert ];
    
    % Write objects to mesh
    h_mesh.obj_name = cat( 2, h_mesh.obj_name, obj_name );
    h_mesh.obj_index( no_existing_face+1:end ) = obj_index + no_existing_obj;
    
    % Write material index
    h_mesh.mtl_index( no_existing_face+1:end ) = mtl_index;
end

end
