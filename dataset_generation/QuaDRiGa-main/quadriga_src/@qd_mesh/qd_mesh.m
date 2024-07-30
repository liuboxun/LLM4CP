classdef qd_mesh  < handle
%QD_MESH Implements data structures and methods to interact with 3D environment models
%
%   This class implements data structures and methods to interact with 3D environment models and
%   external ray tracing (RT) tools. The 3D envirnment is implemented as a polygon mesh, a
%   collection of vertices and faces that define the shape of polyhedral objects. All faces consist
%   of triangles and the model is represented as a face-vertex mesh, a simple list of vertices, and
%   a set of triangles that point to the vertices it uses. 
%
%
% QuaDRiGa Copyright (C) 2011-2019
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

properties
    % Name of the "qd_mesh" object
    name = 'Mesh';          % Name of the "qd_mesh" object
end
    
properties(Dependent)
    % Vertices of the mesh: [ 3 x no_vert] - [ X; Y; Z]
    %   List of all vertices (points in 3D space) of the 3D model. The three rows correspond to the
    %   x, y and z-coordinates of the vertices. Data is stored in single precision. 
    vert
    
    % List of triangular faces: [ 3 x no_mesh ] - uint32
    %   The indices pointing to a closed set of 3 vertices defining a triangle face. Indices are
    %   stored in uint32 format.
    face

    % Object index for each mesh face
    %   Faces can be grouped into objects, e.g. buildings, trees, etc. Each face is assigned an
    %   object index from the list of objects in the qd mesh class. Indices are stored in uint32
    %   format. 
    obj_index
    
    % Material index for each mesh face
    %   Each face gets assigned a material from the material list. Objects can consist of different
    %   materials, e.g. a building can have concrete walls and glass windows. Indices are stored in
    %   uint32 format. 
    mtl_index
    
    % Object names
    %   Cell array of object names
    obj_name
    
    % Name of each object; Cell array containing strings;
    obj_att_par
    
    % Name of each material; Cell array containing strings
    mtl_name
    
    % Material diffuse color
    %   The RGB-color of the material. This is used for visualization proposes using the qd
    %   mesh.visualize method. The rows correspond the red, green and blue component having values
    %   between 0 and 1.
    mtl_color
    
    % Electric properties of the material
    %   Row 1: Real part of the relative permittivity (default 1)
    %   Row 2: Imaginary part of the relative permittivity (default 0)
    %   Row 3: Real part of the relative permeability  (default 1)
    %   Row 4: Imaginary part of the relative permeability (default 0)
    mtl_prop
    
    % Material thickness
    %   Thickness of the material in [m]
    mtl_thickness

    no_vert     % Number of vertices
    no_face     % Number of faces
    no_obj      % Number of objects
    no_mtl      % Number of materials
end

% properties(Dependent,SetAccess=protected)
%     % Path to the RtOpteriX ray-tracing library
%     %   Only available for Linux, RtOpteriX-path must be on the MATLAB/Octave path and binary
%     %   executable must be compiled correctly) 
%     rtopterix_lib
% end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
end

properties(Hidden,Dependent)
    mesh
    mtl
end

properties(Access=private)
   Pvert = [];
   Pface = [];
   Pobj_index = [];
   Pmtl_index = [];
   Pobj_name = {'Default'};
   Pobj_att_par = [ 100; 0; 0; 0; 100 ];
   Pmtl_name = {'Default'};
   Pmtl_color = [ 0.8; 0.8; 0.8 ];
   Pmtl_prop = [ 2 ; -0.5 ; 1 ; 0 ];
   Pmtl_thickness = 0.1;
end

methods
    
    % Constructor
    function h_mesh = qd_mesh
        %QD_MESH Construct an instance of this class
        % Not yet defined
    end
    
    % Get functions
    function out = get.vert( h_mesh )
       out = h_mesh(1,1).Pvert;
    end
    function out = get.face( h_mesh )
       out = h_mesh(1,1).Pface;
    end
    function out = get.obj_index( h_mesh )
       out = h_mesh(1,1).Pobj_index;
    end
    function out = get.mtl_index( h_mesh )
       out = h_mesh(1,1).Pmtl_index;
    end
    function out = get.obj_name( h_mesh )
       out = h_mesh(1,1).Pobj_name;
    end
    function out = get.obj_att_par( h_mesh )
       out = h_mesh(1,1).Pobj_att_par;
    end
    function out = get.mtl_name( h_mesh )
       out = h_mesh(1,1).Pmtl_name;
    end
    function out = get.mtl_color( h_mesh )
       out = h_mesh(1,1).Pmtl_color;
    end
    function out = get.mtl_prop( h_mesh )
       out = h_mesh(1,1).Pmtl_prop;
    end
    function out = get.mtl_thickness( h_mesh )
       out = h_mesh(1,1).Pmtl_thickness;
    end
    function out = get.no_vert( h_mesh )
       out = size( h_mesh(1,1).Pvert ,2 );
    end
    function out = get.no_face( h_mesh )
       out = size( h_mesh(1,1).Pface ,2 );
    end  
    function out = get.no_obj( h_mesh )
       out = size( h_mesh(1,1).Pobj_name ,2 );
    end  
    function out = get.no_mtl( h_mesh )
       out = size( h_mesh.Pmtl_name ,2 );
    end
    % function out = get.rtopterix_lib( ~ )
    %     out = '';
    %     try %#ok
    %         rtopterix_path = path;                              % Parse files in config-folder
    %         if isempty( regexp( rtopterix_path,';' ) ) %#ok      % Linux separates path entries by ":"
    %             rtopterix_path = regexp(rtopterix_path, ':?([^:]*rtopterix)', 'tokens'); % Linux
    %         else                                                % Windows separated path entries by ";"
    %             rtopterix_path = regexp(rtopterix_path, ':?([^;]*rtopterix)', 'tokens'); % Windows
    %         end
    %         if ~isempty( rtopterix_path )
    %             lib_path = rtopterix_path{1}{1};
    %             command = [ fullfile( lib_path,'build','rtopterix' ),' --help' ];
    %             [~,cmdout] = system(command);
    %             if strcmp(cmdout(1:17),'rtopterix version')
    %                 out = lib_path;
    %             end
    %         end
    %     end
    % end
    function m = get.mesh( h_mesh )
        m = reshape( h_mesh(1,1).Pvert(:,h_mesh(1,1).Pface(:)), 9,[] )';
    end
    
    % Set functions
    function set.no_obj(h_mesh,value)
        if numel(value) ~= 1
            error('QuaDRiGa:qd_mesh','??? "no_obj" must be scalar.')
        end
        value = uint32( value );
        no_exist = uint32( size( h_mesh(1,1).Pobj_name ,2 ) ); 
        if value == 0
            error('QuaDRiGa:qd_mesh','??? "no_obj" canot be 0.')
        end
        if value < no_exist
            if ~isempty( h_mesh(1,1).Pobj_index )
                warning('QuaDRiGa:qd_mesh','??? Replacing removed objects with default (first) object.')
                h_mesh(1,1).Pobj_index( h_mesh(1,1).Pobj_index > value ) = uint32(1);
            end
            h_mesh(1,1).Pobj_name = h_mesh(1,1).Pobj_name(1:value);
            h_mesh(1,1).Pobj_att_par = h_mesh(1,1).Pobj_att_par(:,1:value);
        elseif value > no_exist
            new_name = cell( 1,value );
            new_name(1:no_exist) = h_mesh(1,1).Pobj_name;
            for n = no_exist+1 : value
                new_name{n} = ['obj',num2str(n)];
            end
            h_mesh(1,1).Pobj_name = new_name;
            h_mesh(1,1).Pobj_att_par = [ h_mesh(1,1).Pobj_att_par, repmat([ 100;0;0;0;100 ], 1, value-no_exist) ];
        end
    end
    
    function set.no_mtl(h_mesh,value)
        if numel(value) ~= 1
            error('QuaDRiGa:qd_mesh','??? "no_mtl" must be scalar.')
        end
        value = uint32( value );
        no_exist = uint32( size( h_mesh(1,1).Pmtl_name ,2 ) ); 
        if value == 0
            error('QuaDRiGa:qd_mesh','??? "no_mtl" canot be 0.')
        end
        if value < no_exist
            if ~isempty( h_mesh(1,1).Pmtl_index )
                warning('QuaDRiGa:qd_mesh','??? Replacing removed materials with default (first) material.')
                h_mesh(1,1).Pmtl_index( h_mesh(1,1).Pmtl_index > value ) = uint32(1);
            end
            h_mesh(1,1).Pmtl_name = h_mesh(1,1).Pmtl_name(1:value);
            h_mesh(1,1).Pmtl_color = h_mesh(1,1).Pmtl_color(:,1:value);
            h_mesh(1,1).Pmtl_prop = h_mesh(1,1).Pmtl_prop(:,1:value);
            h_mesh(1,1).Pmtl_thickness = h_mesh(1,1).Pmtl_thickness(:,1:value);
        elseif value > no_exist
            new_name = cell( 1,value );
            new_name(1:no_exist) = h_mesh(1,1).Pmtl_name;
            for n = no_exist+1 : value
                new_name{n} = ['mtl',num2str(n)];
            end
            h_mesh(1,1).Pmtl_name = new_name;
            h_mesh(1,1).Pmtl_color = [ h_mesh(1,1).Pmtl_color, repmat([0.8;0.8;0.8], 1, value-no_exist) ];
            h_mesh(1,1).Pmtl_prop = [ h_mesh(1,1).Pmtl_prop, repmat([2;-0.5;1;0], 1, value-no_exist) ];
            h_mesh(1,1).Pmtl_thickness = [ h_mesh(1,1).Pmtl_thickness, repmat(0.1, 1, value-no_exist) ];
        end
    end
    
    function set.vert(h_mesh,value)
        if size( value,1 ) ~= 3
            error('QuaDRiGa:qd_mesh','??? "vert" must have 3 rows.')
        end
        if size( value, 2 ) < size( h_mesh(1,1).Pvert,2 ) && numel( h_mesh(1,1).Pface ) > 1
            warning('QuaDRiGa:qd_mesh','??? Removing vertices invalidtes face list.')
            h_mesh(1,1).Pface = [];
            h_mesh(1,1).Pobj_index = [];
            h_mesh(1,1).Pmtl_index = [];
        end
        if isa(value,'single')
            h_mesh(1,1).Pvert = value;
        else
            h_mesh(1,1).Pvert = single(value);
        end
    end
    
    function set.face(h_mesh,value)
        if size( value,1 ) ~= 3
            error('QuaDRiGa:qd_mesh','??? "face" must have 3 rows.')
        end
        if ~isa(value,'uint32')
           value = uint32( value ); 
        end
        if max( value(:) ) > size( h_mesh(1,1).Pvert,2 )
            error('QuaDRiGa:qd_mesh','??? "face" index cannot exceed number of vertices.')
        end
        value( value == 0 ) = uint32(1);
        no_value = size( value, 2 );
        no_exist = size( h_mesh(1,1).Pface,2 );
        if no_value < no_exist
            h_mesh(1,1).Pobj_index = h_mesh(1,1).Pobj_index(:,1:no_value);
            h_mesh(1,1).Pmtl_index = h_mesh(1,1).Pmtl_index(:,1:no_value);
        elseif no_value > no_exist
            no_new = no_value - no_exist;
            h_mesh(1,1).Pobj_index = [ h_mesh(1,1).Pobj_index, ones(1,no_new,'uint32') ];
            h_mesh(1,1).Pmtl_index = [ h_mesh(1,1).Pmtl_index, ones(1,no_new,'uint32') ];
        end
        h_mesh(1,1).Pface = value;
    end
    
    function set.obj_index(h_mesh,value)
        no_exist = size( h_mesh(1,1).Pface,2 );
        if numel( value ) ~= no_exist
            error('QuaDRiGa:qd_mesh','??? Number of elements in "obj_index" must match number of faces.')
        end
        value = reshape( value, 1, [] );
        if ~isa(value,'uint32')
           value = uint32( value ); 
        end
        if max( value ) > size( h_mesh(1,1).Pobj_name ,2 )
            error('QuaDRiGa:qd_mesh','??? "obj_index" cannot exceed number of objects. Set objects first.')
        end
        value( value == 0 ) = uint32(1);
        h_mesh(1,1).Pobj_index = value;
    end
    
    function set.mtl_index(h_mesh,value)
        no_exist = size( h_mesh(1,1).Pface,2 );
        if numel( value ) ~= no_exist
            error('QuaDRiGa:qd_mesh','??? Number of elements in "mtl_index" must match number of faces.')
        end
        value = reshape( value, 1, [] );
        if ~isa(value,'uint32')
           value = uint32( value ); 
        end
        if max( value ) > size( h_mesh(1,1).Pmtl_name ,2 )
            error('QuaDRiGa:qd_mesh','??? "mtl_index" cannot exceed number of materials. Set materials first.')
        end
        value( value == 0 ) = uint32(1);
        h_mesh(1,1).Pmtl_index = value;
    end
    
    function set.obj_name(h_mesh,value)
        if ~iscell( value )
            error('QuaDRiGa:qd_mesh','??? "obj_name" must be a cell array.')
        end
        value = reshape( value, 1, [] );
        no_value = numel(value);
        if no_value ~= numel( h_mesh(1,1).Pobj_name )
            h_mesh(1,1).no_obj = no_value;
        end
        ind = cellfun(@isempty,value);
        value( ind ) = h_mesh(1,1).Pobj_name(ind);
        h_mesh(1,1).Pobj_name = value;
    end
    
    function set.obj_att_par(h_mesh,value)
        if size( value,1 ) ~= 5
            error('QuaDRiGa:qd_mesh','??? "obj_att_par" must have 5 rows.')
        end
        no_value = size(value,2);
        if no_value ~= size( h_mesh(1,1).Pobj_att_par,2 )
            h_mesh(1,1).no_obj = no_value;
        end
        if ~isa(value,'double')
           value = double( value ); 
        end
        h_mesh(1,1).Pobj_att_par = value;
    end
    
    function set.mtl_name(h_mesh,value)
        if ~iscell( value )
            error('QuaDRiGa:qd_mesh','??? "mtl_name" must be a cell array.')
        end
        value = reshape( value, 1, [] );
        no_value = numel(value);
        if no_value ~= numel( h_mesh(1,1).Pmtl_name )
            h_mesh(1,1).no_mtl = no_value;
        end
        ind = cellfun(@isempty,value);
        value( ind ) = h_mesh(1,1).Pmtl_name(ind);
        h_mesh(1,1).Pmtl_name = value;
    end
    
    function set.mtl_color(h_mesh,value)
        if size( value,1 ) ~= 3
            error('QuaDRiGa:qd_mesh','??? "mtl_color" must have 3 rows.')
        end
        no_value = size(value,2);
        if no_value ~= size( h_mesh(1,1).Pmtl_color,2 )
            h_mesh(1,1).no_mtl = no_value;
        end
        if ~isa(value,'double')
           value = double( value ); 
        end
        h_mesh(1,1).Pmtl_color = value;
    end
    
    function set.mtl_prop(h_mesh,value)
        if size( value,1 ) ~= 4
            error('QuaDRiGa:qd_mesh','??? "mtl_prop" must have 4 rows.')
        end
        no_value = size(value,2);
        if no_value ~= size( h_mesh(1,1).Pmtl_prop,2 )
            h_mesh(1,1).no_mtl = no_value;
        end
        if ~isa(value,'double')
           value = double( value ); 
        end
        h_mesh(1,1).Pmtl_prop = value;
    end
    
    function set.mtl_thickness(h_mesh,value)
        value = reshape( value, 1, [] );
        no_value = numel(value);
        if no_value ~= numel( h_mesh(1,1).Pmtl_thickness )
            h_mesh(1,1).no_mtl = no_value;
        end
        if ~isa(value,'double')
           value = double( value ); 
        end
        h_mesh(1,1).Pmtl_thickness = value;
    end
    
end

methods(Static)
    gpu = has_gpu;
end

end

