classdef qd_arrayant < handle
%QD_ARRAYANT QuaDRiGa array antenna class
%
% DESCRIPTION
% This class combines all functions to create and edit array antennas. An array
% antenna is a set of single antenna elements, each having a specific beam pattern,
% that can be combined in any geometric arrangement. A set of synthetic arrays
% that allow simulations without providing your own antenna patterns is
% provided (see generate method for more details).
%
% REFERENCE
% The main functionality was taken from the Winner channel model. "Kyösti, P.;
% Meinilä, J.; Hentilä, L. & others; {IST-4-027756 WINNER II D1.1.2 v.1.1}:
% WINNER II Channel Models; 2007". New functionality has been added to provide
% geometric polarization calculations and antenna pattern rotations.
%
% EXAMPLE
% This example creates an array antenna of crossed dipoles.
%
%    a = qd_arrayant('dipole');           % Create new qd_arrayant object
%    a.copy_element(1,2);                 % Duplicate the dipole
%    a.rotate_pattern(90,'y',2);          % Rotate the second element by 90Â°
%    a.visualize;                         % Show the output
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
    
properties
    name = 'New array';                     % Name of the array antenna
    
    % Center frequency in [Hz]
    center_frequency = qd_simulation_parameters.speed_of_light;
end

properties(SetAccess=protected)
    % Elevation angles in [rad] were samples of the field patterns are provided
    %   The field patterns are given in spherical coordinates. This
    %   variable provides the elevation sampling angles in radians
    %   ranging from -pi/2 (downwards) to pi/2 (upwards).
    elevation_grid = [ -1.570796326794897,0,1.570796326794897];
    
    % Azimuth angles in [rad] were samples of the field patterns are provided
    %   The field patterns are given in spherical coordinates. This
    %   variable provides the azimuth sampling angles in radians
    %   ranging from -pi to pi.
    azimuth_grid = [ -3.141592653589793,-1.570796326794897,0,...
        1.570796326794897,3.141592653589793];
end

properties(Dependent)
    % Number of antenna elements in the array antenna
    %    Increasing the number of elements creates new elements which
    %    are initialized as copies of the first element. Decreasing the
    %    number of elements deletes the last elements from the array antenna.
    no_elements
    
    % Position of the antenna elements in local Cartesian coordinates (units of [m])
    element_position
    
    % The first component of the antenna pattern
    %   contains the vertical component of the electric field given in spherical
    %   coordinates (aligned with the phi direction of the coordinate system). This
    %   variable is a tensor with dimensions [ elevation, azimuth, element-index ]
    %   describing the e-phi component of the far field of each antenna element in the
    %   array.
    Fa
    
    % The second component of the antenna pattern
    %   contains the horizontal component of the electric field given in spherical
    %   coordinates (aligned with the theta direction of the coordinate system). This
    %   variable is a tensor with dimensions [ elevation, azimuth, element-index ]
    %   describing the e-theta component of the far field of each antenna element in
    %   the array.
    Fb
    
    % Coupling matrix between elements
    %   This matrix describes a pre- or post-processing of the signals
    %   that are fed to the antenna elements. For example, in order to
    %   transmit a LHCP signal, two antenna elements are needed.
    %   They are then coupled by a matrix
    %
    %   	1/sqrt(2) * [1;j]
    %
    %   The rows in the matrix correspond to the antenna elements, the
    %   columns to the signal ports. In this example, the antenna has
    %   one port, i.e. it is fed with one input signal. This signal is
    %   then split into two and fed to the two antenna elements where
    %   the second element radiates the signal with 90 degree phase
    %   shift.
    %   In a similar fashion, it is possible to create fixed
    %   beamforming antennas and include crosstalk between antenna
    %   elements. By default, coupling is set to an identity matrix
    %   which indicates perfect isolation between the antenna elements.
    coupling
    
    no_az                              % Number of azimuth values
    no_el                              % Number of elevation values
    is_wrapped                         % Indicator if the array is wrapped
end

properties(Hidden)
    Pno_elements        = 1;
    Pelement_position   = [0;0;0];
    PFa                 = ones(3,5);
    PFb                 = zeros(3,5);
    Pcoupling           = 1;
    OctEq               = false; % For qf.eq_octave
    Pphase_diff         = [];
end

methods
    
    function h_qd_arrayant = qd_arrayant( array_type, varargin )
        %QD_ARRAYANT Creates a new array object.
        %
        % Calling object:
        %   None (constructor)
        %
        % Description:
        %   The constructor calls qd_arrayant.generate to create new array antennas.
        %   If no input is specified, an vertically polarized omni-antenna is
        %   generated. See qd_arrayant.generate for a description of the input
        %   parameters and the list of supported antenna types.
        %
        %
        % QuaDRiGa Copyright (C) 2011-2021 Fraunhofer Heinrich Hertz Institute
        % e-mail: quadriga@hhi.fraunhofer.de
        %
        % QuaDRiGa is free software: you can redistribute it and/or modify
        % it under the terms of the GNU Lesser General Public License as published
        % by the Free Software Foundation, either version 3 of the License, or
        % (at your option) any later version.
        if nargin > 0
            if ~isempty( array_type )
                h_qd_arrayant = qd_arrayant.generate( array_type , varargin{:} );
            end
        else
            h_qd_arrayant = qd_arrayant.generate( 'omni' );
        end
    end
    
    % Get functions
    function out = get.no_elements(h_qd_arrayant)
        out = h_qd_arrayant.Pno_elements;
    end
    function out = get.element_position(h_qd_arrayant)
        out = h_qd_arrayant.Pelement_position; % Single
    end
    function out = get.Fa(h_qd_arrayant)
        out = h_qd_arrayant.PFa; % Single
    end
    function out = get.Fb(h_qd_arrayant)
        out = h_qd_arrayant.PFb; % Single
    end
    function out = get.coupling(h_qd_arrayant)
        out = h_qd_arrayant.Pcoupling; % Single
    end
    function out = get.no_az(h_qd_arrayant)
        out = numel(h_qd_arrayant.azimuth_grid);
    end
    function out = get.no_el(h_qd_arrayant)
        out = numel(h_qd_arrayant.elevation_grid);
    end
    function out = get.is_wrapped(h_qd_arrayant)
        if h_qd_arrayant.azimuth_grid(1) < -pi + 1e-7 && ...            % First azimuth value must be <= -pi
                h_qd_arrayant.azimuth_grid(end) > pi - 1e-7 && ...      % Last azimuth value must be >= -pi
                h_qd_arrayant.elevation_grid(1) < -pi/2 + 1e-7 && ...   % First elevation value must be -pi/2
                h_qd_arrayant.elevation_grid(end) > pi/2 - 1e-7         % Last elevation value must be pi/2
            out = true;
        else
            out = false;
        end
    end
    
    % Set functions
    function set.name(h_qd_arrayant,value)
        if ~( ischar(value) )
            error('QuaDRiGa:wrongNumberOfInputs:wrongInputValue','??? "name" must be a string.')
        end
        h_qd_arrayant.name = value;
    end
    
    function set.no_elements(h_qd_arrayant,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                && isreal(value) && mod(value,1)==0 && value > 0 )
            error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "no_elements" must be integer and > 0')
        end
        
        if h_qd_arrayant.no_elements > value
            h_qd_arrayant.Pelement_position = h_qd_arrayant.Pelement_position(:,1:value);
            h_qd_arrayant.PFa = h_qd_arrayant.PFa(:,:,1:value);
            h_qd_arrayant.PFb = h_qd_arrayant.PFb(:,:,1:value);
            
            ne = h_qd_arrayant.no_elements-value;
            nc = size(h_qd_arrayant.Pcoupling);
            h_qd_arrayant.Pcoupling = h_qd_arrayant.Pcoupling( 1:nc(1)-ne , 1:max(nc(2)-ne,1) );
            
        elseif h_qd_arrayant.no_elements < value
            ne = value-h_qd_arrayant.no_elements;
            
            h_qd_arrayant.Pelement_position = [ h_qd_arrayant.Pelement_position,...
                h_qd_arrayant.Pelement_position(:,ones( 1,ne )) ];
            
            h_qd_arrayant.PFa = cat( 3, h_qd_arrayant.PFa ,...
                h_qd_arrayant.PFa(:,:,ones( 1,ne )));
            
            h_qd_arrayant.PFb = cat( 3, h_qd_arrayant.PFb ,...
                h_qd_arrayant.PFb(:,:,ones( 1,ne )));
            
            nc = size(h_qd_arrayant.Pcoupling);
            
            if isa( h_qd_arrayant.Pcoupling,'single' )
                C = zeros( nc(1)+ne , nc(2)+ne, 'single');
            else
                C = zeros( nc(1)+ne , nc(2)+ne );
            end
            for n = 1:ne
                C( nc(1)+n,nc(2)+n ) = 1;
            end
            C( 1:nc(1) , 1:nc(2) ) = h_qd_arrayant.Pcoupling;
            h_qd_arrayant.Pcoupling = C;
        end
        
        h_qd_arrayant.Pno_elements = value;
        h_qd_arrayant.Pphase_diff = [];
    end
    
    function set.element_position(h_qd_arrayant,value)
        if ~( isnumeric(value) && isreal(value) )
            error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "element_position" must consist of real numbers')
        elseif ~all( size(value,1) == 3 )
            error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "element_position" must have 3 rows')
        end
        if size(value,2) ~= h_qd_arrayant.no_elements
            h_qd_arrayant.no_elements = size(value,2);
        end
        h_qd_arrayant.Pelement_position = value;
    end
    
    function set.Fa(h_qd_arrayant,value)
        a = numel( h_qd_arrayant.elevation_grid );
        b = numel( h_qd_arrayant.azimuth_grid );
        
        if h_qd_arrayant.Pno_elements == 1
            dims = [ a , b ];
        else
            dims = [ a , b , h_qd_arrayant.Pno_elements];
        end
        
        if ~( isnumeric(value) )
            error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "Fa" must be numeric.')
        elseif ~( numel(size(value)) == numel(dims) && all( size(value) == dims ) )
            error('QuaDRiGa:qd_arrayant:wrongInputValue',['??? "Fa" must be of size [',num2str(a),'x',num2str(b),...
                'x',num2str(h_qd_arrayant.Pno_elements),'].'])
        end
        h_qd_arrayant.PFa = value;
        h_qd_arrayant.Pphase_diff = [];
    end
    
    function set.Fb(h_qd_arrayant,value)
        a = numel( h_qd_arrayant.elevation_grid );
        b = numel( h_qd_arrayant.azimuth_grid );
        
        if h_qd_arrayant.no_elements == 1
            dims = [ a , b ];
        else
            dims = [ a , b , h_qd_arrayant.Pno_elements];
        end
        
        if ~( isnumeric(value) )
            error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "Fb" must be numeric.')
        elseif ~( numel( size(value) ) == numel( dims ) && all( size(value) == dims ) )
            error('QuaDRiGa:qd_arrayant:wrongInputValue',['??? "Fb" must be of size [',num2str(a),'x',num2str(b),...
                'x',num2str(h_qd_arrayant.Pno_elements),'].'])
        end
        h_qd_arrayant.PFb = value;
        h_qd_arrayant.Pphase_diff = [];
    end
    
    function set.coupling(h_qd_arrayant,value)
        if ~( isnumeric(value) && size(value,1) == h_qd_arrayant.Pno_elements && ...
                size(value,2) >= 1 )
            error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "coupling" must be a matrix with rows equal to elements and columns equal to ports')
        end
        h_qd_arrayant.Pcoupling = value;
    end
end

methods(Static)
    [ h_qd_arrayant, par ] = generate( array_type, Ain, Bin, Cin, Din, Ein, Fin, Gin, Hin, Iin, Jin );
    h_array = import_pattern( fVi, fHi , azimuth_grid , elevation_grid )
    [ h_array, l ] = xml_read( fn, fid, pfx, l, ignore_layout );
end
end
