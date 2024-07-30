classdef qd_satellite < handle
%QD_SATELLITE Summary of this class goes here
%   
% DESCRIPTION
%    This class contains all functions to generate and load satellite constellations. Based on the
%    defined constellation tracks can be generated to be used with the QuaDRiGa channel model.
%
% REFERENCE
%    The satellite orbit description is based on the ITU Recommendation ITU-R S.1503-3 (01/2018),
%    "Functional description to be used in developing software tools for determining conformity of
%    non-geostationary-satellite orbit fixed-satellite service systems or networks with limits
%    contained in Article 22 of the Radio Regulations".
%
% Constructor:
%
%   h_qd_satellite = qd_satellite ( constellation, Ain, Bin, Cin, Din, Ein, Fin )
%   
%   The constructor creates new satellite constellations with a number of S satellites. The input
%   parameter 'constellation' defines the constellation type. Additional parameters are specific to
%   the constellation. They are defined by the following list. If no input is specified, an empty
%   object is created.
%
% Constellation:
%
%   custom
%   A custom constellation. All input parameters must be given as vectors having the dimensions 
%   [ 1 x S ].
%
%    * Ain - Semimajor axis in [km]; Default: 42164 km, GEO orbit
%    * Bin - Orbital eccentricity [0-1]; Default: 0
%    * Cin - Orbital inclination in [degree]; Default: 0
%    * Din - Longitude of the ascending node in [degree]; Default: 0
%    * Ein - Argument of periapsis in [degree]; Default: 0
%    * Fin - True anomaly in [degree]; Default: 0
%
%   gso
%   A constellation of equally spaced geostationary satellites.
%
%    * Ain - Number of satellites S; Scalar variable; Default: 3
%    * Bin - Phase offset of the first satellite in [degree]; Scalar variable
%
%   walker-delta
%   Walker-Delta pattern constellation is used for a global coverage of the Earth's surface by a
%   minimum number of satellites in circular orbits. A Walker-Delta pattern contains of total of
%   'S' satellites in 'p' orbital planes with 't=S/p' satellites in each orbital plane. All orbital
%   planes are assumed to be in same inclination with reference to the equator. The phase
%   difference between satellites in adjacent plane is defined as the angle in the direction of
%   motion from the ascending node to the nearest satellite at a time when a satellite in the next
%   most westerly plane is at its ascending node. In order for all of the orbit planes to have the
%   same phase difference with each other, the phase difference between adjacent satellites must be
%   a multiple of 'f*360/S', where 'f' can be an integer between 0 to p-1.
%
%    * Ain - Semimajor axis for all satellites in [km]; Scalar variable
%    * Bin - Orbital inclination in [degree]; Scalar variable
%    * Cin - Number of orbital planes 'p'; Scalar variable
%    * Din - Number of satellites per plane 't'; Scalar variable
%    * Ein - Phase difference in [degree]; Scalar variable
%
%
% Output:
%   h_qd_satellite
%   Handle to the created 'qd_satellite' object.
%
%
% QuaDRiGa Copyright (C) 2011-2020
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
    name = 'new constellation';     % Name of the constellation
    
    % Enable or disable station keeping
    % There are minor launch errors and perturbations that would make the orbit drift unless station
    % keeping was used to ensure the track ground repeats. An important aspect to station keeping is
    % to simulate multiple passes of the non-GSO satellite through an earth station's main beam with
    % slightly different crossing directions. As changing the position within the plane does not
    % affect this, then the main parameter to vary is the longitude of the ascending node. Setting
    % this property to 1 assumes perfect station keeping and deactivates orbit drift. Default: 0 (no
    % station keeping).
    station_keeping = 0;     
    
    epoch = [];                     % Common datetime of epoch for all satellites in days since Jan 1,0000
    
    sat_name = {};                  % Satellite name
end

properties(Constant, Hidden)
    mu = 3.986012e5;                % Gravitational parameter for Earth in [km^3/s^2] (G*M_e)
    R_geo = 42164.1964;             % Radius of geostationary orbit in [km]
    R_e = 6378.137;                 % Radius of the Earth in [km] @ Equator
    c = 2.99792458e5;               % Speed of light in [km/s]
    Omega_e = 4.17807462185047e-3;  % Angular rate of rotation of the Earth in [degrees/s]
    T_e = 86164.09054;              % Earth rotation period in [s]
    J_2 = 0.001082636;              % Factor of the Earth non-sphericity in [1]
    G = 6.67408e-11;                % Gravitational constant in [m^3/s^2/kg]
    M_e = 5.9722e24;                % Mass of Earth in [kg]
    Omega_eR = 100.13;                   % Earth reference orientation in [deg] at the reference time
    Epoch_eR = 737791;              % Reference time (2020-01-01 in days since Jan 1,0000)
end

properties(Dependent)
    n_satellites                    % Number of satellites (read only)
    
    % Semimajor axis (a) in [km]
    % is equal to the sum of the periapsis and apoapsis distances divided by two. For circular
    % orbits, the semimajor axis is the distance between the centers of the bodies, not the distance
    % of the bodies from the center of mass.  
    semimajor_axis
    
    % The orbital eccentricity (e) of a satellite
    % is the parameter that determines the amount by which its orbit around the Earth deviates from
    % a perfect circle. eccentricity = 0 yields in a circular orbit, 0 < eccentricity < 1 yields an
    % elliptical orbit. 
    eccentricity
   
    % Orbital inclination (i) in [degree]
    % measures the tilt of a satellites orbit around the earth. It is expressed as the angle between
    % Earths equatorial plane and the orbital plane or axis of direction of the satellite.
    inclination
    
    % Longitude of the ascending node (Omega) in [degree]
    % horizontally orients the ascending node of the ellipse (where the orbit passes upward through
    % the equatorial plane) with respect to the reference frame's vernal point. As the orbit is
    % fixed in inertial space while the Earth rotates, a time reference for which this angle is
    % valid must be given. In this case it is the start of the simulation.
    lon_asc_node
    
    % Argument of periapsis (omega) in [degree]
    % defines the orientation of the ellipse in the orbital plane, as an angle measured from the
    % ascending node to the periapsis (the closest point the satellite comes to the Earth).
    arg_periapsis
    
    % True anomaly (v) in [degree]
    % defines the position of the satellite along the ellipse at a specific time (the "epoch").
    true_anomaly
end

properties(Dependent,Hidden)
    R_p                             % Pericenter distance, distance from the center of Earth to the satellite at perigee in km
    R_a                             % Apocenter distance, distance from the center of Earth to the satellite at apogee in km
    p                               % Focal parameter in [km]
    n_0                             % Mean angular motion in [degrees/second]
    n_bar                           % Line of nodes in [degrees/second]
    Omega_r                         % Ascending node longitude precession rate in [degrees/second]
    omega_r                         % Perigee argument precession rate in [degrees/second]
    E_0                             % Initial eccentric anomaly in [degrees]
    M_0                             % Initial mean anomaly in [degrees]
end

properties(Dependent,SetAccess=private)
    orbit_period                    % Orbital period in [seconds] (read only)
end

properties(Access=private)
    PR_p
    PR_a
    Pp
    Pn_0
    Pn_bar
    POmega_r
    Pomega_r
    PE_0
    PM_0
    Porbit_period
end
    
properties(Access=private)
    Pa                              % Semi-major axis in km
    Pe                              % Eccentricity (default = 0 (circular orbit))
    Pi                              % Inclination in degrees
    POmega_0                        % Longitude of ascending node in degrees
    Pomega_0                        % Argument of perigee in degrees, default = 0 (circular)
    Pv_0                            % True anomaly in degrees
end

methods
    function h_qd_satellite = qd_satellite( constellation, varargin )
        %QD_SATELLITE Creates a new 'qd_satellite' object (constructor)
        %
        %   See class help of QuaDRiGa documentation for additional information.
        %
        % QuaDRiGa Copyright (C) 2011-2019 Fraunhofer Heinrich Hertz Institute
        % e-mail: quadriga@hhi.fraunhofer.de
        %
        % QuaDRiGa is free software: you can redistribute it and/or modify
        % it under the terms of the GNU Lesser General Public License as published
        % by the Free Software Foundation, either version 3 of the License, or
        % (at your option) any later version.

        if nargin > 0 && ~isempty( constellation )
            h_qd_satellite = qd_satellite.generate( constellation, varargin{:} );
        end
    end
    
    % Get functions of public properties
    function out = get.n_satellites(h_qd_satellite)
        out = numel( h_qd_satellite.Pa );
    end
    function out = get.eccentricity(h_qd_satellite)
        out = h_qd_satellite.Pe;
    end
    function out = get.semimajor_axis(h_qd_satellite)
        out = h_qd_satellite.Pa;
    end
    function out = get.inclination(h_qd_satellite)
        out = h_qd_satellite.Pi;
    end
    function out = get.lon_asc_node(h_qd_satellite)
        out = h_qd_satellite.POmega_0;
    end
    function out = get.arg_periapsis(h_qd_satellite)
        out = h_qd_satellite.Pomega_0;
    end
    function out = get.true_anomaly(h_qd_satellite)
        out = h_qd_satellite.Pv_0;
    end
    
    % Get functions of dependent properties
    function out = get.R_p(h_qd_satellite)          % Pericenter distance in [km]
        out = h_qd_satellite.PR_p;
    end
    function out = get.R_a(h_qd_satellite)          % Apocenter distance in [km]
        out = h_qd_satellite.PR_a;
    end
    function out = get.p(h_qd_satellite)            % Focal parameter in [km]
        out = h_qd_satellite.Pp;
    end
    function out = get.n_0(h_qd_satellite)          % Mean angular motion in [degrees/second]
        out = h_qd_satellite.Pn_0;
    end
    function out = get.n_bar(h_qd_satellite)        % Line of nodes in [degrees/second]
        out = h_qd_satellite.Pn_bar;
    end
    function out = get.Omega_r(h_qd_satellite)      % Ascending node longitude precession rate in [degrees/second]
        out = h_qd_satellite.POmega_r;
    end
    function out = get.omega_r(h_qd_satellite)      % Perigee argument precession rate in [degrees/second]
        out = h_qd_satellite.Pomega_r;
    end
    function out = get.E_0(h_qd_satellite)          % Initial eccentric anomaly in [degrees]
        out = h_qd_satellite.PE_0;
    end
    function out = get.M_0(h_qd_satellite)          % Initial mean anomaly in [degrees]
       out = h_qd_satellite.PM_0;
    end
    function out = get.orbit_period(h_qd_satellite) % Revised orbital period in [seconds]
        out = h_qd_satellite.Porbit_period;
    end
        
    % Set functions
    function set.station_keeping(h_qd_satellite, value)
        if numel(value) ~= 1
            error('QuaDRiGa:qd_satellite:wrongInputValue','??? "station_keeping" must be a scalar');
        end
        h_qd_satellite.station_keeping = logical( value );
        update_dependent_variables( h_qd_satellite );
    end
    
    function set.semimajor_axis(h_qd_satellite, value)
        if ~(isnumeric(value) && isreal(value) && (size(value, 1) == 1) && all(value > 6378.145))
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? "semimajor_axis" must be a real, numeric, row vector, having values greater than 6378.145 km');
        end
        no = numel( h_qd_satellite.Pa );            % Number of existing satellites
        ii = 1:min(no,numel( value ));              % Indices of existing satellites
        init = zeros( 1,numel(value) );             % Initial values for all other variables
        h_qd_satellite.Pa = value;                  % Set new value
        if no > 0; init( 1,ii ) = h_qd_satellite.Pe( 1,ii ); end
        h_qd_satellite.Pe = init;                   % Update dependent property "Pe"
        if no > 0; init( 1,ii ) = h_qd_satellite.Pi( 1,ii ); end
        h_qd_satellite.Pi = init;                   % Update dependent property "Pi"
        if no > 0; init( 1,ii ) = h_qd_satellite.POmega_0( 1,ii ); end
        h_qd_satellite.POmega_0 = init;             % Update dependent property "POmega_0"
        if no > 0; init( 1,ii ) = h_qd_satellite.Pomega_0( 1,ii ); end
        h_qd_satellite.Pomega_0 = init;             % Update dependent property "Pomega_0"
        if no > 0; init( 1,ii ) = h_qd_satellite.Pv_0( 1,ii ); end
        h_qd_satellite.Pv_0 = init;                 % Update dependent property "Pv_0"
        update_dependent_variables( h_qd_satellite );
    end
    
    function set.eccentricity(h_qd_satellite, value)
        if ~(isnumeric(value) && isreal(value) && (size(value, 1) == 1) && all(0 <= value & value < 1))
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? "eccentricity" must be a real, numeric, row vector, having values in [0, 1)')
        end
        if numel( value ) ~= numel( h_qd_satellite.Pa )
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? number of elements must match the number of satellites. Set "semimajor_axis" first.')
        end
        h_qd_satellite.Pe = value;                  % Set new value
        update_dependent_variables( h_qd_satellite );
    end
    
    function set.inclination(h_qd_satellite, value)
        if ~(isnumeric(value) && isreal(value) && (size(value, 1) == 1) && all(-90 <= value & value <= 180))
                error('QuaDRiGa:qd_constellation:wrongInputValue',...
                    '??? "inclination" must be a real, numeric, row vector, having values in [-90, 180]')
        end
        if numel( value ) ~= numel( h_qd_satellite.Pa )
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? number of elements must match the number of satellites. Set "semimajor_axis" first.')
        end
        h_qd_satellite.Pi = value;                  % Set new value
        update_dependent_variables( h_qd_satellite );
    end
    
    function set.lon_asc_node(h_qd_satellite, value)
        if ~(isnumeric(value) && isreal(value) && (size(value, 1) == 1))
            error('QuaDRiGa:qd_constellation:wrongInputValue',...
                '??? "lon_asc_node" must be a real, numeric, row vector, having values in [0, 360)')
        end
        if numel( value ) ~= numel( h_qd_satellite.Pa )
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? number of elements must match the number of satellites. Set "semimajor_axis" first.')
        end
        value = mod( value +180,360)-180;           % Map to (-180,180]
        value( value == -180 ) = 180;
        h_qd_satellite.POmega_0 = value;        	% Set new value
        update_dependent_variables( h_qd_satellite );
    end
    
    function set.arg_periapsis(h_qd_satellite, value)
        if ~(isnumeric(value) && isreal(value) && (size(value, 1) == 1))
            error('QuaDRiGa:qd_constellation:wrongInputValue',...
                '??? "arg_periapsis" must be a real, numeric, row vector, having values in [0, 360)')
        end
        if numel( value ) ~= numel( h_qd_satellite.Pa )
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? number of elements must match the number of satellites. Set "semimajor_axis" first.')
        end
        value = mod( value +180,360)-180;           % Map to (-180,180]
        value( value == -180 ) = 180;
        h_qd_satellite.Pomega_0 = value;        	% Set new value
        update_dependent_variables( h_qd_satellite );
    end
    
    function set.true_anomaly(h_qd_satellite, value)
        if ~(isnumeric(value) && isreal(value) && (size(value, 1) == 1))
            error('QuaDRiGa:qd_constellation:wrongInputValue',...
                '??? "true_anomaly" must be a real, numeric, row vector')
        end
        if numel( value ) ~= numel( h_qd_satellite.Pa )
            error('QuaDRiGa:qd_satellite:wrongInputValue',...
                '??? number of elements must match the number of satellites. Set "semimajor_axis" first.')
        end
        value = mod( value +180,360)-180;           % Map to (-180,180]
        value( value == -180 ) = 180;
        h_qd_satellite.Pv_0 = value;                % Set new value
        update_dependent_variables( h_qd_satellite );
    end
    
    function update_dependent_variables( h_qd_satellite )
        % Pericenter distance in [km]
        h_qd_satellite.PR_p = (1-h_qd_satellite.Pe).*h_qd_satellite.Pa;
        
        % Apocenter distance in [km]
        h_qd_satellite.PR_a = (1+h_qd_satellite.Pe).*h_qd_satellite.Pa;
        
        % Focal parameter in [km]
        h_qd_satellite.Pp = h_qd_satellite.Pa.*(1 - h_qd_satellite.Pe.^2);
        
        % Mean angular motion in [degrees/second]
        h_qd_satellite.Pn_0 = sqrt(h_qd_satellite.mu./h_qd_satellite.Pa.^3)/pi*180;
        
        % Line of nodes in [degrees/second]
        if h_qd_satellite.station_keeping
            h_qd_satellite.Pn_bar  = h_qd_satellite.Pn_0;
        else
            h_qd_satellite.Pn_bar  = h_qd_satellite.Pn_0 .* (1 + 1.5*h_qd_satellite.J_2 .* ...
                h_qd_satellite.R_e.^2 ./ h_qd_satellite.Pp.^2.* ...
                (1 - 1.5*sind(h_qd_satellite.Pi).^2).*sqrt(1-h_qd_satellite.Pe.^2));
        end
        
        % Ascending node longitude precession rate in [degrees/second]
        if h_qd_satellite.station_keeping
            h_qd_satellite.POmega_r = zeros( size( h_qd_satellite.Pa ) );
        else
            h_qd_satellite.POmega_r = -1.5 * h_qd_satellite.J_2 .* h_qd_satellite.R_e.^2 ./ ...
                h_qd_satellite.Pp.^2 .* h_qd_satellite.Pn_bar .* cosd(h_qd_satellite.Pi);
        end
            
        % Perigee argument precession rate in [degrees/second]
        if h_qd_satellite.station_keeping
            h_qd_satellite.Pomega_r = zeros( size( h_qd_satellite.Pa ) );
        else
            h_qd_satellite.Pomega_r = 1.5*h_qd_satellite.J_2 .* h_qd_satellite.R_e.^2 ./ ...
                h_qd_satellite.Pp.^2.* h_qd_satellite.Pn_bar.*(2-2.5.*sind(h_qd_satellite.Pi).^2);
        end
            
        % Initial eccentric anomaly in [degrees]
        Pex = h_qd_satellite.Pe;
        h_qd_satellite.PE_0 = 2*atand(tand(h_qd_satellite.Pv_0./2).*sqrt((1-Pex)./(1+Pex)));
        
        % Initial mean anomaly in [degrees]
        h_qd_satellite.PM_0 = h_qd_satellite.PE_0 - h_qd_satellite.Pe.*sind(h_qd_satellite.PE_0);
        
        % Revised orbital period in [seconds]
        h_qd_satellite.Porbit_period = 360./(h_qd_satellite.Pomega_r + h_qd_satellite.Pn_bar);
    end
end

methods(Static)
    h_qd_satellite = generate( constellation, Ain, Bin, Cin, Din, Ein, Fin );
    h_qd_satellite = read_tle( filename );
end
end
