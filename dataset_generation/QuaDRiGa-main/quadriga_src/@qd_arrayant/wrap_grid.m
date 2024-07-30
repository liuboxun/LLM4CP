function [ Fa, Fb, azimuth_grid, elevation_grid, element_position ] = wrap_grid( h_qd_arrayant, ...
    i_element, precision )
%WRAP_GRID Wraps the antenna patterns around the unit sphere
%
% Calling object:
%   Single object
%
% Description:
%   This function reads the antenna patterns from the qd_arrayant object and checks if the pattern
%   is wrapped. This is defined as:
%
%      * The first angle in the azimuth grid is smaller or equal to -pi
%      * The last angle in the azimuth grid is larger or equal to pi
%      * The first elevation angle is -pi/2
%      * The last elevation angle is pi/2
%
%    These conditions are required for the antenna pattern interpolation which can only
%    interpolate, but not extrapolate. The output of this function are the completed patterns.
%
% Input:
%   i_element
%   The element indices that should be returned.
%
%   precision
%   If set to 'single', single precision variables are returned.
%
% Output:
%   Fa
%   The first component of the antenna pattern
%
%   Fb
%   The second component of the antenna pattern
%
%   azimuth_grid
%   Azimuth angles (theta) in [rad]
%
%   elevation_grid
%   Elevation angles (phi) in [rad]
%
%   element_position
%   Position of the antenna elements
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

if numel( h_qd_arrayant ) > 1
    error('QuaDRiGa:qd_arrayant:wrap_grid','interpolate not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

if exist( 'i_element','var' ) && ~isempty( i_element )
    if ~(any(size(i_element) == 1) && isnumeric(i_element) ...
            && isreal(i_element) && all(mod(i_element, 1) == 0) && all(i_element > 0))
        error('QuaDRiGa:qd_arrayant:wrap_grid','??? "i_element" must be integer and > 0')
    elseif any(i_element > h_qd_arrayant.no_elements)
        error('QuaDRiGa:qd_arrayant:wrap_grid','??? "i_element" exceeds "no_elements"')
    end
else
    i_element = 1:h_qd_arrayant.no_elements;
end

if exist( 'precision','var' ) && ~isempty( precision )
    if strcmp( precision, 'single' )
        use_single_precision = true;
    end
else
    use_single_precision = false;
end

% Copy input values
if use_single_precision
    azimuth_grid = single( h_qd_arrayant.azimuth_grid );
    elevation_grid = single( h_qd_arrayant.elevation_grid );
    Fa = single( h_qd_arrayant.PFa(:,:,i_element) );
    Fb = single( h_qd_arrayant.PFb(:,:,i_element) );
    element_position = single( h_qd_arrayant.element_position(:,i_element) );
else
    azimuth_grid = h_qd_arrayant.azimuth_grid;
    elevation_grid = h_qd_arrayant.elevation_grid;
    Fa = h_qd_arrayant.PFa(:,:,i_element);
    Fb = h_qd_arrayant.PFb(:,:,i_element);
    element_position = h_qd_arrayant.element_position(:,i_element);
end

if azimuth_grid(1) > -pi + 1e-7                 % First azimuth value must be <= -pi
    % Add a copy of the last value at the beginning
    azimuth_grid = [ azimuth_grid( end ) - 2*pi, azimuth_grid ];
    Fa = cat( 2, Fa(:,end,:), Fa );
    Fb = cat( 2, Fb(:,end,:), Fb );
    if azimuth_grid(end) < pi - 1e-7            % Last azimuth value must be >= -pi
        % Add a copy of the second value at the end
        azimuth_grid = [ azimuth_grid, azimuth_grid(2) + 2*pi ];
        Fa = cat( 2, Fa, Fa(:,2,:) );
        Fb = cat( 2, Fb, Fb(:,2,:) );
    end
elseif azimuth_grid(end) < pi - 1e-7            % Last azimuth value must be >= -pi
    % Add a copy of the first value at the end
    azimuth_grid = [ azimuth_grid, azimuth_grid(1) + 2*pi ];
    Fa = cat( 2, Fa, Fa(:,1,:) );
    Fb = cat( 2, Fb, Fb(:,1,:) );
end

% First elevation value must be -pi/2
if elevation_grid(1) > -pi/2 + 1e-7
    elevation_grid = [ -pi/2, elevation_grid ];
    Fa = cat( 1, Fa(1,:,:), Fa );
    Fb = cat( 1, Fb(1,:,:), Fb );
end

% Last elevation value must be pi/2
if elevation_grid(end) < pi/2 - 1e-7
    elevation_grid = [ elevation_grid, pi/2 ];
    Fa = cat( 1, Fa, Fa(end,:,:) );
    Fb = cat( 1, Fb, Fb(end,:,:) );
end

% Update the antenna object if there are no output arguments
if nargout == 0
    if azimuth_grid(1) < -pi
        azimuth_grid(1) = -pi;
    end
    if azimuth_grid(end) > pi
        azimuth_grid(end) = pi;
    end
    set_grid( h_qd_arrayant , azimuth_grid , elevation_grid, 0 );
    h_qd_arrayant.PFa = Fa;
    h_qd_arrayant.PFb = Fb;
    h_qd_arrayant.Pphase_diff = [];
end

end
