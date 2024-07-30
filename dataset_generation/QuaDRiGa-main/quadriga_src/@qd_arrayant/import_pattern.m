function h_array = import_pattern( fVi, fHi , azimuth_grid , elevation_grid )
%IMPORT_PATTERN Converts antenna field patterns into a QuaDRiGa array object
%
% Calling object:
%   None (static method)
%
% Description:
%   This function converts any antenna field pattern into a QuaDRiGa antenna array object. The
%   angle grid is adjusted automatically to the range supported by QuaDRiGa.
%
% Input:
%   fVi
%   The field pattern(s) for the vertical polarization given in spherical coordinates. The first
%   dimension corresponds to the elevation angle. The second dimension is for the azimuth angle.
%   The third dimension belongs to the element number. The default resolution is 1 degree. Hence,
%   the default size of fVi is <181x361x1>. If a different resolution is given or a different
%   angular sampling is given, the optional variables "azimuth_grid" and "elevation_grid" must be
%   defined.
%
%   fHi
%   The field pattern(s) for the horizontal polarization given in spherical coordinates. "fHi" can
%   be empty if no horizontal response is given. If it is given, then "fHi" must have the same size
%   as "fVi".
%
%   azimuth_grid
%   A vector specifying the azimuth sampling points of the patterns in units of radians. This value
%   only needs to be defined if the patterns do not have the default size or angular sapling grid.
%   Values outside the [-pi,pi] range will be converted and the patterns will adjusted accordingly.
%
%   elevation_grid
%   A vector specifying the elevation sampling points of the patterns in units of radians
%   (typically raging from -pi/2 to pi/2). This value only needs to be defined if the patterns do
%   not have the default size.
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

% Parse input: fVi
if ~exist('fVi','var') || isempty(fVi)
    error('QuaDRiGa:qd_arrayant:import_pattern',...
        'Vertical field pattern "fVi" is not defined.')
end

% Parse input: fHi
if exist('fHi','var') && ~isempty(fHi)
    if any( size(fVi) ~= size(fHi) )
        error('QuaDRiGa:qd_arrayant:import_pattern',...
            'Horizontal field pattern "fHi" has different size than vertical pattern "fVi".')
    end
else
    fHi = zeros( size(fVi) );
end

% Parse input: azimuth_grid
if exist('azimuth_grid','var') && ~isempty(azimuth_grid)
    if ~( any( size(azimuth_grid) == 1 ) && isnumeric(azimuth_grid) && isreal(azimuth_grid) &&...
            numel(azimuth_grid) == size( fVi,2 ) )
        error('QuaDRiGa:qd_arrayant:import_pattern','??? "azimuth_grid" must match the size of FVi')
    end
elseif size(fVi,2) ~= 361
    error('QuaDRiGa:qd_arrayant:import_pattern','??? "azimuth_grid" undefined');
else
    azimuth_grid = (-180:180)*pi/180;
end

% Parse input: elevation_grid
if exist('elevation_grid','var') && ~isempty(elevation_grid)
    if ~( any( size(elevation_grid) == 1 ) && isnumeric(elevation_grid) && isreal(elevation_grid) &&...
            numel(elevation_grid) == size( fVi,1 ) )
        error('QuaDRiGa:qd_arrayant:import_pattern','??? "elevation_grid" must match the size of FVi')
    end
elseif size(fVi,1) ~= 181
    error('QuaDRiGa:qd_arrayant:import_pattern','??? "elevation_grid" undefined');
else
    elevation_grid = (-90:90)*pi/180;
end

no_input_elements = size(fVi,3);

azimuth_grid = angle( exp(1j*azimuth_grid) );       % -pi to pi
elevation_grid = angle( exp(1j*elevation_grid) );   % -pi/2 to pi/2

[azimuth_grid,ia] = sort( azimuth_grid );
[elevation_grid,ie] = sort( elevation_grid );

h_array = qd_arrayant;
h_array.set_grid( azimuth_grid , elevation_grid, 0 );
h_array.no_elements = no_input_elements;
h_array.Fa = fVi( ie,ia,: );
h_array.Fb = fHi( ie,ia,: );

if ~h_array.is_wrapped
    h_array.wrap_grid;
end

end
