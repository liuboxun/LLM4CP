function append_array( h_arrayant, a )
%APPEND_ARRAY Appends an array antenna to the existing one 
%
% Calling object:
%   Single object
%
% Description:
%   This method appends the array antenna given in "a" to the existing array object. The antenna
%   patterns from "a" are copied to the calling object. For example, if the calling object has 3
%   elements and "a" has 2 elements, the two elements are added to the calling object which now has
%   5 elements. Element positions and coupling factors are copied as well.
%
% Input:
%   a
%   The array object which is appended to the current array object (scalar object).
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

if ~isa( a, 'qd_arrayant' )
    error('QuaDRiGa:qd_arrayant:append_array','"a" must be an qd_arrayant object.');
end

if numel( h_arrayant ) > 1 || numel( a ) > 1
   error('QuaDRiGa:qd_arrayant:append_array','append array not definded for object arrays.');
else
    h_arrayant = h_arrayant(1,1); % workaround for octave
    a = a(1,1);
end

% Match the sampling grids
if numel( a.elevation_grid ) ~= numel( h_arrayant.elevation_grid ) || ...
        numel( a.azimuth_grid ) ~= numel( h_arrayant.azimuth_grid ) || ...
        any( abs( h_arrayant.elevation_grid - a.elevation_grid ) > 1e-6 ) || ...
        any( abs( h_arrayant.azimuth_grid - a.azimuth_grid ) > 1e-6 )

    warning('QuaDRiGa:qd_arrayant:append_array',...
        ['The qd_arrayant sampling grids do not match, therefore the sampling grid of ''a'' will',...
        ' be changed and the pattern will be interpolated accordingly.']);
    
    b = copy( a );
    set_grid( b , h_arrayant.azimuth_grid , h_arrayant.elevation_grid );
else
    b = a;
end

cpl_in  = h_arrayant.coupling;
cpl_app = a.coupling;

% Set the number of elements in the new array
i_el_in = h_arrayant.no_elements;
h_arrayant.no_elements = i_el_in + b.no_elements;

data = h_arrayant.Fa;
data( :,:,i_el_in + 1 : end ) = b.Fa;
h_arrayant.Fa = data;

data = h_arrayant.Fb;
data( :,:,i_el_in + 1 : end ) = b.Fb;
h_arrayant.Fb = data;

data = h_arrayant.element_position;
data( :,i_el_in + 1 : end ) = b.element_position;
h_arrayant.element_position = data;

% Update the coupling matrix
nc = size(cpl_in);
ne = size(cpl_app);
cpl = zeros( nc(1)+ne(1) , nc(2)+ne(2));
cpl( 1:nc(1) , 1:nc(2) ) = cpl_in;
cpl( nc(1)+1:end , nc(2)+1:end ) = cpl_app;

h_arrayant.coupling = cpl;

% Clear internal varaible
h_arrayant.Pphase_diff = [];

end
