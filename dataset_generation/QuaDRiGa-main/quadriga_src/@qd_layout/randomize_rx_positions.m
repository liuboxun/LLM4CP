function randomize_rx_positions( h_layout, max_dist, min_height, max_height, track_length,...
    rx_ind, min_dist, orientation )
%RANDOMIZE_RX_POSITIONS Generates random RX positions and tracks
%
% Calling object:
%   Single object
%
% Description:
%   This method places the users in the layout at random positions. Users are placed uniformly in
%   the area defined by 'max_dist' and 'min_dist'. Each user will be assigned a linear track with
%   random direction unless a fixed orientation is provided as input argument. The random height of
%   the user terminal will be in between 'min_height' and 'max_height'.
%
% Input:
%   max_dist
%   The maximum distance from the layout center in [m]. Scalar value, default 50 m
%
%   min_height
%   The minimum user height in [m]. Scalar value, default 1.5 m
%
%   max_height
%   The maximum user height in [m]. Scalar value, default 1.5 m
%
%   track_length
%   The length of the linear track in [m]. Scalar value, default 1 m
%
%   rx_ind
%   A vector containing the receiver indices for which the positions should be generated. Default:
%   All receivers
%
%   min_dist
%   The minimum distance from the layout center in [m]. Scalar value, default 0 m
%
%   orientation
%   The fixed heading angle (direction of movement) in [rad] for all user terminals in mathematical
%   sense (0 points east, pi/2 points north). Scalar value. If no orientation is provided, all
%   tracks get assigned with a random heading angle. 
%
%   Alternatively, the orientation can be described by a 3-element vector: The first value describes
%   the "bank angle", i.e. the orientation around an axis drawn through the body of the vehicle from
%   tail to nose in the normal direction of movement. Positive rotation is clockwise (seen from the
%   pilot/drivers perspective). The second value describes the "tilt angle", i.e. the vertical angle
%   relative to the horizontal plane; positive values point upwards. The third value describes the
%   "heading angle". The bank and tilt angles are only used to orient the antenna of the MT. The
%   heading angle is used for both, the antenna orientation and the movement direction.
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

if numel( h_layout ) > 1 
   error('QuaDRiGa:qd_layout:randomize_rx_positions','randomize_rx_positions not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

% Parse input variables
if ~exist( 'max_dist' , 'var' ) || isempty( max_dist )
    max_dist = 50;
elseif ~( all(size(max_dist) == [1 1]) &&...
        isnumeric(max_dist) &&...
        isreal(max_dist) &&...
        max_dist > 0 )
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "max_dist" must be a real scalar  > 0')
end

if ~exist( 'min_height' , 'var' ) || isempty( min_height )
    min_height = 1.5;
elseif ~( all(size(min_height) == [1 1]) &&...
        isnumeric(min_height) &&...
        isreal(min_height) )
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "min_height" must be a real scalar')
end

if ~exist( 'max_height' , 'var' ) || isempty( max_height )
    max_height = 1.5;
elseif ~( all(size(max_height) == [1 1]) &&...
        isnumeric(max_height) &&...
        isreal(max_height) )
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "max_height" must be a real scalar  > 0')
end

if ~exist( 'track_length' , 'var' ) || isempty( track_length )
    track_length = 1;
elseif ~( all(size(track_length) == [1 1]) &&...
        isnumeric(track_length) &&...
        isreal(track_length) )
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "track_length" must be a real scalar  > 0')
end

if ~exist( 'rx_ind' , 'var' ) || isempty( rx_ind )
    rx_ind = 1:1:h_layout.no_rx;
elseif islogical( rx_ind )
    rx_ind = find( rx_ind );
end

if ~exist( 'min_dist' , 'var' ) || isempty( min_dist )
    min_dist = 0;
elseif ~( all(size(min_dist) == [1 1]) &&...
        isnumeric(min_dist) &&...
        isreal(min_dist) &&...
        min_dist > 0 )
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "min_dist" must be a real scalar  > 0')
elseif min_dist > max_dist
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "min_dist" must be smaller than "max_dist"')
end

if ~exist( 'orientation' , 'var' ) || isempty( orientation )
    orientation = [];
elseif all(size( orientation ) == [1,1]) && isnumeric( orientation ) && isreal( orientation )
    orientation = [ 0;0;orientation ];
elseif ~( all(size(orientation) == [3,1]) && isnumeric(orientation) && isreal(orientation) )
    error('QuaDRiGa:qd_layout:randomize_rx_positions','??? "orientation" must be a real scalar or 3-element vector')
end

% Read some default variables
rx_track = h_layout.rx_track;
no_tx = h_layout.no_tx;
samples_per_meter = h_layout.simpar(1,1).samples_per_meter;
sc = ( min_dist / max_dist )^2;

% Generate random positions and tracks
for i_rx = 1:numel( rx_ind )
    n = rx_ind( i_rx );
    
    % Get random initial position of the MT
    a = (1-sc) * rand + sc;
    a = ( max_dist * sqrt(a) ) .* exp( 1j*2*pi*( rand-0.5 ) );
    b = rand * (max_height - min_height) + min_height;
    
    % Generate track
    if track_length > 0 && isempty( orientation )           % No orientation specified
        trk = qd_track.generate( 'linear',track_length );
    elseif track_length > 0                                 % Orientation provided
        trk = qd_track.generate( 'linear',track_length, orientation(3,1) );
        trk.orientation = orientation(:,[1,1]);             % Set tilt and bank
    elseif track_length == 0 && isempty( orientation )      % Random heading
        trk = qd_track([]);
        trk.orientation(3,1) = rand*2*pi-pi;
    else
        trk = qd_track([]);
        trk.orientation = orientation;
    end
    
    trk.name = ['Rx',sprintf('%04d',n)];
    trk.initial_position = [ real(a) ; imag(a) ; b ];
    if track_length > 0
        trk.interpolate_positions( samples_per_meter );
    end
    scenarios = rx_track(1,n).scenario(1);
    trk.scenario = scenarios( ones(1,no_tx),1 );
    
    % Write new track to layout
    rx_track(1,n) = trk;
end
h_layout.rx_track = rx_track;

end
