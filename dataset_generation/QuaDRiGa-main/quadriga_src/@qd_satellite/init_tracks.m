function h_qd_track = init_tracks( h_qd_satellite, ue_pos, t, i_sat, only_visible )
%INIT_TRACKS Creates qd_track objects containing the satellite positions in local coordinates
%
% Calling object:
%   Single object
%
% Description:
%   The QuaDRiGa reference coordinate system is defined by Cartesian (x,y,z) coordinates. Hence,
%   local QuaDRiGa coordinates do not take the curvature of the Earth into account. This method
%   attaches a tangential plane at the UE positions given by geographic coordinates (longitude,
%   latitude). The origin (0,0,0) of the local QuaDRiGa coordinates is is placed at the UE position
%   and the satellite coordinates at time 't' are transformed into local coordinates. The
%   'qd_track' object can be used as 'tx_track' in a 'qd_layout' object.
%
% Input:
%   ue_pos
%   A two-element vector defining the origin of the tangential plane on the surface of the Earth in
%   geographic coordinates (latitude,longitude) in [degree]
%
%   t
%   Vector containing the T time points in seconds relative to the epoch (simulation start).
%   Alternatively, it is possible to use the current clock-time by providing a string describing
%   the clock offset relative to UTC time, e.g. 'utc+02' stands for Central European Summer Time
%   (CEST).
%
%   i_sat
%   Vector containing the S satellite indices for which the orbit should be predicted (optional).
%   By default, all satellites are used.
%
%   only_visible
%   If set to true (default), the output contains only satellites where at least one part of the
%   orbital trajectory is above the horizon.
%
% Output:
%   h_qd_track
%   A vector of 'qd_track' object handles containing the satellite trajectories for each visible
%   satellite.
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

if numel( h_qd_satellite ) > 1
    error('QuaDRiGa:h_qd_satellite:init_tracks','init_tracks not defined for object arrays.');
else
    h_qd_satellite = h_qd_satellite(1,1); % workaround for octave
end

% Parse input arguments
if ~exist('ue_pos','var') || isempty( ue_pos )
    ue_pos = [ 13.3249472 , 52.5163194 ];  % HHI
end

if ~exist('t','var') || isempty( t )
    t = 0;
elseif ischar(t) && numel(t) == 6 && strcmp(t(1:3),'utc') && ~isempty( h_qd_satellite.epoch )
    t = ( now - str2double(t(4:6))/24 - h_qd_satellite.epoch)*86400;
elseif ~( isnumeric(t) && isreal(t) )
    error('QuaDRiGa:h_qd_satellite:init_layout','Invalid time offset or epoch not given.');
else
    t = reshape( t,[],1 );
end

if ~exist('i_sat','var') || isempty( i_sat )
    i_sat = 1 : h_qd_satellite.n_satellites;
end

if ~exist('only_visible','var') || isempty( only_visible )
    only_visible = true;
end

% Determine the satellites that should be  in the layout
% We also calculate "DxyzU" here so save computing time
if only_visible
    [ ~,i_visible ] = h_qd_satellite.ue_perspective( ue_pos, t+0.1 );   % Sat-positions and visibility
    i_sat = intersect( find( any(i_visible,1) ),i_sat);                 % Update indices
end

nS = numel( i_sat );
nT = numel( t );
oT = ones( 1,nT );

% Generate QuaDRiGa track objects
h_qd_track = qd_track([]);
for n = 1 : nS
    iS = i_sat( n );
    
    % Calculate Satellite positions at timepoint t
    [ xyzU,~,orientation ] = h_qd_satellite.ue_perspective( ue_pos, t, iS ); 
    
    % Speed in km/s
    % speed = sqrt(sum((DxyzU(:,:,n)-xyzU).^2,1))*10;
    
    % Write data to track
    h_qd_track(1,n) = qd_track([]);
    if ~isempty( h_qd_satellite.sat_name )
        h_qd_track(1,n).name = h_qd_satellite.sat_name{1,iS};
    end
    h_qd_track(1,n).initial_position = xyzU(:,1)*1e3;              % Values must be in [m]
    h_qd_track(1,n).positions = (xyzU - xyzU(:,oT))*1e3;
    h_qd_track(1,n).orientation = orientation*pi/180;              % Values in [rad]
    h_qd_track(1,n).ReferenceCoord = ue_pos;                       % Save reference position
    
    % Movement profile
    if h_qd_track(1,n).no_snapshots > 1
        [~,dist] = get_length( h_qd_track(1,n) );
        h_qd_track(1,n).movement_profile = [ t'-t(1); dist ];          
    end
end

if nS == 1 % Fix for Octave
    h_qd_track = h_qd_track(1,1); 
end

end
