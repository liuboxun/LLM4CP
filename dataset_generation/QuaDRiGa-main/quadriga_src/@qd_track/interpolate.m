function [ dist, h_track_int ] = interpolate( h_track, method, sample_rate, movement_profile, ...
    algorithm, update_input )
%INTERPOLATE Interpolates the snapshot positions along the track
%
% Calling object:
%   Single object
%
% Description:
%   This function interpolates the positions along the track such that it matches the given sample
%   rate. The channel model operates on a position-based sample grid. That means that the 'builder'
%   generates one CIR for each position on the track. In practise, however, a time continuous
%   evolution of the CIR is often needed. This can be interpolated from the position-based grid,
%   provided that the spatial sample theorem is not violated (i.e. the channel needs to be sampled
%   at least twice per half wave length). In order to do that, enough sample points are needed
%   along the track. There are two methods to obtain the missing sample points:
%
%   distance
%   The distance-based interpolation calculates the missing sample points and places them equally
%   spaced along the track. This corresponds to a constant speed when evaluating the output CIR.
%   The required minimum value for 'sample_rate' is equal to '1/samples_per_meter' from the
%   'qd_simulation_parameters' object.
%
%   time
%   The time-based interpolation is needed in dual-mobility scenarios, where Tx and Rx can move at
%   different speeds. In this case the sample rate is given in [seconds] and a movement profile
%   must be given. If you use the set_speed function of the qd_track object, the movement profile
%   is automatically generated.
%
%   snapshot
%   Time-based interpolation where positions on the track are given by snapshot numbers instead of
%   distances. This is useful when the device is stationary but changes it's orientation over time,
%   e.g. a radar antenna. The sample rate must be given in [seconds]. The movement profile must
%   contain the time-points (first row) vs. snapshot number (second row). Snapshot indices start
%   from 1. Fractional number are possible (e.g. 1.5).
%
%
% Input:
%   method
%   Must be either 'distance', 'time' or 'snapshot'.
%
%   sample_rate
%   The interval between two (output) snapshots given in [meters] for distance-based interpolation
%   and [seconds] for time-based and snapshot-based interpolation.
%
%   movement_profile
%   A matrix describing the time (in seconds) vs. distance (in meters) for time-based
%   interpolation; or (input) snapshot-number vs. distance for snapshot-based interpolation. The
%   first row describes the time points, the second row describes the positions on the track
%   relative to the start point.
%
%   algorithm
%   Selects the interpolation algorithm. Optional are
%      * linear - Linear interpolation (default)
%      * cubic  - Piecewise cubic interpolation (shape preserving)
%    Interpolation of the orientations are always done using the SLERP algorithm (spherical linear
%    interpolation).
%
%   update_input
%   When set to 'true', the input track object is updated. If set to 'false' (default), a new track
%   object is created.
%
% Output:
%   dist
%   The interpolated distances relative to the track start point.
%
%   h_track_int
%   A new 'qd_track' object containing the interpolated positions, orientations, segments and
%   parameters. If 'update_input' is set to true, the calling track handle is returned.
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

if numel( h_track ) > 1
    error('QuaDRiGa:qd_track:interpolate:object_array',...
        '"qd_track.interpolate" is not definded for object arrays.');
else
    h_track = h_track(1,1); % workaround for octave
end

% Check sample rate
if ~exist('sample_rate','var') || numel( sample_rate ) ~= 1
    error('QuaDRiGa:qd_track:interpolate:invalid_sample_rate','Sample rate is invalid.');
end

% Check movement profile
if ~exist( 'movement_profile','var' ) || isempty( movement_profile )
    movement_profile = h_track.movement_profile;
end

% Select the interpolation algorithm
if exist('algorithm','var') && ~isempty(algorithm)
    switch algorithm
        case 'linear'
            use_linear_interpolation = true;
        case 'cubic'
            use_linear_interpolation = false;
        otherwise
            error('Track interpolation method type not found; supported are: linear, cubic');
    end
else
    use_linear_interpolation = true;
end

if ~exist( 'update_input','var' ) || isempty( update_input )
    if nargout == 0
        update_input = true;
    else
        update_input = false;
    end
end

no_snapshots = h_track.no_snapshots;                                % Read number of snapshots

% Calculate distance vector of the input track
positions = h_track.positions;                                      % Read positions for faster processing
[ max_dist, dist_in ] = get_length( h_track );

% Calculate interpolation target
do_not_interpolate = false;
switch method
    case 'distance'
        
        if no_snapshots == 1                                      	% Check number of input snaps
            error('QuaDRiGa:qd_track:interpolate:single_snapshot_in',...
                'For "distance" interpolation, the number of input snapshots must be > 1.');
        end
        
        no_snapshots_out = ceil((max_dist-sample_rate/2) / sample_rate ); % Number of snapshot in the output channel
        
        if no_snapshots_out < 2                                         % Check number of output snaps
            do_not_interpolate = true;
            dist = dist_in;
        else
            sr = max_dist / no_snapshots_out;                           % Calculate the effective sample rate
            dist = ( 0 : no_snapshots_out )*sr;                        	% The interpolation distances
        end
        
    case 'time'
        
        if isempty( movement_profile )
            error('QuaDRiGa:qd_track:interpolate:no_movement_profile',...
                'For "time" interpolation, you must provide a movement profile.');
        end
        
        max_time = movement_profile(1,end);                        	% Get the maximum time
        
        if movement_profile(2,end) > max_dist                    	% Check track length
            error('QuaDRiGa:qd_track:interpolate:no_movement_profile',...
                'Maximum distance in the movement profile exceeds track length');
        else
            max_dist = movement_profile(2,end);
        end
        
        if sample_rate > max_time                                   % Number of snapshot in the output channel
            no_snapshots_out = 1;
        else
            no_snapshots_out = ceil((max_time-sample_rate/2) / sample_rate ); 
        end
        sr = max_time / no_snapshots_out;                           % Calculate the effective sample rate
        time = ( 0 : no_snapshots_out )*sr;                       	% The snapshot times
        
        if use_linear_interpolation                                 % Interpolate the distances
            dist = qf.interp( movement_profile(1,:), 0, movement_profile(2,:), time, [], true );
        else
            dist = pchip( movement_profile(1,:), movement_profile(2,:), time );
        end
        
    case 'snapshot'
        
        % Check movement profile
        if ~exist( 'movement_profile','var' ) || isempty( movement_profile )
            if isempty( h_track.movement_profile )
                error('QuaDRiGa:qd_track:interpolate:no_movement_profile',...
                    'For "snapshot" interpolation, you must provide a movement profile.');
            else
                movement_profile = h_track.movement_profile;
            end
        end
        if movement_profile(2,end) > no_snapshots                   % Check track length
            error('QuaDRiGa:qd_track:interpolate:no_movement_profile',...
                'Maximum snapshot number in the movement profile exceeds number of snapshots in the track.');
        elseif movement_profile(2,1) < 1
            error('QuaDRiGa:qd_track:interpolate:no_movement_profile',...
                'Minimum snapshot number in the movement profile must be >= 1.');
        end
        
        dist_in = 0 : no_snapshots - 1;                             % Translate snapshots into distance
        max_time = movement_profile(1,end);                        	% Get the maximum time
        max_dist = no_snapshots-1;                                  % Get the maximum distance
        
        no_snapshots_out = ceil((max_time-sample_rate/2) / sample_rate ); % Number of snapshot in the output channel
        sr = max_time / no_snapshots_out;                           % Calculate the effective sample rate
        time = ( 0 : no_snapshots_out )*sr;                       	% The snapshot times
        
        if use_linear_interpolation                                 % Interpolate the distances
            dist = qf.interp( movement_profile(1,:), 0, movement_profile(2,:), time, [], true ) - 1;
        else
            dist = pchip( movement_profile(1,:), movement_profile(2,:), time );
        end
        
    otherwise
        error('QuaDRiGa:qd_track:interpolate:invalid_method',...
            'Interpolation method must be either "distance" or "time".');
end

dist( dist<0 ) = 0;                                                 % Fix interpolation artefacts
dist( dist>max_dist ) = max_dist;

% Create interpolated track object
if do_not_interpolate
    h_track_int = copy( h_track );
    
elseif update_input || nargout > 1                                      % Only interpolate if required
    no_snapshots_out = numel( dist );                               % Actual number of output snapshots
    if h_track.closed && dist_in(end) - dist(end) < 1e-6            % Check if output track is closed
        closed = true;
    else
        closed = false;
    end
    
    % Process positions
    if numel( dist_in ) > 1                                         % Dynamic track
        pos_int = zeros( 3,no_snapshots_out );
        for n = 1 : 3                                               % Interpolate the positions
            if use_linear_interpolation
                pos_int(n,:) = qf.interp( dist_in, 0, positions(n,:), dist, [], true );
            else
                pos_int(n,:) = pchip( dist_in, positions(n,:), dist );
            end
        end
        if closed                                                   % Start / End must be identical for closed tracks
            pos_int(:,end) = pos_int(:,1);
        end
            
        % Adjust the segments on the track
        segment_index_int = h_track.segment_index;
        no_seg = numel( segment_index_int );
        for i_seg = 1 : no_seg
            [ ~, segment_index_int(i_seg) ] = min( abs( dist - dist_in( segment_index_int(i_seg) ) ) );
        end
        
        % Interpolate parameters, if provided
        if ~isempty( h_track.par )
            par = h_track.par;                                    	% Copy "par" from input track
            names = {'kf','pg'};                                   	% Only KF and PG change with distance
            
            for i_par = 1 : numel(names)
                val = par.( names{i_par} );
                if ~isempty( val )
                    if use_linear_interpolation
                        val_int = qf.interp( dist_in, 0, val, dist );
                    else
                        val_int = pchip( dist_in, val, dist );
                    end
                    if closed
                        val_int(end) = val_int(1);
                    end
                    par.( names{i_par} ) = val_int;
                end
            end
        else
            par = [];
        end
        
        % Interpolate the orientations
        orientation = h_track.orientation;
        if ~isempty( orientation )
            orientation_int = zeros(3,no_snapshots_out);
            
            % Roll (full circle)
            % We need "Spherical Linear Interpolation" here. This is not implemented in MATLAB.
            orientation_int(1,:) = qf.slerp( dist_in, orientation(1,:), 0, dist, true );
            
            % Pitch and Yaw
            [ orientation_int(3,:), orientation_int(2,:) ] =...
                qf.slerp( dist_in, orientation(3,:), orientation(2,:), dist, true  );
        else
            orientation_int = [];
        end
        
        if strcmp( method, 'snapshot' )
            % Update the snapshot indices
            movement_profile(2,:) = ...
                qf.interp( movement_profile(1,[1,end]), 0, [1,no_snapshots_out], movement_profile(1,:));
            movement_profile(2, movement_profile(2,:) > no_snapshots_out ) = no_snapshots_out;
            movement_profile(2, movement_profile(2,:) < 1 ) = 1;
        else
            % Recalculate "dist" based on the interpolated positions
            dist = [ 0 , cumsum(  sum( ( pos_int(:,2:end) - pos_int(:,1:end-1) ).^2,1 ).^0.5 ) ];
            length = dist( end );
            
            % The maximum distance in the movement profile might have chaneged due to interpolation
            % artefacts. This is fixed here.
            if ~isempty( movement_profile )
                ii = movement_profile(2,:) > length;
                movement_profile(2,ii) = length;
            end
        end
    else                                                                % Static track
        o_snapshots_out = ones(1,no_snapshots_out);
        pos_int = positions(:,o_snapshots_out);
        orientation_int = h_track.orientation(:,o_snapshots_out);
        segment_index_int = 1;
        no_seg = 1;
        if ~isempty( h_track.par )
            par = h_track.par;                                          % Copy "par" from input track
            names = {'kf','pg'};                                        % Only KF and PG change with distance
            for i_par = 1 : numel(names)
                val = par.( names{i_par} );
                if ~isempty( val )
                    val_int = val(:,o_snapshots_out);
                    par.( names{i_par} ) = val_int;
                end
            end
        else
            par = [];
        end
        if all(size( movement_profile ) == [2,2]) && all(movement_profile(2,:) == [1 1])
            movement_profile(2,2) = no_snapshots_out;
        else % Interpolate Movement profile
            movement_profile(2,:) = ...
                qf.interp( movement_profile(1,[1,end]), 0, [1,no_snapshots_out], movement_profile(1,:));
        end
    end
    
    if update_input % Write output to current track opject
        h_track.Ppositions = pos_int;                                   % Interpolated positions
        h_track.Pno_snapshots = no_snapshots_out;                       % New number of snapshots
        h_track.Porientation = orientation_int;                         % Interpolated orientations
        h_track.Psegment_index = segment_index_int;                     % Interpolated segment positions
        h_track.Pmovement_profile = movement_profile;                   % Set movement profile
        h_track_int = h_track;                                          % Copy handle
        
    else % Create new output track
        h_track_int = qd_track([]);                                     % New track object
        h_track_int.name = h_track.name;                                % Copy name
        h_track_int.Pinitial_position = h_track.Pinitial_position;      % Same initial position
        h_track_int.Ppositions = pos_int;                               % Interpolated positions
        h_track_int.Pno_snapshots = no_snapshots_out;                   % New number of snapshots
        h_track_int.Porientation = orientation_int;                     % Interpolated orientations
        h_track_int.Pno_segments = no_seg;                              % Same number of segments
        h_track_int.Psegment_index = segment_index_int;                 % Interpolated segment positions
        h_track_int.Pscenario = h_track.Pscenario;                      % Same scenarios
        h_track_int.Pmovement_profile = movement_profile;               % Set movement profile
    end
    
end

if strcmp( method, 'snapshot' )
    % Return snapshot number
    dist = dist + 1;
end

end
