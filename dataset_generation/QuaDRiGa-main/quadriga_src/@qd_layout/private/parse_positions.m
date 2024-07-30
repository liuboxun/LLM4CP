function [ rx_pos, tx_pos, rx_ind, rx_track, tx_track ] = parse_positions( h_layout, i_rx, i_tx )
%PARSE_POSITIONS Extracts the individual Tx and Rx positions from the layout
%
% Calling object:
%   Single object
%
% Description:
%   This function processes all Tx and Rx positions in a layout and extracts relevant information
%   for the channel builder.
%
% Input:
%   i_rx
%   Vector of Rx indices (Default: all Rxs)
%
%   i_tx
%   Vector of Tx indices (Default: all Txs)
%
% Output:
%   rx_pos
%   Coordinates for Rx positions (segment start positions) in [m] given as [3 x N] matrix. N
%   corresponds to the number of all segments in the layout. The rows correspond to the x,y and z
%   coordinate.
%
%   tx_pos
%   Coordinates for Tx positions (segment start positions) in [m] given as [3 x N x T] matrix. T
%   corresponds to the number of transmitters in the layout.
%
%   rx_ind
%   [2 x N] matrix containing the segment-to-rx linking information. The first row contains the Rx-
%   index. The second row contains the segment-index for each Rx.
%
%   rx_track
%   [N x T] array of "qd_track" objects containing the receive positions, the scenario, and
%   optional LSPs for each Tx-Rx link.
%
%   tx_track
%   [N x T] array of "qd_track" objects containing the transmit positions, for each Tx-Rx link.
%
%   rx_array
%   [F x N x T] array of "qd_arrayant" objects containing the receive antennas for each Tx-Rx link.
%   F corresponds to the number of all carrier frequencies.
%
%   tx_array
%   [F x N x T] array of "qd_arrayant" objects containing the transmit antennas for each Tx-Rx link.
%   F corresponds to the number of all carrier frequencies.
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

if numel( h_layout ) > 1
    error('QuaDRiGa:qd_layout:parse_positions','parse_positions not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

no_rx = h_layout.no_rx;
no_tx = h_layout.no_tx;

rxT = h_layout.rx_track;
txT = h_layout.tx_track;

% Check if the scenario definition in all tracks matches the number of transmitters in the layout
o_tx = ones( 1,no_tx );
for r = 1 : no_rx
   if size( rxT(1,r).scenario,1 ) == 1 && no_tx > 1
       rxT(1,r).scenario = rxT(1,r).scenario(o_tx,:);
   elseif size( rxT(1,r).scenario,1 ) ~= no_tx
       error('QuaDRiGa:qd_layout:parse_positions',...
           'Number of rows in the scenario definition does not match the number of transmitters in the layout.');
   end
end

if ~exist( 'i_rx' , 'var' ) || isempty(i_rx)
    i_rx = 1 : no_rx;
end

if ~exist( 'i_tx' , 'var' ) || isempty(i_tx)
    i_tx = 1 : no_tx;
end

% Track splitting is time consuming. Hence, we only do it if it is needed.
if nargout > 3
    split_tracks = true;
else
    split_tracks = false;
end

% The method "qd_layout.set_scenario" needs the tx-rx distance to determine the LOS probability
% The segment positions on a rx track determine the scenario. However, since both tx and rx move, we
% must adjust the tx positions as well to get the tx-rx distance.
maxV = 0;
timebased = h_layout.dual_mobility & nargout < 4;
for r = 1 : numel( i_rx )
    timebased = timebased & ~isempty( rxT(1,i_rx(1,r)).movement_profile );
    if timebased
        maxV = max(maxV,rxT(1,i_rx(1,r)).movement_profile(2,end)./rxT(1,i_rx(1,r)).movement_profile(1,end));
    end
end
for t = 1 : numel( i_tx )
    timebased = timebased & ~isempty( txT(1,i_tx(1,t)).movement_profile );
    if timebased
        maxV = max(maxV,txT(1,i_tx(1,t)).movement_profile(2,end)./txT(1,i_tx(1,t)).movement_profile(1,end));
    end
end
if timebased
    if ~isempty( h_layout.update_rate )
        sample_rate = h_layout.update_rate;
    else
        sample_rate = 1/(h_layout.simpar(1,1).samples_per_meter*maxV);
    end
    for r = 1 : numel( i_rx )
        [~,rxT(1,i_rx(1,r))] = interpolate( rxT(1,i_rx(1,r)), 'time', sample_rate, [], [], 0 );
    end
    for t = 1 : numel( i_tx )
        [~,txT(1,i_tx(1,t))] = interpolate( txT(1,i_tx(1,t)), 'time', sample_rate, [], [], 0 );
    end
end

% Check if scenario definition matches the number of snapshots
rx_ind = [];
for ir = 1 : numel( i_rx )
    r = i_rx( ir );
    if size( rxT(1,r).scenario, 2 ) ~= rxT(1,r).no_segments
        error('QuaDRiGa:qd_layout:parse_positions',...
            ['"',rxT(1,r).name,'": Number of scenarios does not match the number of segments.'])
    end
    no_seg = rxT(1,r).no_segments;
    rx_ind = [ rx_ind , [ones(1,no_seg)*r ; 1:no_seg] ];
end
no_seg = size( rx_ind, 2 );

% Interpolate tx tracks that have more that 1 snapshot (dual-mobility)
for t = 1 : numel( i_tx )
    it  = i_tx( t );

    % Length of the track
    max_dist = get_length( txT(1,it) );
    
    if txT(1,it).no_snapshots > 1 && ~isempty( h_layout.update_rate ) && ~isempty( txT(1,it).movement_profile )
        
        % Time-based interpolation
        max_time = txT(1,it).movement_profile(1,end);
        sample_rate = h_layout.update_rate;
        no_snapshots_out = ceil((max_time-sample_rate/2) / sample_rate ) + 1;
        
        % Interpolate Tx track
        if txT(1,it).no_snapshots ~= no_snapshots_out
            txT(1,it) = copy( txT(1,it) );
            interpolate( txT(1,it), 'time', sample_rate );
        end
        
    elseif txT(1,it).no_snapshots > 1 && max_dist > 1e-6 && isempty( txT(1,it).movement_profile )
        % If there is a movement profile, the tracks are interpolated in "qd_layout.get_channels"
        % Without a movement profile, all tracks must have the same length. In this case, we
        % interpolate them according to the sample density.
        
        % Distance-based interpolation
        sample_rate = 1/h_layout.simpar(1,1).samples_per_meter;
        
        % Interpolate the tx track
        no_snapshots_out = ceil((max_dist-sample_rate/2) / sample_rate ) + 1;
        if txT(1,it).no_snapshots ~= no_snapshots_out
            txT(1,it) = copy( txT(1,it) );
            interpolate( txT(1,it), 'distance', sample_rate );
        end
    end
    
end

% Interpolate rx tracks that have more that 1 snapshot (dual-mobility)
for r = 1 : numel( rxT )
    
    % Length of the track
    max_dist = get_length( rxT(1,r) );
    
    if rxT(1,r).no_snapshots > 1 && ~isempty( h_layout.update_rate ) && ~isempty( rxT(1,r).movement_profile )
        
        % Time-based interpolation
        max_time = rxT(1,r).movement_profile(1,end);
        sample_rate = h_layout.update_rate;
        no_snapshots_out = ceil((max_time-sample_rate/2) / sample_rate ) + 1;
        
        % Interpolate Tx track
        if rxT(1,r).no_snapshots ~= no_snapshots_out
            rxT(1,r) = copy( rxT(1,r) );
            interpolate( rxT(1,r), 'time', sample_rate );
        end
        
    elseif h_layout.dual_mobility && rxT(1,r).no_snapshots > 1 && max_dist > 1e-6 && isempty( rxT(1,r).movement_profile )
        % If there is a movement profile, the tracks are interpolated in "qd_layout.get_channels"
        % Without a movement profile, all tracks must have the same length. In this case, we
        % interpolate them according to the sample density.
           
        % Distance-based interpolation
        sample_rate = 1/h_layout.simpar(1,1).samples_per_meter;
        
        % Interpolate the tx track
        no_snapshots_out = ceil((max_dist-sample_rate/2) / sample_rate ) + 1;
        if rxT(1,r).no_snapshots ~= no_snapshots_out
            rxT(1,r) = copy( rxT(1,r) );
            interpolate( rxT(1,r), 'distance', sample_rate );
        end
    end
end

% Initialize track splitting
if split_tracks
    rx_track = qd_track([]);
    tx_track = qd_track([]);
end

% Parse the positions
rx_pos = zeros( 3, no_seg );                        % MT positions
tx_pos = zeros( 3, no_seg, numel(i_tx) );           % Corresponding BS positions
for r = 1 : no_seg
    segment_index    = rxT(1,rx_ind(1,r)).segment_index(1,rx_ind(2,r));
    initial_position = rxT(1,rx_ind(1,r)).initial_position;
    track_position   = rxT(1,rx_ind(1,r)).positions(:,segment_index );
    rx_pos( :,r )    = initial_position + track_position;
    
    for t = 1 : numel( i_tx )
        it = i_tx( t );
        
        initial_position = txT(1,it).initial_position;
        if txT(1,it).no_snapshots > 1                     % Tx is mobile
            
            if rxT(1,rx_ind(1,r)).no_snapshots == 1       % Tx is mobile, Rx is static
                % Get the Tx position relative to the tx initial position
                track_position = [ 0;0;0 ];
                
                if split_tracks
                    % Copy Rx and Tx tracks
                    rx_track(r,t) = get_subtrack( rxT(1,rx_ind(1,r)), 1, t );
                    tx_track(r,t) = copy( txT(1,it) );
                    
                    % Duplicate Rx-track positions so that for each Tx-pos there is a Rx-pos
                    oS = ones(1,tx_track(r,t).no_snapshots);
                    if isempty( rx_track(r,t).orientation )
                        rx_track(r,t).orientation = [0;0;0];
                    end
                    rx_orientation = rx_track(r,t).orientation;
                    rx_track(r,t).positions = rx_track(r,t).positions(:,oS);
                    rx_track(r,t).orientation = rx_orientation(:,oS);
                end
                
            elseif txT(1,it).no_snapshots ~= rxT(1,rx_ind(1,r)).no_snapshots
                error('QuaDRiGa:qd_layout:parse_positions',...
                    ['"',txT(1,it).name,'_',rxT(1,rx_ind(1,r)).name,...
                    '": Assigned tracks must have the same number of snapshots.']);
                
            else % Tx is mobile, Rx is mobile
                
                if split_tracks
                    % Initialize the temporary tracks with default variables 
                    rx_track_tmp = rxT(1,rx_ind(1,r));
                    tx_track_tmp = txT(1,it);
                    if rx_track_tmp.closed && ~tx_track_tmp.closed
                        rx_track_tmp = copy( rx_track_tmp );
                        rx_track_tmp.positions(1,end) = rx_track_tmp.positions(1,end) + 1e-14;
                    elseif ~rx_track_tmp.closed && tx_track_tmp.closed
                        tx_track_tmp = copy( tx_track_tmp );
                        tx_track_tmp.positions(1,end) = tx_track_tmp.positions(1,end) + 1e-14;               
                    end
                    
                    if rx_track_tmp.no_segments > 1
                        % Extract Rx track segment
                        rx_track(r,t) = get_subtrack( rx_track_tmp, rx_ind(2,r), t );
                        
                        % Extract Tx track segment
                        tx_track_tmp.segment_index = rx_track_tmp.segment_index;
                        tx_track(r,t) = get_subtrack( tx_track_tmp, rx_ind(2,r) );
                        tx_track(r,t).name = tx_track_tmp.name;
                        tx_track_tmp.segment_index = 1;
                    else
                        % Copy Rx and Tx tracks
                        rx_track(r,t) = get_subtrack( rx_track_tmp, 1, t );
                        tx_track(r,t) = get_subtrack( tx_track_tmp, 1 );
                    end
                end
            end
            
        else % Tx is static
            if split_tracks
                % Extract Rx track segment
                trk = rxT(1,rx_ind(1,r));
                if trk.no_segments == 1 && isempty( trk.par )
                    rx_track(r,t) = qd_track([]);
                    copy( rx_track(r,t), trk );
                    rx_track(r,t).scenario = trk.scenario{t,1};
                else
                    rx_track(r,t) = get_subtrack( trk, rx_ind(2,r), t );
                end
                tx_track(r,t) = txT(1,it);                % Copy handle
            end
            
            % Get the relative Tx positions
            track_position = [ 0;0;0 ];
        end
        tx_pos(:,r,t) = initial_position + track_position;
    end
end

end
