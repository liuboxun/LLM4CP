function isdual = check_dual_mobility( h_builder, show_warnings )
%CHECK_DUAL_MOBILITY Checks the input data of the builder
%
% Calling object:
%   Object array
%
% Description:
%   This function checks the input variables "simpar", "tx_array", "rx_array", "tx_track", and
%   "rx_track" of the builder object(s) for conformity. The data representation is adjusted such
%   that all other methods of the builder class can work with the variables without checking them
%   again. The method also checks if dual-mobility processing is required. This information is
%   stored in "qd_builder.dual_mobility" to be accessed by other methods.
%
% Input:
%   show_warnings
%   If set to true (default), 'qd_builder.check_dual_mobility' performs a set of tests to determine
%   if all provided input variables are correctly initialized. This can be disabled by setting
%   'show_warnings = 0'.
%
% Output:
%   isdual
%   A logical array indicating for each builder if dual-mobility processing is required (true) or
%   if single-mobility processing is done (false).
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

% Parse input variables.
if ~exist( 'show_warnings','var' ) || isempty( show_warnings )
    show_warnings = true;
end

if numel(h_builder) > 1
    
    sic = size( h_builder );
    isdual = false( sic );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        isdual(i1,i2,i3,i4) = check_dual_mobility( h_builder(i1,i2,i3,i4), show_warnings );
    end
    
else
    h_builder = h_builder(1,1);                                                 % Fix for Octave
    isdual = false;                                                             % No dual-mobility by default
    
    % If there is only a single Tx and Rx track in the builder, we can save computing time since we
    % don't need to check all "virtual" tracks for consistency.
    single_tx_track = false;
    single_rx_track = false;
    
    % Process rx-positions
    if ~isempty( h_builder.rx_track ) && ~isa(h_builder.rx_track(1,1),'qd_track')
            error('QuaDRiGa:qd_builder:check_dual_mobility:wrong_Rx_track_class',...
                'Rx-track must be of class "qd_track".');
    end
    if isempty( h_builder.rx_positions ) && ~isempty( h_builder.rx_track )      % No rx-position, but rx-track is given
        % Read rx-positions from rx-tracks
        n_mobiles = size( h_builder.rx_track,2 );
        rx_position = zeros( 3,n_mobiles );
        for r = 1 : n_mobiles                                                   % Read rx-position from rx tracks
            rx_position(:,r) = h_builder.rx_track(1,r).initial_position;
        end
        h_builder.rx_positions = rx_position;
    end
    
    % Read the number of mobiles
    n_mobiles = h_builder.no_rx_positions;
    
    if n_mobiles > 0
        
        % Check if simulation parameters are correct
        if isempty( h_builder.simpar ) || ~isa( h_builder.simpar ,'qd_simulation_parameters' )
            error('QuaDRiGa:qd_builder:check_dual_mobility:no_simpar',...
                'You must provide valid simulation parameters.');
        end
        
        n_freq = numel( h_builder.simpar(1,1).center_frequency );                    % Read the number of frequencies
        o_freq = ones(1,n_freq);
        o_mobiles = ones(1,n_mobiles);
        
        % Make sure the rx_tracks are correct
        if ~isempty( h_builder.rx_track ) && size( h_builder.rx_track,1 ) > 1
            error('QuaDRiGa:qd_builder:check_dual_mobility:rx_track_rows',...
                'Rx-track object array cannot have rows.');
        end
        if isempty( h_builder.rx_track )                                        % No rx-track given
            h_builder.rx_track = qd_track([]);                                  % Create new track
            h_builder.rx_track.initial_position = h_builder.rx_positions(:,1);  % Set first Rx position
            h_builder.rx_track.orientation = [0;0;0];                           % Default Rx orientation
            if n_mobiles > 1
                h_builder.rx_track = h_builder.rx_track(1,o_mobiles);           % Copy handles to other Rx
            end
            single_rx_track = true;                                             % Indicate that there is only one Rx track
            
        elseif size( h_builder.rx_track,2 ) == 1 && n_mobiles > 1               % Single rx-track, but different Rx locations
            % Make sure there are orientation information
            if isempty( h_builder.rx_track(1,1).orientation )                   % Add orientation information
                h_builder.rx_track(1,1).calc_orientation;
            end
            h_builder.rx_track(1,1).initial_position = h_builder.rx_positions(:,1);
            if n_mobiles > 1
                h_builder.rx_track = h_builder.rx_track(1,o_mobiles);           % Copy handles to other Rx
            end
            single_rx_track = true;                                             % Indicate that there is only one Rx track
            
        elseif size( h_builder.rx_track,2 ) ~= n_mobiles
            error('QuaDRiGa:qd_builder:check_dual_mobility:rx_track_size_mismatch',...
                'Number of rx-tracks must match the number of mobiles.');
            
        else % As many rx-tracks as there are mobiles
            for r = 1 : n_mobiles                                               % Make sure there is orientation information
                if isempty( h_builder.rx_track(1,r).orientation )               % Add orientation information
                    h_builder.rx_track(1,r).calc_orientation;
                end
            end
        end
        
        % Make sure the rx_arrays are correct
        if isempty( h_builder.rx_array )
            h_builder.rx_array = qd_arrayant;
        elseif ~isa(h_builder.rx_array(1,1),'qd_arrayant')
            error('QuaDRiGa:qd_builder:check_dual_mobility:wrong_Rx_array_class',...
                'Rx-antenna must be of class "qd_arrayant".');
        end
        
        if numel( h_builder.rx_array ) == 1                                     % Single Rx-array for all mobiles and frequencies
            h_builder.rx_array = h_builder.rx_array( o_freq, o_mobiles );       % Copy handles to all frequencies and mobiles
            
        elseif size( h_builder.rx_array,1 ) == n_freq && size( h_builder.rx_array,2 ) == 1 && n_mobiles > 1
            h_builder.rx_array = h_builder.rx_array( :, o_mobiles );            % Copy handles to all mobiles
            
        elseif size( h_builder.rx_array,1 ) == 1 && size( h_builder.rx_array,2 ) == n_mobiles && n_freq > 1
            h_builder.rx_array = h_builder.rx_array( o_freq, : );               % Copy handles to all frequencies
            
        elseif size( h_builder.rx_array,1 ) ~= n_freq || size( h_builder.rx_array,2 ) ~= n_mobiles
            error('QuaDRiGa:qd_builder:check_dual_mobility:rx_array_size_mismatch',...
                'Number of rx-array antennas must match the number of mobiles and frequencies.');
        end
        
        % Process the tx-positions
        if ~isempty( h_builder.tx_track ) && ~isa(h_builder.tx_track(1,1),'qd_track')
            error('QuaDRiGa:qd_builder:check_dual_mobility:wrong_Tx_track_class',...
                'Tx-track must be of class "qd_track".');
        end
        if isempty( h_builder.tx_position )                                     % No tx_position given
            if isempty( h_builder.tx_track )                                    % No tx-track given
                error('QuaDRiGa:qd_builder:check_dual_mobility:tx_position_undefined',...
                    'You must provide a tx position or tx track.');
                
            elseif size( h_builder.tx_track,2 ) == 1                            % Single tx track
                % Copy tx position from first tx-track
                h_builder.tx_position = h_builder.tx_track(1,1).initial_position(:,o_mobiles);
                
            elseif size( h_builder.tx_track,2 ) == n_mobiles                    % Multiple tx tracks
                tx_position = zeros( 3, n_mobiles );
                for t = 1 : n_mobiles
                    tx_position(:,t) = h_builder.tx_track(1,t).initial_position;
                end
                h_builder.tx_position = tx_position;
                
            elseif size( h_builder.tx_track,2 ) > 1 && n_mobiles == 1           % Multiple tx tracks but only one mobile
                n_mobiles = size( h_builder.tx_track,2 );                       % Adjust number of mobiles
                o_mobiles = ones( 1, n_mobiles );
                tx_position = zeros( 3, n_mobiles );
                for t = 1 : n_mobiles
                    tx_position(:,t) = h_builder.tx_track(1,t).initial_position;
                end
                h_builder.tx_position = tx_position;
                h_builder.rx_positions = h_builder.rx_positions( :,o_mobiles );	% Duplicate Rx positions
                h_builder.rx_track = h_builder.rx_track( 1,o_mobiles );        	% Duplicate Rx tracks
                h_builder.rx_array = h_builder.rx_array( 1,o_mobiles );       	% Duplicate Rx arrays
                
            else
                error('QuaDRiGa:qd_builder:check_dual_mobility:tx_track_size_mismatch',...
                    'Number of tx-tracks does not match the number of mobiles.');
            end
            
        elseif size( h_builder.tx_position,2 ) == 1 && n_mobiles > 1
            h_builder.tx_position = h_builder.tx_position(:,o_mobiles);         % Duplicate tx position for all mobiles
            
        elseif size( h_builder.tx_position,2 ) > 1 && n_mobiles == 1
            n_mobiles = size( h_builder.tx_position,2 );                        % Adjust number of mobiles
            o_mobiles = ones( 1, n_mobiles );
            h_builder.rx_positions = h_builder.rx_positions( :,o_mobiles );     % Duplicate Rx positions
            h_builder.rx_track = h_builder.rx_track( 1,o_mobiles );             % Duplicate Rx tracks
            h_builder.rx_array = h_builder.rx_array( 1,o_mobiles );             % Duplicate Rx arrays
            single_rx_track = true;                                             % Indicate that there is only one Rx track
            
        elseif size( h_builder.tx_position,2 ) ~= n_mobiles
            error('QuaDRiGa:qd_builder:check_dual_mobility:tx_position_size_mismatch',...
                    'Number of tx-positions does not match the number of mobiles.');
        end
        
        % Test for dual-mobility
        % Dual-mobility is needed if there are at least two different tx_positions
        tmp = h_builder.tx_position - h_builder.tx_position(:,o_mobiles);
        if any( abs( tmp(:) ) > 1e-5 )
            isdual = true;
        end
        
        % Check if the tx track object array has only one row
        if ~isempty( h_builder.tx_track ) && size( h_builder.tx_track,1 ) > 1
            error('QuaDRiGa:qd_builder:check_dual_mobility:tx_track_rows',...
                'Tx-track object array cannot have rows.');
        end
        
        % Dual-mobility is needed if there is at least one tx track with more than one snapshot
        if ~isempty( h_builder.tx_track )
            for t = 1 : size( h_builder.tx_track,2 )
                if h_builder.tx_track(1,t).no_snapshots > 1
                    isdual = true;
                end
            end
        end
        
        % Set the dual-mobility indicator of the builder object
        h_builder.dual_mobility = isdual;
        
        % Process tx tracks
        if isempty( h_builder.tx_track )                                        % No tx-track given
            h_builder.tx_track = qd_track([]);                                  % Create new track
            h_builder.tx_track.initial_position = h_builder.tx_position(:,1);   % Set first tx position
            h_builder.tx_track.orientation = [0;0;0];                           % Default tx orientation
            if n_mobiles > 1
                h_builder.tx_track = h_builder.tx_track(1,o_mobiles);           % Copy handles to other tx
            end
            single_tx_track = true;                                             % Indicate that there is only one Tx track
            
        elseif size( h_builder.tx_track,2 ) == 1 && n_mobiles > 1               % Single tx-track, but different tx locations
            % Make sure there are orientation information
            if isempty( h_builder.tx_track(1,1).orientation )                   % Add orientation information
                h_builder.tx_track(1,1).calc_orientation;
            end
            if n_mobiles > 1
                h_builder.tx_track = h_builder.tx_track(1,o_mobiles);           % Copy handles to other tx
            end
            single_tx_track = true;                                             % Indicate that there is only one Tx track
            
        elseif size( h_builder.tx_track,2 ) ~= n_mobiles
            error('QuaDRiGa:qd_builder:check_dual_mobility:tx_track_size_mismatch2',...
                'Number of tx-tracks must match the number of mobiles.');
            
        else % As many tx-tracks as there are mobiles
            for t = 1 : n_mobiles                                               % Make sure there is orientation information
                if isempty( h_builder.tx_track(1,t).orientation )               % Add orientation information
                    h_builder.tx_track(1,t).calc_orientation;
                end
            end
        end
        
        % Make sure the tx_arrays are correct
        if isempty( h_builder.tx_array )
            h_builder.tx_array = qd_arrayant;
            
        elseif ~isa(h_builder.tx_array(1,1),'qd_arrayant')
            error('QuaDRiGa:qd_builder:check_dual_mobility:wrong_Tx_array_class',...
                'Tx-antenna must be of class "qd_arrayant".');
        end
        
        if numel( h_builder.tx_array ) == 1                                     % Single Rx-array for all mobiles and frequencies
            h_builder.tx_array = h_builder.tx_array( o_freq, o_mobiles );       % Copy handles to all frequencies and mobiles
            
        elseif size( h_builder.tx_array,1 ) == n_freq && size( h_builder.tx_array,2 ) == 1 && n_mobiles > 1
            h_builder.tx_array = h_builder.tx_array( :, o_mobiles );            % Copy handles to all mobiles
            
        elseif size( h_builder.tx_array,1 ) == 1 && size( h_builder.tx_array,2 ) == n_mobiles && n_freq > 1
            h_builder.tx_array = h_builder.tx_array( o_freq, : );               % Copy handles to all frequencies
            
        elseif size( h_builder.tx_array,1 ) ~= n_freq || size( h_builder.tx_array,2 ) ~= n_mobiles
            error('QuaDRiGa:qd_builder:check_dual_mobility:tx_array_size_mismatch',...
                'Number of tx-array antennas must match the number of mobiles and frequencies.');
        end
        
        % Check if tracks have the same length
        if single_tx_track && single_rx_track
            n_mobiles = 1;                                                      % Only check first track
        end
        for r = 1 : n_mobiles
            % Check if the number of snapshots matches
            no_snapshots_tx = h_builder.tx_track(1,r).no_snapshots;
            no_snapshots_rx = h_builder.rx_track(1,r).no_snapshots;
            if no_snapshots_tx ~= 1 && no_snapshots_tx ~= no_snapshots_rx
                error('QuaDRiGa:qd_builder:check_dual_mobility:tx_rx_track_lenght_mismatch',...
                    'Tx tracks must have a single snapshot or match the number of snapshots in the rx-track.');
            end
            
            % Check if there are identical positions (causes NaNs in the coefficients)
            if no_snapshots_tx == 1 && no_snapshots_rx == 1
                dist = h_builder.rx_track(1,r).initial_position - h_builder.tx_track(1,r).initial_position;
                
            elseif no_snapshots_tx == 1
                dist = h_builder.rx_track(1,r).positions_abs - ...
                    h_builder.tx_track(1,r).initial_position(:,ones(1,no_snapshots_rx));
                
            else
                dist = h_builder.tx_track(1,r).positions_abs - h_builder.rx_track(1,r).positions_abs;
            end
            if any( all( abs(dist) < 1e-14, 1 ) )
                error('QuaDRiGa:qd_builder:check_dual_mobility:colocated_tx_rx',...
                    ['Tx "',h_builder.tx_track(1,r).name,'" and Rx "',...
                    h_builder.rx_track(1,r).name,'" are in the same location.']);
            end
        end
    end
    
end

end
