function [ h_channel, h_builder ] = get_channels( h_layout, check_parfiles, overlap, algorithm, init_builder )
%GET_CHANNELS Generate the channel coefficients.
%
% Calling object:
%   Single object
%
% Description:
%   This method executes all necessary functions to generate the channel coefficients. These are:
%
%   1. Interpolation of the tracks to match the sample density. This avoids unnecessary
%      computations. The minimum sample density (qd_layout.simpar.sample_density) is 1 sample per
%      half-wavelength for static transmitters and 2 samples per half-wavelength for mobile
%      transmitters (dual-mobility). The default value is 2.5. The interpolation is only done if a
%      sample rate is provided.
%   2. Generation of channel builder objects and assigning track segments to builders.
%   3. Generation of large and small-scale-fading parameters, including spatial consistency.
%   4. Splitting of builder object for multi-frequency simulations (only when multiple carrier
%      frequencies are given in (qd_layout.simpar.center_frequency).
%   5. Generation of drifting channel coefficients for each track-segment.
%   6. Merging of channel segments, including modeling the birth and death of scattering clusters.
%   7. Interpolation of channel coefficients to match the sample rate (only if sample rate is
%      provided).
%   8. Formatting of output channel objects to an object array with dimensions: [ Rx , Tx , Freq. ]
%
%   If qd_layout.simpar.show_progress_bars is set to 1, a progress report is provided on the command
%   prompt.
%
% Input:
%   check_parfiles
%   check_parfiles = 0 / 1 (default: 1) Disables (0) or enables (1) the parsing of shortnames and
%   the validity-check for the config-files. This is useful, if you know that the parameters in the
%   files are valid. In this case, this saves execution time.
%
%   overlap
%   The length of the overlapping part relative to the segment length (segments are specified in
%   qd_layout.rx_track. It can have values in between 0 (no overlap) and 1 (ramp along the entire
%   segment).
%
%   algorithm
%   Selects the interpolation algorithm for the tracks and channel coefficients. Optional are
%    * linear - Linear interpolation (default)
%    * cubic  - Shape preserving piecewise cubic interpolation
%
%    Interpolation of the transceiver orientations and channel phases are always done using the
%    SLERP algorithm (spherical linear interpolation). Spherical cubic interpolation is currently
%    not supported.
%
%   init_builder
%   If set to true, the channel builders are initialized but no channels are generated.
%
% Output:
%   h_channel
%   A vector 'qd_channel' objects.
%
%   h_builder
%   A vector of 'qd_builder' objects.
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
    error('QuaDRiGa:qd_layout:get_channels:object_array','set_scenario not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

% Parse input variables
if ~exist( 'check_parfiles' , 'var' ) || isempty( check_parfiles )
    check_parfiles = true;
elseif check_parfiles ~= 0 && check_parfiles ~= 1
    error('QuaDRiGa:qd_layout:get_channels:check_parfiles','"check_parfiles" mus be either 0 or 1.');
end

if exist( 'overlap' , 'var' ) && ~isempty( overlap )
    if ~( isnumeric(overlap) && all(size(overlap) == [1 1]) && isreal(overlap) ...
            && overlap<=1 && overlap>=0 )
        error('??? "overlap" must be scalar, and in between 0 and 1')
    end
else
    overlap = 0.5;
end

% Select the interpolation algorithm
if ~exist( 'algorithm' , 'var' ) || isempty( algorithm )
    algorithm = [];
end

if ~exist( 'init_builder' , 'var' ) || isempty( init_builder )
    init_builder = false;
end

update_rate = h_layout.update_rate;
verbose = h_layout.simpar(1,1).show_progress_bars;
no_rx = h_layout.no_rx;
no_tx = h_layout.no_tx;
n_freq = numel( h_layout.simpar(1,1).center_frequency );
start_time_model_run = clock;
restore_tracks_in_layout = false;

% Check if tracks changed
if ~isempty( h_layout.h_qd_builder_init ) && ~init_builder
    if isempty( h_layout.track_checksum ) ||...
            h_layout.track_checksum ~= checksum( h_layout.rx_track ) + checksum( h_layout.tx_track ) + checksum( h_layout.simpar )
        warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
        h_layout.h_qd_builder_init = [];
        h_layout.builder_index = [];
    end
end

% Initialize channel builders
if isempty( h_layout.h_qd_builder_init ) || init_builder
    
    % Check if all track names are unique
    h_layout.has_unique_track_names;
    
    % Print status message
    if verbose
        if ~strcmp( h_layout.name , 'Layout' )
            disp(['Layout: ',h_layout.name])
        end
        str = num2str( no_rx );
        if no_rx == 1
            str = [ str, ' receiver, ' ];
        else
            str = [ str, ' receivers, ' ];
        end
        str = [ str, num2str( no_tx ) ];
        if no_tx == 1
            str = [ str, ' transmitter, ' ];
        else
            str = [ str, ' transmitters, ' ];
        end
        str = [ str, num2str(numel(h_layout.simpar(1,1).center_frequency)) ];
        if numel(h_layout.simpar(1,1).center_frequency) == 1
            str = [ str, ' frequency (' ];
        else
            str = [ str, ' frequencies (' ];
        end
        str = [ str, sprintf('%1.1f GHz, ', h_layout.simpar(1,1).center_frequency/1e9) ];
        str = [ str(1:end-2) , ')'];
        
        disp(['Starting channel generation using QuaDRiGa v',h_layout.simpar(1,1).version]);
        disp(str);
    end
    
    samling_limit = min( h_layout.simpar(1,1).wavelength / 2 );
    sampling_ok = true;
    if ~isempty( update_rate )
        
        % Copy all tracks in the layout to a temporary variable
        restore_tracks_in_layout = true;
        rx_track = h_layout.rx_track;       % Copy original handles
        tx_track = h_layout.tx_track;
        h_layout.Prx_track = copy( rx_track );
        h_layout.Ptx_track = copy( tx_track );
        
        % Determine maximum speed
        v_max = 0;
        trk = h_layout.rx_track;
        trk( 1, end+1 : end+numel( h_layout.tx_track ) ) = h_layout.tx_track;
        simulation_time = zeros( 1,numel(trk) );                            % Save the simulation time
        for i_trk = 1 : numel(trk)
            if trk(1,i_trk).no_snapshots > 1
                mp = trk(1,i_trk).movement_profile;                         % Get movement profile
                if isempty( mp )
                    error('QuaDRiGa:qd_layout:get_channels:no_mp',...
                        'Each track must have a movement profile');
                else
                    v = ( mp(2,2:end) - mp(2,1:end-1) ) ./ ( mp(1,2:end) - mp(2,1:end-1) );     % Current speed
                    v_max = max( [v, v_max] );                              % Max speed [m/s]
                    simulation_time(1,i_trk) = mp(1,end);
                end
            end
        end
        sr = h_layout.simpar(1,1).samples_per_meter * v_max;                     % SR in samples per second
        
        % If the sample rate is greater than the update rate, it is more economic to generate the
        % channels at the desired update rate and not use the channel interpolation.
        if update_rate > 1/sr
            sr = 1/update_rate;
            use_channel_interpolation = false;
        else
            use_channel_interpolation = true;
        end
        
        % When using dynamic simulations, the channel observation time must be identical for all
        % transceivers. This makes sure that all channels get interpolated to the same number of
        % snapshots in the next step. Here, we check this criteria.
        if h_layout.dual_mobility
            tmp = mean(simulation_time( simulation_time>0 ));
            tmp(2:3) = tmp(1) + [-0.5 0.5]./sr;
            if ~all( simulation_time == 0 | ( simulation_time>tmp(2) & simulation_time<tmp(3) ) )
                error('QuaDRiGa:qd_layout:get_channels:incorrect_simulation_time',...
                    'Maximum simulation time must be the same for all tracks in dual-mobility mode.');
            end
            simulation_time = tmp(1);
            if verbose
                fprintf(['Channel observation time: ',num2str(simulation_time),' seconds\n' ]);
            end
        end
        
        % Interpolate tracks to sampling rate limit only if channel update rate is given
        if verbose
            os = update_rate * sr;
            fprintf(['Interpolating tracks (v = ',...
                num2str(v_max),' m/s, SR = ',num2str(sr),' samples/s, update factor = '...
                ,num2str( os,'%1.3f' ),')\n' ]);
        end
        sr = 1./sr;                                                         % Update rate in [s]
        
        % Interpolate tracks (use handles to update objects in the layout)
        no_snap_per_trk = zeros( 1,numel(trk) );
        for i_trk = 1 : numel(trk)
            if trk(1,i_trk).no_snapshots > 1
                if get_length( trk(1,i_trk) ) == 0
                    trk(1,i_trk).interpolate('snapshot',sr,[],algorithm,true);
                else
                    trk(1,i_trk).interpolate('time',sr,[],algorithm,true);
                end
            end
            no_snap_per_trk(1,i_trk) = trk(1,i_trk).no_snapshots;
        end
        
        % Interpolate static Rx tracks
        for i_rx = 1 : h_layout.no_rx
            if h_layout.rx_track(1,i_rx).no_snapshots == 1
                movement_profile = [ 0 simulation_time; 1 1 ];
                h_layout.rx_track(1,i_rx).interpolate('snapshot',sr,movement_profile,[],true);
            end
        end
        
        % Verify that all tracks have the same number of snapshots
        max_no_snap_per_trk = max( no_snap_per_trk );
        if h_layout.dual_mobility && ~all( no_snap_per_trk == 1 | no_snap_per_trk == max_no_snap_per_trk )
            error('QuaDRiGa:qd_layout:get_channels:incorrect_snapshots',...
                'Tracks must have the same number of snapshots. Check speed and track length.');
        end
        
    else % No input sampling rate is given
        use_channel_interpolation = false;
    end
    
    % Check if tracks fulfill the sampling theoreme
    for i_rx = 1 : no_rx
        if h_layout.rx_track(1,i_rx).no_snapshots > 1
            [~,dist] = get_length( h_layout.rx_track(1,i_rx) );
            if any( diff(dist) > samling_limit )
                sampling_ok = false;
            end
        end
    end
    if h_layout.dual_mobility
        for i_tx = 1 : no_tx
            if h_layout.tx_track(1,i_tx).no_snapshots > 1
                [~,dist] = get_length( h_layout.tx_track(1,i_tx) );
                if any( diff(dist) > samling_limit )
                    sampling_ok = false;
                end
            end
        end
    end
    if ~sampling_ok
        warning('QuaDRiGa:layout:get_channels:sampling_ok',...
            'Sample density in tracks does not fulfill the sampling theoreme.');
    end
    
    % Create builder objects
    if verbose
        fprintf('Generating channel builder objects')
    end
    if use_channel_interpolation
        h_layout.update_rate = [];
        h_builder = h_layout.init_builder( check_parfiles );
        h_layout.update_rate = update_rate;
    else
        h_builder = h_layout.init_builder( check_parfiles );
    end
    if verbose
        if numel( h_builder ) == 1
            fprintf(' - 1 builder, ')
        else
            fprintf([' - ',num2str(numel( h_builder )),' builders, '])
        end
        cnt = 0;
        sic = size( h_builder );
        for i_cb = 1 : numel(h_builder)
            [ i1,i2 ] = qf.qind2sub( sic, i_cb );
            cnt = cnt + h_builder(i1,i2).no_rx_positions;
        end
        cnt = cnt * numel(h_layout.simpar(1,1).center_frequency);
        if cnt == 1
            fprintf('1 channel segment\n')
        else
            fprintf([num2str(cnt),' channel segments\n'])
        end
    end
    
    % Initializing random generators
    if verbose
        disp('Initializing random generators')
    end
    init_sos( h_builder);
    
    % Split builder object for multi-frequency simulations
    if n_freq > 1 && h_layout.simpar(1,1).use_3GPP_baseline
        h_builder = split_multi_freq( h_builder );    % Split the builders for multiple frequencies
    end
    
    % Save pre-initialized builders to hidden layout variable
    h_layout.h_qd_builder_init = h_builder;
    h_layout.use_channel_interpolation = use_channel_interpolation;
    
    % Calculate a checksum for tracks and simulation parameters to detect changes in the layout
    if use_channel_interpolation || isempty( update_rate )
        h_layout.track_checksum = checksum( h_layout.rx_track ) + checksum( h_layout.tx_track ) + checksum( h_layout.simpar );
    else
        h_layout.track_checksum = checksum( rx_track ) + checksum( tx_track ) + checksum( h_layout.simpar );
    end
    
else % Use existing channel builder objects
    h_builder = h_layout.h_qd_builder_init;
    use_channel_interpolation = h_layout.use_channel_interpolation;
    if isempty( use_channel_interpolation ) || ~islogical( use_channel_interpolation )
        error('QuaDRiGa:qd_layout:get_channels:use_channel_interpolation',...
                'The variable "use_channel_interpolation" was not set correctly.');
    end
end

if init_builder
    % If we only initialize the builders, stop here and return empty channel 
    h_channel = [];
    
else % Generate all channel objects
    
    % Generate SSF parameters
    if verbose
        disp('Generating parameters')
    end
    gen_parameters( h_builder, 5 );
    
    % Split builder object for multi-frequency simulations
    if n_freq > 1 && ~h_layout.simpar(1,1).use_3GPP_baseline
        if verbose
            fprintf('Preparing multi-frequency simulations')
        end
        h_builder = split_multi_freq( h_builder );
        if verbose
            fprintf([' - ',num2str(numel( h_builder )),' builders\n'])
        end
    end
    
    % Display some warning of mobility is not supported by 3GPP baseline model
    if h_layout.simpar(1,1).use_3GPP_baseline
        if h_layout.dual_mobility
            warning('QuaDRiGa:qd_layout:get_channels:dual_mobility',...
                'Dual-mobility simulations are not supported by 3GPP baseline model. Results will be incorrect.');
        end
        single_mobility = false;
        for i_track = 1 : h_layout.no_rx
            if h_layout.rx_track(1,i_track).no_segments > 1
                single_mobility = true;
            elseif h_layout.rx_track(1,i_track).no_snapshots > 1 && get_length( h_layout.rx_track(1,i_track) ) > 5
                single_mobility = true;
            end
        end
        if single_mobility
            warning('QuaDRiGa:qd_layout:get_channels:dual_mobility',...
                'Segments and mobility beyond 5 m are not supported by 3GPP baseline model. Results will be incorrect.');
        end
    end
    
    % Generate channel coefficients
    h_channel = get_channels( h_builder );
    
    % Merge channel coefficients
    h_channel = merge( h_channel, overlap, verbose);
    
    % Get names
    n_channel = numel(h_channel);
    names = {};
    for i_channel = 1:n_channel
        names{i_channel} = h_channel(1,i_channel).name;
        ind = regexp( names{i_channel},'_','once' );
        if ~isempty( ind )
            names{i_channel} = names{i_channel}(ind+1:end);
        end
    end
    
    if use_channel_interpolation
        tStart = clock;
        if verbose; fprintf('Interpolate  ['); end; m0=0;
        
        % Apply speed profile, if provided
        channels_done = false(1,n_channel);
        for i_trk = 1 : no_rx
            if verbose; m1=ceil(i_trk/no_rx*50); if m1>m0; for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
            
            trk = h_layout.rx_track(1,i_trk);                               % Copy handle
            if get_length( trk ) == 0                                       % Get the distance vector
                dist = trk.interpolate( 'snapshot', update_rate, [], algorithm );
            else
                dist = trk.interpolate( 'time', update_rate, [], algorithm );
            end
            
            for i_channel = 1 : n_channel
                if ~channels_done( i_channel )
                    if ~isempty( regexp( names{i_channel} ,  trk.name, 'once' ) )
                        h_channel(1,i_channel) = interpolate( h_channel(1,i_channel), dist, algorithm );
                        
                        par_tmp = h_channel(1,i_channel).par;
                        par_tmp.update_rate = update_rate;   % Store update rate
                        h_channel(1,i_channel).par = par_tmp;
                        
                        channels_done( i_channel ) = true;
                    end
                end
            end
        end
        if verbose
            fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
        end
    end
    
    % Reshape the channel object to the form [ Rx , Tx , Freq ]
    if verbose
        fprintf('Formatting output channels')
    end
    if numel(h_channel) == no_rx * no_tx * n_freq
        % Reorder the channels such that they match the order in the layout
        h_channel = qf.reshapeo( h_channel, [ no_rx, no_tx, n_freq ] );
        
        rx_name = cell(1,size( h_channel,1 ));                  % Get the correct RX order
        for ir = 1 : size( h_channel,1 )
            ii = regexp( h_channel(ir,1,1).name,'_' );
            rx_name{ir} = h_channel(ir,1,1).name(ii(1)+1:end);
        end
        rx_order = zeros( 1,h_layout.no_rx );
        for ir = 1 : h_layout.no_rx
            rx_order( ir ) = find( strcmp( h_layout.rx_track(1,ir).name, rx_name ) );
        end
        
        tx_name = cell(1,size( h_channel,2 ));                  % Get the correct TX order
        for it = 1 : size( h_channel,2 )
            ii = regexp( h_channel(1,it,1).name,'_' );
            if n_freq == 1
                tx_name{it} = h_channel(1,it).name(1:ii(1)-1);
            else
                tx_name{it} = h_channel(1,it,1).name(5:ii(1)-1);
            end
        end
        tx_order = zeros( 1,h_layout.no_tx );
        for it = 1 : h_layout.no_tx
            tx_order( it ) = find( strcmp( h_layout.tx_track(1,it).name, tx_name ) );
        end
                    
        h_channel = h_channel( rx_order, tx_order, 1:n_freq );
    end
    
    if verbose
        if numel( h_channel ) == 1
            fprintf(' - 1 channel object\n')
        else
            fprintf([' - ',num2str(numel( h_channel )),' channel objects\n'])
        end
    end
    
    if verbose
        disp(['Total runtime: ', num2str(round( etime(clock, start_time_model_run))),' seconds']);
    end
end

% Enable warnings
warning('on','QuaDRiGa:qd_builder:use_baseline_ground_reflection');
warning('on','QuaDRiGa:qd_builder:gen_parameters:existing');

% Restore the original tracks in the layout
if restore_tracks_in_layout && ~use_channel_interpolation
    h_layout.Prx_track = rx_track;
    h_layout.Ptx_track = tx_track;
end

end
