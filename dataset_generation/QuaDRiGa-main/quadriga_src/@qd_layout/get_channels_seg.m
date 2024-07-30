function [ h_channel, h_builder ] = get_channels_seg( h_layout, tx, rx, seg, freq, overlap )
%GET_CHANNELS_SEG Returns the channel coefficients for a single TX-RX link
%
% Calling object:
%   Single object
%
% Description:
%   This method can be used to obtain the channel coefficients for a single TX-RX link. Thus, the
%   channel model can be run in "streaming-mode", where updates are provided on the fly. This can
%   significantly reduce the memory requirements for long time-sequences or large numbers of BSs or
%   MTs. The random-number generators are initialized for the entire simulation setup, ensuring
%   full spatial consistency. Caching is used to avoid multiple calculation of the same overlapping
%   regions, e.g. when calculating segments one by one.  
%
%   Any changes made to the layout after calling "get_channels" or "get_channels_seg" will reset the
%   pre-initialized parameters. In this case, the generated channel coefficient will be different
%   for the same TX-RX link. A warning message will be shown in the command prompt.
%
% Input:
%   tx
%   The index of the transmitter (e.g. the BS)
%
%   rx
%   The index of the receiver, or track (e.g. the MT)
%
%   seg
%   The segment indices on the track. If it is not provided or empty, the entire track is returned.
%   It is also possible to concatenate successive segments, i.e.: [1:3] or [3:5], etc.
%
%   freq
%   The frequency index in case of multi-frequency simulations. If it is not provided or empty,
%   channels are generated for all frequencies defined in "h_layout.simpar.center_frequency"
%
%   overlap
%   The length of the overlapping part relative to the segment length (segments are specified in
%   qd_layout.rx_track. It can have values in between 0 (no overlap) and 1 (ramp along the entire
%   segment).
%
% Output:
%   h_channel
%   The channel for the requested segment. In multi-frequency mode, a vector 'qd_channel' objects
%   is returned.
%
%   h_builder
%   The 'qd_builder' objects for the entire simulation.
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

persistent h_channel_cache cache_index
if isempty( h_channel_cache )
    cache_index = 0;
end

% Parse the input variables
if numel( h_layout ) > 1
    error('QuaDRiGa:qd_layout:get_channels:object_array','set_scenario not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

if ~exist( 'tx' , 'var' ) || isempty( tx ) || numel(tx) > 1
    error('QuaDRiGa:qd_layout:get_channels_seg','??? "tx" must be given as scalar integer number')
end

if ~exist( 'rx' , 'var' ) || isempty( rx ) || numel(rx) > 1
    error('QuaDRiGa:qd_layout:get_channels_seg','??? "rx" must be given as scalar integer number')
end

if ~exist( 'seg' , 'var' ) || isempty( seg )
    seg = 1 : h_layout.rx_track( 1,rx ).no_segments;
elseif numel(seg) > 1 
    seg = seg( : ).';
    if numel(seg) > 1 && ~all( diff(seg) == 1 )
        error('QuaDRiGa:qd_layout:get_channels_seg','??? "seg" must be a consecutive list of segments')
    end
end

n_freq = numel( h_layout.simpar(1,1).center_frequency );
if ~exist( 'freq' , 'var' ) || isempty( freq )
    freq = 1 : n_freq;
else
    freq = sort( freq(:));
end

if exist( 'overlap' , 'var' ) && ~isempty( overlap )
    if ~( isnumeric(overlap) && all(size(overlap) == [1 1]) && isreal(overlap) && overlap<=1 && overlap>=0 )
        error('QuaDRiGa:qd_layout:get_channels_seg',...
            '??? "overlap" must be scalar, and in between 0 and 1')
    end
else
    overlap = 0.5;
end

% Check if tracks changed
if ~isempty( h_layout.h_qd_builder_init )
    if isempty( h_layout.track_checksum ) ||...
            h_layout.track_checksum ~= checksum( h_layout.rx_track ) + checksum( h_layout.tx_track ) + checksum( h_layout.simpar )
        warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
        h_layout.h_qd_builder_init = [];
        h_layout.builder_index = [];
    end
end

% Check if qd_builder objects have been initialized, initialize if needed
builder_index = h_layout.builder_index;
if isempty( h_layout.h_qd_builder_init )
    [ ~, h_builder ] = get_channels( h_layout, 1, overlap, [], true );
    h_builder = split_rx( h_builder );
    h_layout.h_qd_builder_init = h_builder;
    builder_index = [];
    
elseif isempty( builder_index )
    h_builder = split_rx( h_layout.h_qd_builder_init );
    h_layout.h_qd_builder_init = h_builder;
    
else
    h_builder = h_layout.h_qd_builder_init;
end

% Create an index that associates each builder to [rx,tx,seg]
if isempty( builder_index )
    rx_name = h_layout.rx_name;
    tx_name = h_layout.tx_name;
    max_no_segments = max( [ h_layout.rx_track.no_segments ] );
    builder_index = zeros( h_layout.no_rx, h_layout.no_tx, max_no_segments );
    for ib = 1 : numel( h_builder )
        tmp = regexp( h_builder(1,ib).rx_track(1,1).name,'_', 'once' );
        if isempty( tmp )
            ir = strcmp( h_builder(1,ib).rx_track(1,1).name, rx_name );
            is = 1;
        else
            ir = strcmp( h_builder(1,ib).rx_track(1,1).name(1:tmp-1), rx_name );
            is = str2double( h_builder(1,ib).rx_track(1,1).name(tmp+4:end) );
        end
        it = strcmp( h_builder(1,ib).tx_track(1,1).name, tx_name );
        builder_index(ir,it,is) = ib;
    end
    h_layout.builder_index = builder_index;
end

% Check if we want channel interpolation
use_channel_interpolation = h_layout.use_channel_interpolation;
if isempty( use_channel_interpolation ) || ~islogical( use_channel_interpolation )
    error('QuaDRiGa:qd_layout:get_channels_seg:use_channel_interpolation',...
        'The variable "use_channel_interpolation" was not set correctly.');
end

if ~all(all( builder_index(:,:,1) ~= 0 ))
    error('QuaDRiGa:qd_layout:get_channels_seg:incorrect_builder',...
        'The builder initialization is incorrect. All TX-RX pairs must have at least one segment.');
end

% Temporary disable the progress bar
show_progress_bars = h_layout.simpar(1,1).show_progress_bars;
h_layout.simpar(1,1).show_progress_bars = false;

if sum( builder_index(rx,tx,:) ~= 0 ) == 1                              % Only one segment
    
    ib = builder_index(rx,tx,1);                                        % Get builder index
    if isempty( h_builder(1,ib).taus )                                  % Parameters do not exist?
        h_builder(1,ib).gen_parameters;                                 % Generate parameters
    end
    if n_freq > 1
        h_builder_tmp = split_multi_freq( h_builder(1,ib) );            % Split builders for multiple frequencies
    else
        h_builder_tmp = h_builder(1,ib);                                % Copy handle
    end
    h_channel = get_channels( h_builder_tmp(1,freq) );                  % Generate channels
    for ic = 1 : numel( h_channel )
        ii = regexp( h_channel(1,ic).name, '_' );
        h_channel(1,ic).name = h_channel(1,ic).name(ii(1)+1:end);       % Remove scenario string from channel name
    end
    
else    % Multiple segments
    
    % Get the channels for all segments in the list
    for is = 1 : numel( seg )
        ib = builder_index(rx,tx,seg(is));                              % Index of the current segment
        if isempty( h_builder(1,ib).taus )                              % Parameters do not exist?
            h_builder(1,ib).gen_parameters;                             % Generate parameters
        end
        if numel( cache_index ) == numel(freq) + 1 &&...                % The current channel is already cashed?
                all( cache_index == [ ib, freq ] ) 
            if is == 1                                                  % Copy handle
                h_channel_curr = h_channel_cache;
            else
                h_channel_curr(is,:) = h_channel_cache;
            end
        else                                                            % Build channels
            if n_freq > 1                                               % Multiple frequencies ?
                h_builder_tmp = split_multi_freq( h_builder(1,ib) ); 	% Split builders for multiple frequencies
            else
                h_builder_tmp = h_builder(1,ib);                        % Copy handle
            end
            if is == 1                                                  % Generate channels
                h_channel_curr = get_channels( h_builder_tmp(1,freq) ); 
            else
                h_channel_curr(is,:) = get_channels( h_builder_tmp(1,freq) );
            end
        end
    end
    
    % Get the index of the next segment (if needed)
    if seg(is) < size( builder_index,3 ) && builder_index(rx,tx,seg(is)+1) ~= 0
        ib = builder_index(rx,tx,seg(is)+1);
    elseif h_layout.rx_track(1,rx).closed &&...
            ( h_layout.tx_track(1,tx).no_snapshots == 1 || h_layout.tx_track(1,tx).closed )
        ib = builder_index(rx,tx,1);
    else
        ib = 0;
    end
    
    % Get the next channel
    if ib ~= 0
        if isempty( h_builder(1,ib).taus )                              % Parameters do not exist?
            h_builder(1,ib).gen_parameters;                             % Generate parameters
        end
        if n_freq > 1                                                   % Multiple frequencies ?
            h_builder_tmp = split_multi_freq( h_builder(1,ib) );        % Split builders for multiple frequencies
        else
            h_builder_tmp = h_builder(1,ib);                            % Copy handle
        end
        h_channel_next = get_channels( h_builder_tmp(1,freq) );         % Generate channels
        h_channel_cache = h_channel_next;                               % Cache channels
        cache_index = [ ib, freq ];                                     % Store cache index and frequency
    end
    
    % Special case: The next segment is the first one in case of closed tracks. The segment number
    % must be changed in order to get the correct merger behavior.
    if ib == builder_index(rx,tx,1)
        current_segment_number = str2double( h_channel_curr(is,1).name(end-3:end) );
        h_channel_next.name = [ h_channel_next.name(1:end-4),num2str(current_segment_number+1,'%04d') ];
        h_channel_cache = [];                                           % Remove cached channel
        cache_index = 0;
    end
    
    % Append next segment to current channel
    if ib ~= 0
        h_channel_curr(is+1,:) = h_channel_next;
    end
    
    % The first segment must start at the "initial_position == 1", otherwise the merger
    % assumes that the tracks are closed.
    if h_channel_curr(1,1).initial_position ~= 1
        snap_ind = h_channel_curr(1,1).initial_position : h_channel_curr(1,1).no_snap;
        for iF = 1 : numel( freq )
            
            % The current channel might come from the cache. In this case, we must create a copy
            % here to prevent altering the cache.
            if ~isempty( h_channel_cache ) && qf.eqo( h_channel_curr(1,iF), h_channel_cache(1,iF ) )
                h_channel_curr(1,iF) = copy( h_channel_curr(1,iF) );
            end
            
            delay = h_channel_curr(1,iF).delay;
            if size( h_channel_curr(1,iF).tx_position, 2 ) == h_channel_curr(1,iF).no_snap
                h_channel_curr(1,iF).tx_position = h_channel_curr(1,iF).tx_position(:,snap_ind);
            end
            rx_position = h_channel_curr(1,iF).rx_position(:,snap_ind);
            h_channel_curr(1,iF).coeff = h_channel_curr(1,iF).coeff(:,:,:,snap_ind);
            
            par_tmp = h_channel_curr(1,iF).par;
            par_tmp.pg = par_tmp.pg(:,snap_ind,:,:);
            h_channel_curr(1,iF).par = par_tmp;
            
            if h_channel_curr(1,iF).individual_delays
                h_channel_curr(1,iF).delay = delay(:,:,:,snap_ind);
            else
                h_channel_curr(1,iF).delay = delay(:,snap_ind);
            end
            h_channel_curr(1,iF).rx_position = rx_position;
            h_channel_curr(1,iF).initial_position = 1;
        end
    end
    
    % Merge the channels
    h_channel = merge( h_channel_curr, overlap, false );
    
    % Append name string to indicate which sub-segments are returned
    if numel(seg) < h_layout.rx_track( 1,rx ).no_segments
        for iF = 1 : numel( freq )
            h_channel(1,iF).name = [ h_channel(1,iF).name, num2str(seg,'-S%02d') ];
        end
    end
    
    % Get the list of desired snapshots
    if ( ib ~= 0 && use_channel_interpolation ) || ib == builder_index(rx,tx,1)
        snap_ind = 1 : h_channel(1,1).no_snap - h_channel_next(1,1).no_snap + h_channel_next(1,1).initial_position;
    elseif ib ~= 0
        snap_ind = 1 : h_channel(1,1).no_snap - h_channel_next(1,1).no_snap + h_channel_next(1,1).initial_position - 1;
    else
        snap_ind = [];
    end
    
    % Remove unwanted snapshots from next segment
    if ~isempty( snap_ind )
        for iF = 1 : numel( freq )
            delay = h_channel(1,iF).delay;
            if size( h_channel(1,iF).tx_position, 2 ) == h_channel(1,iF).no_snap
                h_channel(1,iF).tx_position = h_channel(1,iF).tx_position(:,snap_ind);
            end
            rx_position = h_channel(1,iF).rx_position(:,snap_ind);
            h_channel(1,iF).coeff = h_channel(1,iF).coeff(:,:,:,snap_ind);
            
            par_tmp = h_channel(1,iF).par;
            par_tmp.pg = par_tmp.pg(:,snap_ind,:,:);
            h_channel(1,iF).par = par_tmp;
            
            if h_channel(1,iF).individual_delays
                h_channel(1,iF).delay = delay(:,:,:,snap_ind);
            else
                h_channel(1,iF).delay = delay(:,snap_ind);
            end
            h_channel(1,iF).rx_position = rx_position;
            h_channel(1,iF).initial_position = 1;
        end
    end
end

% Interpolate channels
if use_channel_interpolation
         
    if get_length( h_layout.rx_track(1,rx) ) == 0
        dist_track = 1:h_layout.rx_track(1,rx).no_snapshots;
        dist_int = h_layout.rx_track(1,rx).interpolate( 'snapshot', h_layout.update_rate );
        
        i_seg = [ h_layout.rx_track(1,rx).segment_index, h_layout.rx_track(1,rx).no_snapshots ];
        i_chan_first = i_seg(seg(1));
        i_chan_last = i_seg(seg(end)+1);
        
    else
        [ ~ , dist_track ] = get_length( h_layout.rx_track(1,rx) );
        dist_int = h_layout.rx_track(1,rx).interpolate( 'time', h_layout.update_rate );
        
        % Find the channel start-position on the track
        oS = ones(1,h_layout.rx_track(1,rx).no_snapshots);
        tmp = h_layout.rx_track(1,rx).positions_abs - h_channel(1,1).rx_position(:,oS);
        i_chan_first = find( sum(abs(tmp).^2,1) < 1e-14, 1 );            	% Channel start index
        
        % Find the channel stop-position on the track
        tmp = h_layout.rx_track(1,rx).positions_abs - h_channel(1,1).rx_position(:,oS*h_channel(1,1).no_snap);
        i_chan_last = find( sum(abs(tmp).^2,1) < 1e-14, 1, 'last' );       	% Channel stop index
    end
    
    % Channel interpolation points relative to channel start
    i_int = dist_int > dist_track(i_chan_first) - 1e-14;                % Always include first point
    if i_chan_last ~= h_layout.rx_track(1,rx).no_snapshots
        i_int = i_int & dist_int < dist_track(i_chan_last) - 1e-14;     % Do not include end point (part of next segment)
    else
        i_int = i_int & dist_int < dist_track(i_chan_last) + 1e-14;     % Include last point only in the last segment
    end
    dist_chan_int = dist_int(i_int) - dist_track( i_chan_first );
    
    if get_length( h_layout.rx_track(1,rx) ) == 0
        dist_chan_int = dist_chan_int + 1;
    end
    
    % Interpolate channels
    for iF = 1 : numel( freq )
        h_channel(1,iF) = interpolate( h_channel(1,iF), dist_chan_int );
        
        par_tmp = h_channel(1,iF).par;
        par_tmp.update_rate = h_layout.update_rate;
        h_channel(1,iF).par = par_tmp;
    end
end

if numel( h_channel ) == 1 % Needed for Octave
    h_channel = h_channel(1,1);
end

% Set progress-bar state to old value
h_layout.simpar(1,1).show_progress_bars = show_progress_bars;

end
