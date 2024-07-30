function c = merge( h_channel, overlap, verbose )
%MERGE Combines channel segments into a continuous time evolution channel.
%
% Calling object:
%   Object array
%
% Description:
%   The channel merger implements the continuous time evolution with smooth transitions between
%   segments. Each segment of a track is split in two parts: an overlapping area with the previous
%   segment and an exclusive part with no overlapping. Each segment is generated independently by
%   the channel builder. However, the distance dependent autocorrelation of the large scale
%   parameters was considered when the parameters were drawn from the corresponding statistics.
% 
%   Transition from segment to segment is carried out by replacing taps of the previous segment by
%   the taps of the current segment, one by one. The modeling of the birth/death process is done as
%   published in the documentation of the WIM2 channel model. The route between adjacent channel
%   segments is split into sub-intervals equal to the minimum number of taps in both overlapping
%   segments. During each sub-interval the power of one old tap ramps down and one new tap ramps up.
%   Power ramps are modeled by a modified sinus function to allow smooth transitions.
% 
%   Taps from the old and new segments are coupled based on their power. If the number of clusters
%   is different in the channel segments, the weakest clusters are ramped up or down without a
%   counterpart from the new/old segment. The merging is only done for the NLOS components since the
%   LOS component has a deterministic behavior. The LOS component is thus just scaled in power.
%
% Input:
%   overlap
%   The length of the overlapping part relative to the segment length. It can have values in
%   between 0 (no overlap) and 1 (ramp along the entire segment).  A value of 0 disables the
%   merging process and the channel segments are simply concatenated. A value of 1 constantly
%   merges the channels. The default setting is 0.5.
%
%   verbose
%   Enables (1, default) or disables (0) the progress bar.
%
% Output:
%   c
%   An array of 'qd_channel' objects containing the merged coefficients and delays.
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

% Parse input arguments
if exist( 'overlap' , 'var' ) && ~isempty( overlap )
    if ~( isnumeric(overlap) && all(size(overlap) == [1 1]) && isreal(overlap) && overlap<=1 && overlap>=0 )
        error('??? "overlap" must be scalar, and in between 0 and 1')
    end
else
    overlap = 0.5;
end

if exist( 'verbose' , 'var' )
    if ~( isnumeric(verbose) || ~isnan(verbose) || all(size(verbose) == [1 1]) )
        error('"verbose" must be numeric.')
    else
        verbose = logical(verbose);
    end
else
    verbose = true;
end

% Get the number of channels
n_channel = numel(h_channel);
if size( h_channel,2 ) ~= n_channel
    h_channel = qf.reshapeo( h_channel, [1,n_channel] );
end

% Show a progress bar if there are more than 10 segments
if n_channel < 3
    verbose = false;
end
if verbose
    fprintf('Merging      [');
    vb_dots = 50;
    tStart = clock;
    m0=0;
end
i_bar = 0;                                  % A counter for the progress bar

% Parses the channel names
[ trk_names, seg_ind_all, order, trk_has_gr, seg_has_gr ] = parse_channel_names( h_channel );
h_channel = h_channel( 1,order );           % Order channels to match the names

% See if we have spherical waves in the channels
individual_delays = h_channel(1,1).individual_delays;

% Do prepare some preliminary data formatting
for i_channel = 1 : n_channel
    h_channel( 1,i_channel ).individual_delays = true;      % Use per-antenna-delays
    
    % The channel object can have an additional field for the position data.
    % However, this field is not mandatory. If it is empty, no processing of the
    % positions will be done.
    if isempty( h_channel( 1,i_channel ).rx_position )      % Always use rx-positions
        h_channel( 1,i_channel ).rx_position = zeros( 3,h_channel( 1,i_channel ).no_snap );
    end
    
    % The channel object can have an additional field for the path loss.
    % However, this field is not mandatory. If it is empty, no processing of the
    % positions will be done.
    if isempty( h_channel(1,i_channel).par ) || ...
            ~isfield( h_channel(1,i_channel).par,'pg' ) || ...
            isempty( h_channel(1,i_channel).par(1).pg )
        if isempty( h_channel(1,i_channel).par )
            h_channel(1,i_channel).par = struct( 'pg', zeros( 1,h_channel( 1,i_channel ).no_snap ) );
        else
            par = h_channel(1,i_channel).par;
            par.pg = zeros( 1,h_channel( 1,i_channel ).no_snap );
            h_channel(1,i_channel).par = par;
        end
    end
end

c = qd_channel;                             % Initialize empty output variable
for i_trk = 1 : numel( trk_names )          % Do for each track
    
    % Find the indices of the channel objects that belong to the current track
    seg_ind = find( seg_ind_all == i_trk );
    n_seg = numel(seg_ind);
    
    if n_seg == 1
        % If there is only one segment in the channel, we do not need to do anything.
        
        % Copy the channel
        c(1,i_trk) = copy( h_channel(1, seg_ind(1) ) );
        c(1,i_trk).name = trk_names{i_trk};
        
        % Update progress bar
        i_bar = i_bar + 1;
        if verbose; m1=ceil(i_bar/n_channel*vb_dots); if m1>m0
                for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
    else
        % Calculate the dimensions of the output channel
        no_txant = h_channel(1, seg_ind(1) ).no_txant;
        no_rxant = h_channel(1, seg_ind(1) ).no_rxant;
        if trk_has_gr( i_trk )
            no_path = max( cat( 1 , h_channel(1, seg_ind ).no_path) ) + 2;
        else
            no_path = max( cat( 1 , h_channel(1, seg_ind ).no_path) ) + 1;
        end
        initial_position = cat( 1 , h_channel(1, seg_ind ).initial_position);
        no_snap = sum( cat( 1 , h_channel(1, seg_ind ).no_snap) - initial_position + 1 );
        if all( initial_position > 1 )
            no_snap = no_snap + 1;          % One extra snapshot for the end-point of closed tracks
            closed = true;
        else
            closed = false;
        end
        
        % Initialize output variables and delays
        coeff = zeros( no_rxant , no_txant , no_path , no_snap );
        delay = zeros( no_rxant , no_txant , no_path , no_snap );
        rx_position = zeros( 3,no_snap );
        pg = zeros( 1,no_snap );
        
        % Processing of the Tx position for dual-mobility
        if size( h_channel(1, seg_ind(1) ).tx_position, 2 ) > 1
            tx_position = zeros( 3,no_snap );
            dual_mobility = true;
        else
            tx_position = h_channel(1, seg_ind(1) ).tx_position;
            dual_mobility = false;
        end
        
        for i_seg = 1 : n_seg           % Do for each segment

            % Update progress bar
            i_bar = i_bar + 1;
            if verbose; m1=ceil(i_bar/n_channel*vb_dots); if m1>m0
                    for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
            
            % Find the channel object that overlaps with the current one
            % "ic" stands for channel index (either channel 1 or 2)
            ic1 = seg_ind(i_seg);                                            % The current channel object
            ic2 = find_overlapping_segment( h_channel, ic1, seg_ind_all );   % The overlapping channel object
            
            % Determine the overlapping area of the two segments and the positions in the output channel.
            % "is" stands for snapshot index (channel 1/2, non-overlapping/overlapping, output )
            [ is1n, is1o, is2o, isn, iso ] = find_overlapping_snapshots( h_channel, ic1, ic2, overlap, seg_ind_all );
            
            % Initialize the path indices for the first segment.
            % "ip" stands for path indices.
            if i_seg == 1
                ip = init_path_indices( h_channel, ic1, seg_ind_all, seg_has_gr );
            end

            % Copy the exclusive data from the current segment to the output channel
            coeff( :,:, ip, isn ) = h_channel( 1,ic1 ).coeff( :,:,:, is1n );
            delay( :,:, ip, isn ) = h_channel( 1,ic1 ).delay( :,:,:, is1n );
            rx_position( :,isn ) = h_channel( 1,ic1 ).rx_position( :,is1n );
            try
                pg( :,isn ) = h_channel( 1,ic1 ).par(1).pg( :,is1n );
            end
            if dual_mobility
                tx_position( :,isn ) = h_channel( 1,ic1 ).tx_position( :,is1n );
            end
            
            % Merge the two segments. This is only necessary, if there is an overlapping segment.
            if ~isempty( ic2 )
                
                % Get the coefficients and delays from the two overlapping segments
                cf1 = h_channel(1,ic1).coeff(:,:,:,is1o);
                dl1 = h_channel(1,ic1).delay(:,:,:,is1o);
                cf2 = h_channel(1,ic2).coeff(:,:,:,is2o);
                dl2 = h_channel(1,ic2).delay(:,:,:,is2o);
                
                [ cf, dl, ip, ramp ] = merge_coeff( cf1, dl1, cf2, dl2, ip, seg_has_gr(ic1), seg_has_gr(ic2), trk_has_gr(i_trk) );
                L = size( cf,3 );
                
                % Write to output
                coeff( :,:,1:L, iso ) = cf;
                delay( :,:,1:L, iso ) = dl;
                              
                % Process rx position
                rx_pos1 = h_channel(1,ic1).rx_position(:,is1o);
                rx_pos2 = h_channel(1,ic2).rx_position(:,is2o);
                rx_position( :,iso ) = rx_pos1 .* ([1;1;1]*(1-ramp)) + rx_pos2 .* ([1;1;1]*ramp);
                
                % Process tx position
                if dual_mobility
                    tx_pos1 = h_channel(1,ic1).tx_position(:,is1o);
                    tx_pos2 = h_channel(1,ic2).tx_position(:,is2o);
                    tx_position( :,iso ) = tx_pos1 .* ([1;1;1]*(1-ramp)) + tx_pos2 .* ([1;1;1]*ramp);
                end
                
                % Process path gain
                try
                    pg1 = h_channel(1,ic1).par(1).pg(:,is1o);
                    pg2 = h_channel(1,ic2).par(1).pg(:,is2o);
                    pg( :,iso ) = pg1 .* (1-ramp) + pg2 .* ramp;
                end
            end
        end
        
        % Copy the exclusive data from the next segment in case of closed tracks
        if closed && ~isempty( ic2 )
            is2n = h_channel( 1,ic2 ).initial_position;
            coeff( :,:, ip, end ) = h_channel( 1,ic2 ).coeff( :,:,:, is2n );
            delay( :,:, ip, end ) = h_channel( 1,ic2 ).delay( :,:,:, is2n );
            rx_position( :,end ) = h_channel( 1,ic2 ).rx_position( :,is2n );
            try
                pg( :,end ) = h_channel( 1,ic2 ).par(1).pg( :,is2n );
            end
            if dual_mobility
                tx_position( :,end ) = h_channel( 1,ic2 ).tx_position( :,is2n );
            end
        end
        
        % There might be some paths that are always 0
        has_power = true( 1, size(coeff,3) );
        for l = 2 : no_path
            if all( coeff(:,:,l,:) == 0 )
                has_power(l) = false;
            end
        end
        
        % Create an output channel array
        c(1,i_trk) = qd_channel( coeff(:,:,has_power,:),delay(:,:,has_power,:) );
        c(1,i_trk).individual_delays = individual_delays;
        c(1,i_trk).name = trk_names{i_trk};
        c(1,i_trk).version = h_channel(1,ic1).version;
        c(1,i_trk).rx_position = rx_position;
        c(1,i_trk).tx_position = tx_position;
        c(1,i_trk).center_frequency = h_channel(1,ic1).center_frequency;
        if any( pg ~= 0 ) || trk_has_gr(i_trk)
            par_tmp = struct;
            if any( pg ~= 0 )
                par_tmp.pg = pg;
            end
            if trk_has_gr(i_trk) % Indicate that the merged track has a ground reflection component
                par_tmp.has_ground_reflection = 1;
            end
            c(1,i_trk).par = par_tmp;
        end
    end
end

% Sort channel by name
n_channel = numel(c);
names = {};
for i_channel = 1:n_channel
    names{i_channel} = c(1,i_channel).name;
end
[~,ind] = sort(names);
c = c(1,ind);

if verbose
    fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
end

end
