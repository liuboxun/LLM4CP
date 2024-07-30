function h_channel = get_channels( h_builder, vb_dots, use_gpu, only_coeff )
%GET_CHANNELS Calculate the channel coefficients
%
% Calling object:
%   Object array
%
% Output:
%   h_channel
%   A vector of qd_channel objects
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

if ~exist( 'vb_dots','var' ) || isempty( vb_dots )
    vb_dots = [];
end

if ~exist( 'use_gpu','var' ) || isempty( use_gpu )
    use_gpu = qd_simulation_parameters.has_gpu;
elseif logical( use_gpu ) && ~qd_simulation_parameters.has_gpu
    use_gpu = 0;
end

if ~exist( 'only_coeff','var' ) || isempty( only_coeff )
    only_coeff = 0;
elseif numel(h_builder) > 1
    error('QuaDRiGa:qd_builder:get_channels','Coefficient only output does not support dq_builder arrays.');
end

% Array indexing is needed for Octave
verbose = h_builder(1,1,1,1).simpar(1,1).show_progress_bars;
if verbose && isempty( vb_dots )
    fprintf('Channels     [');
    vb_dots = 50;
    tStart = clock;
    show_progress = true;
else
    show_progress = false;
end
m0=0;

if numel(h_builder) > 1
    
    % Equally distribute the dots in the progress bar
    sic = size( h_builder );
    vb_dots = zeros( 1,numel(h_builder) );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        if verbose
            vb_dots(i_cb) = h_builder(i1,i2,i3,i4).no_rx_positions;
        else
            % Workaround for Octave 4
            if numel( sic ) == 4
                h_builder(i1,i2,i3,i4).simpar(1,1).show_progress_bars = false;
            elseif numel( sic ) == 3
                h_builder(i1,i2,i3).simpar(1,1).show_progress_bars = false;
            else % 2 and 1
                h_builder(i1,i2).simpar(1,1).show_progress_bars = false;
            end
        end
    end
    if verbose
        vb_dots = init_progress_dots(vb_dots);
    end
    
    % Call each builder in the builder array and concatenate the output channels
    cnt = 1;
    h_channel = qd_channel;
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        if h_builder( i1,i2,i3,i4 ).no_rx_positions > 0
            tmp = h_builder( i1,i2,i3,i4 ).get_channels( vb_dots(i_cb), use_gpu );
            h_channel( 1, cnt : cnt+size(tmp,2)-1 ) = tmp;
            cnt = cnt + size(tmp,2);
        end
    end
    
else
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Check if we have a single-frequency builder
    if numel( h_builder.simpar(1,1).center_frequency ) > 1
        error('QuaDRiGa:qd_builder:get_channels','get_channels only works for single-frequency simulations.');
    end
    
    % Check if SSF parameters have been generated already
    if isempty( h_builder.taus )
        error('QuaDRiGa:qd_builder:get_channels','Small-scale fading parameters have not been generated yet.');
    end
    
    % Check if the builder is a dual-mobility builder and that the inputs are correctly formatted
    dual_mobility = h_builder.dual_mobility;
    if dual_mobility == -1
        h_builder.check_dual_mobility;
    end
    
    % Set initial parameters
    n_mobiles = h_builder.no_rx_positions;
    
    % These variables are often needed. Pre-computing them saves a lot of time
    if only_coeff
        if any( h_builder.NumSubPaths > 1 )
            error('QuaDRiGa:qd_builder:get_channels','Sub-paths are not supported in coefficients-only mode.');
        end
        use_3GPP_baseline = false;
        use_ground_reflection = false;
    else
        use_3GPP_baseline = h_builder.simpar(1,1).use_3GPP_baseline; % logical
        use_ground_reflection = h_builder.check_los > 1.5; % logical
        if use_3GPP_baseline && use_ground_reflection
            % For 3GPP-Baseline, GR is just another path. No need for GR-drifting
            use_ground_reflection = false;
        end
    end
    wave_no = 2*pi/h_builder.simpar(1,1).wavelength;
    
    % If Laplacian PAS is used, the intra-cluster angles are increased by a factor of sqrt(2). To
    % compensate, the intra-cluster powers must be adjusted. This is done by weighting the path
    % amplitudes, depending on the number of subpaths per cluster. The weights are set here.
    if ~only_coeff && strcmp( h_builder.scenpar.SubpathMethod, 'Laplacian' )
        use_laplacian_pas = true;
        laplacian_weights = {1, [1.18 0.78], ...
            [0.60 1.05 1.24], ...
            [0.86 0.40 1.71 0.42], ...
            [1.05 0.45 0.85 0.85 1.50], ...
            [0.65 0.59 1.87 0.39 1.17 0.46], ...
            [0.77 0.75 1.06 0.61 1.02 0.53 1.74], ...
            [0.74 0.79 1.07 0.51 0.94 0.57 1.52 1.38], ...
            [0.96 0.62 0.92 0.75 1.08 0.51 1.71 1.24 0.63], ...
            [0.90 0.71 0.96 0.78 1.04 0.51 1.52 1.14 0.57 1.37], ...
            [0.92 0.66 0.89 0.79 1.03 0.52 1.47 1.13 0.60 1.36 1.15], ...
            [0.84 0.71 0.91 0.69 1.03 0.54 1.53 1.17 0.50 1.37 1.13 1.01], ...
            [0.79 0.67 1.01 0.61 0.96 0.49 1.61 1.24 0.54 1.33 1.08 1.13 0.86], ...
            [0.98 0.71 0.91 0.84 1.15 0.53 1.45 1.12 0.71 1.43 1.25 1.01 0.83 0.47], ...
            [0.99 0.70 0.89 0.83 1.09 0.57 1.32 1.11 0.64 1.37 1.16 1.01 0.80 0.53 1.41], ...
            [0.96 0.75 0.93 0.82 1.02 0.59 1.35 1.07 0.70 1.30 1.16 0.99 0.82 0.45 1.36 1.18], ...
            [0.95 0.70 0.92 0.80 1.04 0.58 1.28 1.07 0.71 1.28 1.10 1.01 0.79 0.50 1.32 1.17 1.25], ...
            [0.89 0.83 0.97 0.82 1.02 0.67 1.30 1.16 0.69 1.17 1.05 1.10 0.89 0.58 1.26 1.25 1.29 0.53], ...
            [0.91 0.79 1.01 0.85 0.99 0.66 1.29 1.14 0.70 1.20 1.04 1.03 0.92 0.56 1.23 1.21 1.27 0.53 1.15], ...
            [0.89 0.79 0.98 0.83 0.97 0.71 1.27 1.11 0.67 1.15 1.08 1.05 0.89 0.50 1.21 1.24 1.27 0.59 1.13 1.15]};
    else
        use_laplacian_pas = false;
    end
    
    % Create new channel object
    if only_coeff
        h_channel = zeros(h_builder.rx_array(1,1).no_elements, ...
            h_builder.tx_array(1,1).no_elements, ...
            h_builder.NumClusters, h_builder.no_rx_positions, 'single');
    else
        h_channel = qd_channel;
    end
    
    % The loop for each user position
    for i_mobile = 1 : n_mobiles
        if verbose
            m1=ceil(i_mobile/n_mobiles*vb_dots);
            if m1>m0
                for m2=1:m1-m0
                    fprintf('o');
                end
                m0=m1;
            end
        end
        
        % Get the list of zero-power paths - we do not return paths with zero-power
        iClst       = h_builder.gain(i_mobile,:) ~= 0;
        iClst(1)    = true;
        if use_ground_reflection
            iClst(2) = true;
        end
        iPath       = clst_expand( iClst, h_builder.NumSubPaths );
        n_clusters  = sum( iClst );
        o_clusters  = ones(1,n_clusters);
        n_paths     = sum( iPath );
        n_subpaths  = h_builder.NumSubPaths( iClst );
        n_rxant     = h_builder.rx_array(1,i_mobile).no_elements;
        n_txant     = h_builder.tx_array(1,i_mobile).no_elements;
        o_txrxant    = ones(1,n_txant*n_rxant);
        
        % Check if the positions are correct
        if any( abs( h_builder.rx_track(1,i_mobile).initial_position - h_builder.rx_positions(:,i_mobile) ) > 1e-5 )
            if i_mobile > 1 && ~qf.eqo( h_builder.rx_track(1,i_mobile), h_builder.rx_track(1,1) )
                warning('QuaDRiGa:qd_builder:get_channels',...
                    'Rx position in track does not match the initial position, using initial position.');
            end
            h_builder.rx_track(1,i_mobile).initial_position = h_builder.rx_positions(:,i_mobile);
        end
        if any( abs( h_builder.tx_track(1,i_mobile).initial_position - h_builder.tx_position(:,i_mobile) ) > 1e-5 )
            if i_mobile > 1 && ~qf.eqo( h_builder.tx_track(1,i_mobile), h_builder.tx_track(1,1) )
                warning('QuaDRiGa:qd_builder:get_channels',...
                    'Tx position in track does not match the initial position, using initial position.');
            end
            h_builder.tx_track(1,i_mobile).initial_position = h_builder.tx_position(:,i_mobile);
        end
        
        % Read some commonly needed variables in order to save time.
        n_links     = n_rxant*n_txant;
        o_links     = ones(1,n_links,'uint8');
        n_snapshots = h_builder.rx_track(1,i_mobile).no_snapshots;
        o_snapshots = ones(1,n_snapshots,'uint8');
        initial_pos = h_builder.rx_track(1,i_mobile).segment_index( min( [h_builder.rx_track(1,i_mobile).no_segments,2] ));
        
        % Extract the random initial phases
        pin = h_builder.pin(i_mobile,iPath); % double
        
        if use_3GPP_baseline
            % If we do not use drifting and have a linear track, then the Doppler component is only
            % dependent on the rotating phases of the taps. So, we do not recalculate the antenna
            % response for each snapshot.
            
            % Get the angles of the subpaths and perform random coupling.
            [ aod,eod,aoa,eoa,delay ] = get_subpath_angles( h_builder, i_mobile, use_laplacian_pas );
            aod = aod(:,iPath);
            eod = eod(:,iPath);
            aoa = aoa(:,iPath);
            eoa = eoa(:,iPath);
            delay = delay(:,iClst);
            
            % Calculate the distance-dependent phases
            lambda  = h_builder.simpar(1,1).wavelength;
            if h_builder.simpar(1,1).use_absolute_delays
                d_lms   = h_builder.simpar(1,1).speed_of_light * delay;
            else
                r       = h_builder.rx_positions(:,i_mobile) - h_builder.tx_position(:,i_mobile);
                norm_r  = sqrt(sum(r.^2)).';
                d_lms   = norm_r + h_builder.simpar(1,1).speed_of_light * delay;
            end
            phase   = 2*pi/lambda * mod(d_lms, lambda);
            phase   = clst_expand( phase, n_subpaths );
            
            % Doppler component
            % Without drifting, the Doppler component is calculated by plane wave approximation
            % using the distance from the initial position.
            tmp = h_builder.rx_track(1,i_mobile).positions;
            dist_rx = sqrt( sum([ tmp(1,:) - tmp(1,1) ; tmp(2,:) - tmp(2,1) ; tmp(3,:) - tmp(3,1)   ].^2 ) );
            no_snap_process = 1;       % Only process the first snapshot, everything else is approximated
            update_tx_ant = true;      % Interpolate Tx array response
            
        elseif only_coeff
            % Initialize the path delays
            aod = h_builder.AoD(i_mobile,iClst);
            eod = h_builder.EoD(i_mobile,iClst);
            aoa = h_builder.AoA(i_mobile,iClst);
            eoa = h_builder.EoA(i_mobile,iClst);
            delay = h_builder.taus(i_mobile,iClst);
            
            lambda  = h_builder.simpar(1,1).wavelength;
            d_lms   = h_builder.simpar(1,1).speed_of_light * delay;
            phase   = 2*pi/lambda * mod(d_lms, lambda);
            
            no_snap_process = 1;
            
        else
            % Calculate the scatterer positions
            lbs_pos = h_builder.lbs_pos(:,iPath,i_mobile);
            fbs_pos = h_builder.fbs_pos(:,iPath,i_mobile);
            
            % Initialize the path delays
            delay = zeros( n_snapshots, n_clusters, n_rxant, n_txant );
            no_snap_process = n_snapshots;
        end
        
        % Travel directions
        rx_orientation = h_builder.rx_track(1,i_mobile).orientation;
        tx_orientation = h_builder.tx_track(1,i_mobile).orientation;
        if size( tx_orientation,2 ) == 1
            tx_orientation = tx_orientation(:,o_snapshots);
        end
        
        % Placeholder for the coefficient calculation
        cn    = zeros( n_links , n_clusters , n_snapshots );
        
        % Placeholder for the radiated power
        ppat  = zeros( n_links , n_clusters , n_snapshots );
        
        if use_gpu == 2 % Single precision GPU Acceleration
            gM = single( h_builder.xprmat(:,iPath,i_mobile) );     	% The NLOS polarization transfer matrix
            gM(1) = gM(1) + 1j*1e-45;                               % Make sure it is complex-valued
            
        else % Double precision (CPU or GPU)
            gM = h_builder.xprmat(:,iPath,i_mobile);               	% The NLOS polarization transfer matrix
            gM(1) = gM(1) + 1j*4e-324;                              % Make sure it is complex-valued
        end
        gM = permute( gM,[3,4,2,1] );                               % Convert to [ 1 x 1 x n_paths x 4 ]
        
        % Transfer to GPU
        if use_gpu
            try
                gM = gpuArray( gM );
            catch
                use_gpu = false;
            end
        end
        
        % Do for each snapshot
        for i_snapshot = 1 : no_snap_process          % Track positions
            
            % Interpolate the antenna patterns for the NLOS paths
            if use_3GPP_baseline || only_coeff % Planar waves
                
                % Interpolate the receive antenna patterns
                [ gVr, gHr, Pr, aoa, eoa ] = interpolate( h_builder.rx_array(1,i_mobile), aoa, eoa, [], ...
                    rx_orientation(:,i_snapshot), 14.3239, use_gpu(use_gpu~=0)+2 );
                gVr = reshape( gVr, n_rxant,1,n_paths );
                gHr = reshape( gHr, n_rxant,1,n_paths );
                Pr = reshape( Pr, n_rxant,1,n_paths );
                
                % Interpolate the transmit antenna patterns
                [ gVt, gHt, Pt ] = interpolate( h_builder.tx_array(1,i_mobile), aod, eod, [], ...
                    tx_orientation(:,i_snapshot), 14.3239, use_gpu(use_gpu~=0)+2 );
                gVt = reshape( gVt, 1,n_txant,n_paths );
                gHt = reshape( gHt, 1,n_txant,n_paths );
                Pt = reshape( Pt, 1,n_txant,n_paths );
                
                % Calculate the Doppler profile.
                doppler = cos(aoa+pi).*cos(eoa);
                
            else
                if i_snapshot == 1 % Initialize drifting
                    [ aod, eod, aoa, eoa, phase, delay(i_snapshot,:,:,:),...
                        aod_los, eod_los, aoa_los, eoa_los, theta_r, update_tx_ant ] =...
                        update_drifting( h_builder(1,1), i_snapshot, i_mobile, fbs_pos, lbs_pos );
                    
                else % Update drifting
                    [ aod, eod, aoa, eoa, phase, delay(i_snapshot,:,:,:),...
                        aod_los, eod_los, aoa_los, eoa_los, theta_r, update_tx_ant ] =...
                        update_drifting( h_builder(1,1), i_snapshot );
                end
                aod = permute(aod,[4,2,1,3]);
                eod = permute(eod,[4,2,1,3]);
                aoa = permute(aoa,[3,2,1,4]);
                eoa = permute(eoa,[3,2,1,4]);
                
                % Interpolate the receive antenna patterns
                [ gVr, gHr ] = interpolate( h_builder.rx_array(1,i_mobile), aoa, eoa, [], ...
                    rx_orientation(:,i_snapshot), 14.3239, use_gpu(use_gpu~=0)+2 );
                gVr = reshape( gVr, n_rxant,1,n_paths );
                gHr = reshape( gHr, n_rxant,1,n_paths );
                
                % Interpolate the transmit antenna patterns only when needed
                if update_tx_ant
                    [ gVt, gHt ] = interpolate( h_builder.tx_array(1,i_mobile), aod, eod, [], ...
                        tx_orientation(:,i_snapshot), 14.3239, use_gpu(use_gpu~=0)+2 );
                    gVt = reshape( gVt, 1,n_txant,n_paths );
                    gHt = reshape( gHt, 1,n_txant,n_paths );
                end
            end
            
            % Calculate the NLOS channel coefficients
            gG = repmat(gVr,[1,n_txant,1]) .* repmat( gM(:,:,:,1),[n_rxant,n_txant,1] ) .* repmat(gVt,[n_rxant,1,1]) + ...
                repmat(gVr,[1,n_txant,1]) .* repmat( gM(:,:,:,3),[n_rxant,n_txant,1] ) .* repmat(gHt,[n_rxant,1,1]) + ...
                repmat(gHr,[1,n_txant,1]) .* repmat( gM(:,:,:,2),[n_rxant,n_txant,1] ) .* repmat(gVt,[n_rxant,1,1]) + ...
                repmat(gHr,[1,n_txant,1]) .* repmat( gM(:,:,:,4),[n_rxant,n_txant,1] ) .* repmat(gHt,[n_rxant,1,1]);
            
            if ~use_3GPP_baseline && ~only_coeff
                
                % Calculate the RX antenna response for the LOS and GR path
                aoa_los = reshape( permute( aoa_los, [3,4,2,1] ) ,n_rxant,[] );
                eoa_los = reshape( permute( eoa_los, [3,4,2,1] ) ,n_rxant,[] );
                [ gVLr,gHLr ] = interpolate( h_builder.rx_array(1,i_mobile), aoa_los, eoa_los, [], ...
                    rx_orientation(:,i_snapshot), 14.3239, use_gpu(use_gpu~=0)+2 );
                gVLr = reshape( gVLr, n_rxant, n_txant, [] );
                gHLr = reshape( gHLr, n_rxant, n_txant, [] );
                
                % Calculate the TX antenna response for the LOS and GR path
                aod_los = reshape( permute( aod_los, [4,3,2,1] ) ,n_txant,[] );
                eod_los = reshape( permute( eod_los, [4,3,2,1] ) ,n_txant,[] );
                [ gVLt,gHLt ] = interpolate( h_builder.tx_array(1,i_mobile), aod_los, eod_los, [], ...
                    tx_orientation(:,i_snapshot), 14.3239, use_gpu(use_gpu~=0)+2 );
                if use_ground_reflection
                    gVLt = reshape( gVLt, n_txant, n_rxant, [] );   % Warning: wrong order!
                    gHLt = reshape( gHLt, n_txant, n_rxant, [] );   % Warning: wrong order!
                else
                    gVLt = gVLt.';
                    gHLt = gHLt.';
                end
                
                % Calculate the additional polarization scaling factors for the ground reflection
                if use_ground_reflection
                    epsilon_r = h_builder.gr_epsilon_r( i_mobile );   % Relative permittivity
                    
                    Z      = sqrt( epsilon_r - (cos(theta_r)).^2 );
                    R_par  = (epsilon_r * sin(theta_r) - Z) ./ (epsilon_r * sin(theta_r) + Z);
                    R_per  = ( sin(theta_r) - Z) ./ ( sin(theta_r) + Z);
                    
                    % Read the path power scaling that was used in "generate_initial_paths.m"
                    P_LOS = h_builder.pow(i_mobile,1);
                    P_GR  = h_builder.pow(i_mobile,2);
                    if P_GR == 0
                        Sl = 1;
                        gSv = 0;
                        gSh = 0;
                    else
                        % Compensate for the power scaling in "generate_initial_paths.m"
                        Rsq   = 2 * P_GR / (P_LOS + P_GR);
                        gSv = sqrt(2/Rsq) * R_par;       % GR path vertical pol.
                        gSh = sqrt(2/Rsq) * R_per;       % GR path horizontal pol.
                        if P_LOS == 0
                            Sl = 0;
                        else
                            Sl = 1 / sqrt( 1-Rsq/2 );   % LOS path
                        end
                    end
                    
                    % Transfer to GPU
                    if use_gpu == 1 % double
                        gSv(1) = gSv(1) + 1j*4e-324; gSv = gpuArray( gSv );
                        gSh(1) = gSh(1) + 1j*4e-324; gSh = gpuArray( gSh );
                    elseif use_gpu == 2 % single
                        gSv(1) = gSv(1) + 1j*1e-45; gSv = gpuArray( single( gSv ) );
                        gSh(1) = gSh(1) + 1j*1e-45; gSh = gpuArray( single( gSh ) );
                    end
                    
                    % Update the LOS channel coefficients (LOS polarization transfer matrix is [ 1 0 ; 0 -1 ])
                    gG(:,:,1) = Sl * gVLr(:,:,1) .*  gVLt(:,:,1).' - Sl * gHLr(:,:,1) .* gHLt(:,:,1).';
                    gG(:,:,2) = gSv .* gVLr(:,:,2) .*  gVLt(:,:,2).' - gSh .* gHLr(:,:,2) .* gHLt(:,:,2).';
                else
                    % Update the LOS channel coefficients (LOS polarization transfer matrix is [ 1 0 ; 0 -1 ])
                    gG(:,:,1) = gVLr(:,:,1) .*  gVLt(:,:,1) - gHLr(:,:,1) .* gHLt(:,:,1);
                end
            end
            
            % Obtain coefficients from GPU
            if use_gpu
                gG = double( gather( gG ) );
            end
            
            % The phases
            if use_3GPP_baseline || only_coeff
                % In drifting mode, we have to update the coefficient matrix with the time-variant
                % Doppler profile.
                ccp = gG .* exp( -1j*( repmat(permute(pin,[1,3,2]),n_rxant,n_txant) + ...
                    wave_no*( repmat(Pt,[n_rxant,1,1]) + repmat(Pr,[1,n_txant,1]) ) + ...
                    repmat(permute(phase(1,:),[1,3,2]),n_rxant,n_txant) ) );
            else
                % The phases already contain the effect of the AoD. Hence, the parallel projection
                % of the array antennas is not needed.
                ccp = gG .* exp(-1j*( repmat(permute(pin,[1,3,2]),n_rxant,n_txant) +  permute(phase,[3,4,2,1] )));
            end
            ccp = reshape( ccp, n_rxant*n_txant,n_paths );
            
            % Sum over the sub-paths in a cluster. This changes the cluster power due to the random
            % phases. This is corrected later.
            if ~only_coeff
                ls = 1;
                for l = 1 : n_clusters
                    le = ls + n_subpaths(l) - 1;
                    if le ~= ls
                        if use_laplacian_pas
                            tmp = ccp(:,ls:le) .* (ones(n_rxant*n_txant,1) * laplacian_weights{n_subpaths(l)});
                            ppat(:,l,i_snapshot) = sum( abs(tmp).^2,2 );
                            cn(:,l,i_snapshot)   = sum( tmp,2 );
                        else
                            ppat(:,l,i_snapshot) = sum( abs(ccp(:,ls:le)).^2,2 );
                            cn(:,l,i_snapshot)   = sum( ccp(:,ls:le),2 );
                        end
                    else
                        ppat(:,l,i_snapshot) = abs(ccp(:,ls)).^2;
                        cn(:,l,i_snapshot)   = ccp(:,ls);
                    end
                    ls = le + 1;
                end
            end
        end
        
        if use_3GPP_baseline
            % Only one snapshot is calculated, the others are
            % emulated by phase rotation.
            
            % Combine pattern and phase for the first snapshot
            %c = c.*cp;
            c = ccp;
            
            for i_snapshot = 2 : n_snapshots
                % Generate rotating Doppler for the successive snapshots
                cp = exp( -1j * wave_no * doppler * dist_rx(i_snapshot) );
                cp = cp( ones(1,n_links) , : );
                
                % Combine antenna patterns and phases
                ccp = c.*cp;
                
                % Sum over the sub-paths in a cluster. This changes the cluster power due to the random
                % phases. This is corrected later.
                ls = 1;
                for l = 1 : n_clusters
                    le = ls + n_subpaths(l) - 1;
                    if le ~= ls
                        if use_laplacian_pas
                            tmp = ccp(:,ls:le) .* (ones(n_rxant*n_txant,1) * laplacian_weights{n_subpaths(l)});
                            ppat(:,l,i_snapshot) = sum( abs(tmp).^2,2 );
                            cn(:,l,i_snapshot)   = sum( tmp,2 );
                        else
                            ppat(:,l,i_snapshot) = sum( abs(ccp(:,ls:le)).^2,2 );
                            cn(:,l,i_snapshot)   = sum( ccp(:,ls:le),2 );
                        end
                    else
                        ppat(:,l,i_snapshot) = abs(ccp(:,ls)).^2;
                        cn(:,l,i_snapshot)   = ccp(:,ls);
                    end
                    ls = le + 1;
                end
            end
        end
        
        if only_coeff
            % The path powers already contain the path gain
            p_cl = sqrt( h_builder.gain(i_mobile,iClst ) );
            cn =  repmat(p_cl,n_links,1) .* ccp;
        else
            
            % The path powers
            p_cl = h_builder.pow(i_mobile*ones(1,n_links),iClst );
            
            % The powers of the antenna patterns at the given angles (power-sum)
            p_pat = sum( ppat,3 ) ./ size(ppat,3);
            
            % The powers in the current channel coefficients (complex sum)
            p_coeff = sum( abs(cn).^2, 3 ) ./ size(cn,3);
            
            % Correct the powers
            p_correct = sqrt( p_cl .* p_pat ./ p_coeff ./ n_subpaths(o_links,:) );
            p_correct( p_pat < 1e-30 ) = 0; % Fix NaN caused by 0/0
            cn = p_correct(:,:,ones(1,n_snapshots)) .* cn;
            
            % Now we apply the K-Factor and the shadowing profile
            if use_3GPP_baseline || isempty( h_builder.sos )
                
                % Get the PL for the initial position only
                [ ~, ~, path_loss , scale_sf ] = h_builder.get_pl( h_builder.rx_track(1,i_mobile),...
                    [],h_builder.tx_track(1,i_mobile) );
                rx_power = -path_loss + 10*log10( h_builder.sf(1,i_mobile) ) .* scale_sf;
                rx_power = sqrt( 10.^( 0.1 * rx_power ) );
                
                % The initial KF is already applied in path powers. Here,
                % we only need to apply the SF and the path loss.
                cn = cn * rx_power;
                
            else
                
                % Calculate the path gain along the track segment
                path_gain = -h_builder.get_pl( h_builder.rx_track(1,i_mobile), [], h_builder.tx_track(1,i_mobile) );
                
                % We have a dynamic SF and KF profile that varies over the positions on the track.
                % Get shadowing profile along the track from the SOS generators.
                [sf,kf] = h_builder.get_sf_profile( h_builder.rx_track(1,i_mobile), h_builder.tx_track(1,i_mobile) );
                
                % When changing the cluster powers (e.g., by "add_paths"), the SF changes as well. We
                % obtain the difference of the SF by reading the initial SF values from the builder and
                % the dynamic SF profile.
                sf_init_builder = h_builder.sf(1,i_mobile);
                sf_init_sos     = sf( initial_pos );
                
                % We now correct the dynamic SF values (linear scale).
                sf = sf .* sf_init_builder/sf_init_sos;
                
                % Calculate the Rx power (sum-power of all clusters)
                rx_power = path_gain + 10*log10( sf );
                rx_power = 10.^( 0.1 * rx_power );
                rx_power = permute( rx_power, [1,3,2] );
                
                % Get the KF scaling
                if kf( initial_pos ) < 1e-10
                    kf_scale = ones( size( kf ) );
                else
                    kf_scale = kf ./ kf( initial_pos );
                end
                kf_scale = permute( kf_scale, [1,3,2] );
                
                % Get the normalized power for the LOS and NLOS components ( p_los + p_nlos = 1 )
                if use_ground_reflection
                    p_los  = h_builder.pow( i_mobile,1 ) + h_builder.pow( i_mobile,2 );
                    p_nlos = sum( h_builder.pow(i_mobile,3:end) );
                else
                    p_los  = h_builder.pow( i_mobile,1 );
                    p_nlos = sum( h_builder.pow(i_mobile,2:end) );
                end
                
                % Adjust the path powers to apply the varying KF along the track segment
                if p_los > 1e-4 && p_nlos > 1e-4 && any(kf_scale(:) ~= 1)
                    
                    % Adjust the power of the LOS component to match the target KF
                    cn(:,1,:) = cn(:,1,:) .* repmat( sqrt(kf_scale),[n_links,1,1] );
                    if use_ground_reflection
                        cn(:,2,:) = cn(:,2,:) .* repmat( sqrt(kf_scale),[n_links,1,1] );
                    end
                    
                    % The power adjustment of the LOS component changes the total RX power
                    % This needs to be compensated in the total RX power
                    rx_power = rx_power ./ ( p_los .* kf_scale + p_nlos );
                end
                
                % Adjust the overall power of the channel coefficients
                rx_power = sqrt( rx_power );
                cn = cn .* rx_power(o_txrxant,o_clusters,:);
            end
        end
        
        % Apply antenna coupling
        Ct = h_builder.tx_array(1,i_mobile).coupling;
        Cr = h_builder.rx_array(1,i_mobile).coupling.';
        
        % Reshape objects
        cn = reshape( cn , n_rxant , n_txant , n_clusters , n_snapshots );
        
        % Apply the antenna coupling
        c = zeros( size(Cr,1) , size(Ct,2) , n_clusters , n_snapshots );
        for i_snapshot = 1:n_snapshots
            for i_cluster = 1:n_clusters
                c(:,:,i_cluster,i_snapshot) = Cr * cn(:,:,i_cluster,i_snapshot) * Ct;
            end
        end
        clear cn
        
        if only_coeff
            h_channel(:,:,iClst,i_mobile) = c;
            
        elseif use_3GPP_baseline
            h_channel(1,i_mobile) = qd_channel( c , delay' , initial_pos );
            
        else
            % When we use high precision, the delays on all elements are different. However, antenna
            % coupling will merge the coefficients of different antennas. This needs to be
            % considered by the delays too. The delays on different elements are weighted by the
            % powers in the coupling matrix.
            Cr_dl = zeros( size( Cr ));
            for i_rx = 1:size( Cr,1 )
                tmp = abs( Cr( i_rx , : ) ).^2;
                Cr_dl( i_rx , : ) = tmp./sum(tmp);
            end
            
            Ct_dl = zeros( size( Ct ));
            for i_tx = 1:size( Ct,2 )
                tmp = abs( Ct( : , i_tx ) );
                Ct_dl( : , i_tx ) = tmp./sum(tmp);
            end
            
            % Here, we scale the delays for each path by the coupling powers.
            delay = permute( delay, [3,4,2,1] );
            dl = zeros( size(Cr,1) , size(Ct,2) , n_clusters , n_snapshots );
            for i_snapshot = 1:n_snapshots
                for i_cluster = 1:n_clusters
                    dl(:,:,i_cluster,i_snapshot) = Cr_dl * delay(:,:,i_cluster,i_snapshot) * Ct_dl;
                end
            end
            h_channel(1,i_mobile) = qd_channel( c , dl , initial_pos );
        end
        clear c
        
        % Store the channel name
        if ~only_coeff
            % This is important because the merger uses the name string to connect the channels.
            channel_name = h_builder.name;
            if isempty( channel_name ) || isempty( regexp( channel_name , '_', 'once' ) )
                channel_name = 'Scen_*';
            end
            % The "*" is added when there are multiple tx positions in the builder
            tmp = regexp( channel_name , '\*' );
            if ~isempty( tmp )
                channel_name = [ channel_name(1:tmp-1), h_builder.tx_track(1,i_mobile).name ];
            end
            channel_name = [ channel_name ,'_', h_builder.rx_track(1,i_mobile).name ]; %#ok
            h_channel(1,i_mobile).name = channel_name;
            h_channel(1,i_mobile).rx_position = h_builder.rx_track(1,i_mobile).positions_abs;
            h_channel(1,i_mobile).tx_position = h_builder.tx_track(1,i_mobile).positions_abs;
            h_channel(1,i_mobile).center_frequency = h_builder.simpar(1,1).center_frequency(1,1);
            
            % Save Additional LSF and SSF information
            clear par_struct
            try
                if use_ground_reflection
                    par_struct.has_ground_reflection = 1;
                end
                par_struct.ds_parset = h_builder.ds( i_mobile ); % [s]
                par_struct.kf_parset = 10*log10( h_builder.kf( i_mobile ) ); % [dB]
                if use_3GPP_baseline
                    par_struct.pg_parset = 10*log10( rx_power.^2 ); % [dB]
                else
                    par_struct.pg_parset = 10*log10( mean(rx_power(:)).^2 ); % [dB]
                    par_struct.pg = 10*log10(abs( reshape( mean(mean(rx_power,1),2) , 1,[] ) ).^2);
                end
                par_struct.sf_parset = 10*log10( h_builder.sf( i_mobile ));
                par_struct.asD_parset = h_builder.asD( i_mobile ); % [deg]
                par_struct.asA_parset = h_builder.asA( i_mobile ); % [deg]
                par_struct.esD_parset = h_builder.esD( i_mobile ); % [deg]
                par_struct.esA_parset = h_builder.esA( i_mobile ); % [deg]
                if ~isempty(h_builder.xpr)
                    par_struct.XPR_parset = 10*log10( h_builder.xpr( i_mobile ) ); % [dB]
                end
                
                % Save the individual per-path values
                par_struct.AoD_cb = h_builder.AoD( i_mobile,iClst ) * 180/pi; % [deg]
                par_struct.AoA_cb = h_builder.AoA( i_mobile,iClst ) * 180/pi; % [deg]
                par_struct.EoD_cb = h_builder.EoD( i_mobile,iClst ) * 180/pi; % [deg]
                par_struct.EoA_cb = h_builder.EoA( i_mobile,iClst ) * 180/pi; % [deg]
                par_struct.pow_cb = h_builder.pow( i_mobile,iClst );          % [W]
                par_struct.gain_cb = h_builder.gain( i_mobile,iClst );          % [W]
                
                % Calculate the spreads at the output of the builder
                par_struct.ds_cb  = qf.calc_delay_spread( h_builder.taus( i_mobile,iClst ), h_builder.pow( i_mobile, iClst ) );
                par_struct.asD_cb = qf.calc_angular_spreads( h_builder.AoD( i_mobile,iClst ), h_builder.pow( i_mobile, iClst ) ) * 180/pi;
                par_struct.asA_cb = qf.calc_angular_spreads( h_builder.AoA( i_mobile,iClst ), h_builder.pow( i_mobile, iClst ) ) * 180/pi;
                par_struct.esD_cb = qf.calc_angular_spreads( h_builder.EoD( i_mobile,iClst ), h_builder.pow( i_mobile, iClst ) ) * 180/pi;
                par_struct.esA_cb = qf.calc_angular_spreads( h_builder.EoA( i_mobile,iClst ), h_builder.pow( i_mobile, iClst ) ) * 180/pi;
                
                if ~use_3GPP_baseline
                    par_struct.NumSubPaths = h_builder.NumSubPaths(1,iClst);
                    par_struct.fbs_pos = fbs_pos;
                    par_struct.lbs_pos = lbs_pos;
                end
                
                % Save update rate
                mp = h_builder.rx_track(1,i_mobile).movement_profile;
                if n_snapshots > 1 && all(size(mp)==[2,2]) && mp(1,1)==0 && mp(2,1)==0 && ...
                        abs(mp(2,2)-get_length(h_builder.rx_track(1,i_mobile))) < 1e-6
                    par_struct.update_rate = mp(1,2)/n_snapshots;
                end
                h_channel(1,i_mobile).par = par_struct;
            end
        end
    end
end

% Fix for octave
if ~only_coeff && numel( h_channel ) == 1
    h_channel = h_channel(1,1);
end

if show_progress
    fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
end

end
