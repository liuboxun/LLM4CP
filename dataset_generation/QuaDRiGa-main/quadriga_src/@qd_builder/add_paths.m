function h_bld = add_paths( h_builder, parameters )
%ADD_PATHS Adds multipath components to an existing channel builder
%
% Calling object:
%   Object array
%
% Description:
%   This method can be used to construct a 'qd_builder' object or object array in which the
%   multipath components (MPCs) are a combination of several different scenario configurations.
%   This is achieved by first creating a 'qd_builder' object array which contains the TX and RX
%   positions, trajectories and antenna configurations (e.g., by calling 'qd_layout.init_builder').
%   Then, MPCs can be added to the builder by calling 'add_paths' multiple times, each time
%   providing a different set of propagation 'parameters'. The input variable 'parameters' can have
%   different formats (see below). 
%
%   Depending on the LOS/NLOS/GR state of the calling 'h_builder' and the provided 'parameters', the
%   order in which the paths are added leads to different results as follows:
%
% Effects:
%   1. 'h_builder' has no existing paths New paths are generated as defined by the provided
%   'parameters' and written to 'h_builder'. LSF parameters in 'h_builder' are calculated from the
%   generated paths, but the SOS generators that were used to create the new paths are not written
%   to 'h_builder'.
% 
%   2. 'h_builder' has only NLOS paths (KF < -30 dB) and 'parameters' are for NLOS only Existing
%   paths in 'h_builder' are not changed. New paths are generated and added to 'h_builder'. LSF
%   parameters in 'h_builder' are updated to match the combined channel.
% 
%   3. 'h_builder' has a LOS path (KF > -30 dB) and 'parameters' are for NLOS only Existing paths in
%   'h_builder' are not changed. New paths are generated and added to 'h_builder'. LSF parameters in
%   'h_builder' are updated.
% 
%   4. 'h_builder' has only NLOS paths and 'parameters' are for a LOS scenario Existing NLOS paths
%   in 'h_builder' are not changed. New paths (including a LOS path) are generated and added to
%   'h_builder'. LSF parameters in 'h_builder' are updated. The combined channel will describe a LOS
%   scenario.
% 
%   5. 'h_builder' has a LOS path and 'parameters' are for a LOS scenario Existing NLOS paths in
%   'h_builder' are not changed. New paths (including a LOS path) are generated and added to
%   'h_builder'. The existing LOS path in 'h_builder' will be overwritten by the newly created LOS
%    path. LSF parameters in 'h_builder' are updated. The combined channel will describe a LOS
%   scenario.
% 
%   6. 'parameters' contain a ground-reflection (GR) path An GR path (second cluster in the cluster
%   list) will be added to 'h_builder'. The existing LOS and (optional GR) paths will be overwritten
%   by the newly created LOS and GR paths. LSF parameters in 'h_builder' are updated. The combined
%   channel will describe a LOS+GR scenario.
% 
%   7. 'h_builder' has a LOS and a GR path and 'parameters' are for a LOS scenario only The LOS
%   component will be overwritten by the newly created LOS path, the GR path will be removed. LSF
%   parameters in 'h_builder' are updated. The combined channel will describe a LOS scenario.
%
% Input:
%   parameters
%   Scenario definition for the added paths. This variable can have 4 formats:
%
%   1. Variable is not given or empty (e.g. 'parameters = []')  Uses the scenario definitions from
%   the calling 'h_builder' object or object array. If there are no preexisting MPCs in
%   'h_builder', the output of 'add_paths' will be equivalent to calling 'gen_parameters'. Existing
%   SOS random generators will be reused, missing ones will be created. Existing LSF parameters
%   will be discarded. Preexisting SSF parameters and scatterer positions will not be changed.
%
%   2. String containing the parameter name  Loads the corresponding scenario parameters (e.g.,
%   from a '.conf' file) into a new 'qd_builder' object; copies the TX and RX positions from the
%   calling 'h_builder' object or object array into the new builder; generates SOS generators, LSF
%   and SSF parameters and adds the new paths to 'h_builder'.
%
%   3. A single 'qd_builder' object  Uses the provided 'qd_builder' object to generate the new
%   paths. It is possible to edit the 'scenpar' or 'plpar' properties, pre-initialize the SOS
%   generators or provide specific LSPs. The 'add_paths' method then copies the TX and RX positions
%   from the calling 'h_builder' object or object array into the new builder; generates  SSF
%   parameters and adds the new paths to 'h_builder'.
%
%   4. A 'qd_builder' object array  The TX and RX positions in the 'qd_builder' array must match
%   the positions in the calling 'h_builder' object or object array (e.g., by providing a copy of
%   'h_builder'). It is possible to edit the scenario parameters, SOS generators and LSPs on a per-
%   user basis. Provided SSF parameters will be discarded. The 'add_paths' method then generates
%   SSF parameters and adds the new paths to 'h_builder'.
%
% Output:
%   h_bld
%   The 'qd_builder' object array that was used to generate the new paths.
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


% Check if simulation settings are consistent in "h_builder"
sic = size( h_builder );
if numel( sic ) > 2
    error('QuaDRiGa:qd_builder:add_clusters','Array of qd_builder objects cannot have more that 2 dimensions.');
end
chk = zeros( sic );
for i_cb = 1 : numel(h_builder)
    [ i1,i2 ] = qf.qind2sub( sic, i_cb );
    if i_cb == 1
        chk(1,1) = h_builder(1,1).simpar(1,1).checksum;
    elseif qf.eqo( h_builder(1,1).simpar, h_builder(i1,i2).simpar )
        chk(i1,i2) = chk(1,1);
    else
        chk(i1,i2) = h_builder(i1,i2).simpar(1,1).checksum;
    end
end
if any( chk(:) ~= chk(1) )
    error('QuaDRiGa:qd_builder:add_clusters','All builders must have the same simulations settings.');
end

% Parse "parameters"
if ~exist('parameters','var') || isempty( parameters )
    h_bld = copy( h_builder );
    assemble_tx_rx = false;
    parameters = [];
    
elseif ischar( parameters )
    h_bld = qd_builder( parameters );
    h_bld.simpar = h_builder(1,1).simpar;
    assemble_tx_rx = true;
    
elseif isa( parameters(1,1),'qd_builder' ) && numel( parameters ) == 1
    if any(any( qf.eqo( parameters, h_builder ) ))
        h_bld = copy( parameters );
        parameters = [];
    else
        h_bld = parameters;
    end
    h_bld.simpar = h_builder(1,1).simpar;
    assemble_tx_rx = true;
    
elseif isa( parameters(1,1),'qd_builder' ) && numel( parameters ) > 1
    if qf.eqo( parameters(1,1), h_builder(1,1) )
        h_bld = copy( parameters );
        parameters = [];
    else
        h_bld = parameters;
    end
    for i_cb = 1 : numel(h_bld)
        [ i1,i2 ] = qf.qind2sub( sic, i_cb );
        h_bld(i1,i2).simpar = h_builder(1,1).simpar;
    end
    assemble_tx_rx = false;
    
else
    error('QuaDRiGa:qd_builder:add_clusters',...
        '"parameters" must be a string, qd_builder object or empty variable.');
end

% Assemble existing Tx and Rx positions to generate LSF and SSF with the new builder
tx_pos = [];
rx_pos = [];
iB = [];
for i_cb = 1 : numel(h_builder)
    [ i1,i2 ] = qf.qind2sub( sic, i_cb );
    if h_builder(i1,i2).dual_mobility == -1 || ...
            any( size( h_builder(i1,i2).tx_position ) ~= size( h_builder(i1,i2).rx_positions ) )
        check_dual_mobility( h_builder(i1,i2), false );
    end
    n_rx_positions = h_builder(i1,i2).no_rx_positions;
    if n_rx_positions > 0
        tx_pos = [tx_pos,h_builder(i1,i2).tx_position]; %#ok!
        rx_pos = [rx_pos,h_builder(i1,i2).rx_positions]; %#ok!
        
        % Entries in iB:
        % (1) First Builder Index
        % (2) Second Builder Index
        % (3) Number of existing paths
        % (4) LOS state
        % (5) Position index within builder
        
        iB = [ iB, [repmat( [ i1; i2; numel( h_builder(i1,i2).NumSubPaths );...
            h_builder(i1,i2).check_los ],1,n_rx_positions);1:n_rx_positions] ]; %#ok!
    end
end

% Assign TX and RX positions to new builder
if assemble_tx_rx
    h_bld.tx_position = tx_pos;
    h_bld.rx_positions = rx_pos;
    h_bld.tx_track = [];
    h_bld.rx_track = [];
else
    gen_fbs_lbs( h_bld,0 );
    gen_ssf_parameters( h_bld,0,0 );
    if isempty( parameters )
        gen_lsf_parameters( h_bld,0,0 );
    end
end
check_dual_mobility( h_bld, false );

% Generate parameters
gen_ssf_parameters( h_bld, 0 );   % Clear existing SSF parameters
init_sos( h_bld, 2 );             % Reuse existing SOS parameters
gen_lsf_parameters( h_bld, 2 );   % Reuse existing LSF parameters
gen_ssf_parameters( h_bld, 1 );   % Create new SSF parameters

% Split builder such that each object only has one link
h_bld = split_rx( h_bld );

% Add new clusters to input builders
for iN = 1 : numel( h_bld )
    
    % Find entry in "h_builder" by matching the positions in "h_bld" and "h_builder"
    iP = (tx_pos(1,:) - h_bld(1,iN).tx_position(1)).^2 + (tx_pos(2,:) - h_bld(1,iN).tx_position(2)).^2 +...
        (tx_pos(3,:) - h_bld(1,iN).tx_position(3)).^2 < 0.0001^2;
    iP = iP & (rx_pos(1,:) - h_bld(1,iN).rx_positions(1)).^2 + (rx_pos(2,:) - h_bld(1,iN).rx_positions(2)).^2 +...
        (rx_pos(3,:) - h_bld(1,iN).rx_positions(3)).^2 < 0.0001^2;
    if sum( iP ) ~= 1
        error('QuaDRiGa:qd_builder:add_clusters','Number of positions does not match!');
    end
    
    % Get the indices
    i1 = iB(1,iP);                                  % First Builder Index
    i2 = iB(2,iP);                                  % Second Builder Index
    iMT = iB(5,iP);                                 % Position index within builder
    isLOSe = iB(4,iP);                              % LOS state of existing builder
    isLOSn = h_bld(1,iN).check_los;                 % LOS state of added paths
    n_exist = iB(3,iP);                             % Number of existing paths
    n_new = h_bld(1,iN).NumClusters;                % Number of added paths
    n_mobiles = h_builder(i1,i2).no_rx_positions;   % Number of mobiles in "h_builder"
    n_freq = h_builder(i1,i2).no_freq;              % Number of frequencies in "h_builder"
    
    % Add zero-power LOS path to the builder, set "n_exist" to 1
    if isLOSe == -1                 % There are no existing paths
        h_builder(i1,i2).NumSubPaths = 1;
        h_builder(i1,i2).NumClusters = 1;
        h_builder(i1,i2).gain = zeros(n_mobiles,1,n_freq);
        h_builder(i1,i2).taus = zeros(n_mobiles,1);
        h_builder(i1,i2).xprmat = repmat([1;0;0;-1],[1,1,n_mobiles,n_freq]);
        h_builder(i1,i2).pin = zeros(n_mobiles,1,n_freq);
        
        los_angles = h_builder(i1,i2).get_angles*pi/180;
        h_builder(i1,i2).AoD = los_angles(1,:)';
        h_builder(i1,i2).AoA = los_angles(2,:)';
        h_builder(i1,i2).EoD = los_angles(3,:)';
        h_builder(i1,i2).EoA = los_angles(4,:)';
        
        if isempty( h_builder(i1,i2).subpath_coupling )
            h_builder(i1,i2).subpath_coupling = rand(4,1,n_freq);
        end
        
        if ~h_builder(i1,i2).simpar(1,1).use_3GPP_baseline
            gen_fbs_lbs( h_builder(i1,i2) );       	% Update FBS / LBS positions
        end
        
        isLOSe = check_los( h_builder(i1,i2) );
        n_exist = 1;
        tmp = iB(1,:) == i1 & iB(2,:) == i2;
        iB(3,tmp) = n_exist; %#ok!
        iB(4,tmp) = isLOSe; %#ok!
    end
    
    % Add zero-power ground reflection to existing builder
    if isLOSe < 1.5 && isLOSn > 1.5
        tx_pos_tmp = h_builder(i1,i2).tx_position;
        rx_pos_tmp = h_builder(i1,i2).rx_positions;
        d_2d = hypot( tx_pos_tmp(1,:) - rx_pos_tmp(1,:), tx_pos_tmp(2,:) - rx_pos_tmp(2,:) );
        d_2d( d_2d<1e-5 ) = 1e-5;
        AoD = atan2( rx_pos_tmp(2,:) - tx_pos_tmp(2,:) , rx_pos_tmp(1,:) - tx_pos_tmp(1,:) );   % Azimuth at BS
        AoA = angle(exp(1j*(pi+AoD)));                                          % Azimuth at MT
        EoD = -atan2( ( rx_pos_tmp(3,:) + tx_pos_tmp(3,:) ), d_2d );                    % Ground Reflection Elevation at BS and MT
        dTR = sqrt(sum((tx_pos_tmp - rx_pos_tmp).^2));
        dTR_gr = sqrt(sum(abs(  [ rx_pos_tmp(1:2,:);-rx_pos_tmp(3,:) ] - tx_pos_tmp ).^2,1));
        tau_gr = (dTR_gr-dTR)./qd_simulation_parameters.speed_of_light;
        
        h_builder(i1,i2).NumSubPaths = [1,1,h_builder(i1,i2).NumSubPaths(2:end)];
        h_builder(i1,i2).NumClusters = numel( h_builder(i1,i2).NumSubPaths );
        h_builder(i1,i2).gain = cat( 2, h_builder(i1,i2).gain(:,1,:), zeros(n_mobiles,1,n_freq), h_builder(i1,i2).gain(:,2:end,:)  );
        h_builder(i1,i2).AoD = [ h_builder(i1,i2).AoD(:,1), AoD', h_builder(i1,i2).AoD(:,2:end) ];
        h_builder(i1,i2).AoA = [ h_builder(i1,i2).AoA(:,1), AoA', h_builder(i1,i2).AoA(:,2:end) ];
        h_builder(i1,i2).EoD = [ h_builder(i1,i2).EoD(:,1), EoD', h_builder(i1,i2).EoD(:,2:end) ];
        h_builder(i1,i2).EoA = [ h_builder(i1,i2).EoA(:,1), EoD', h_builder(i1,i2).EoA(:,2:end) ];
        h_builder(i1,i2).xprmat = cat( 2, h_builder(i1,i2).xprmat(:,1,:,:), ...
            repmat([1;0;0;-1],[1,1,n_mobiles,n_freq]), h_builder(i1,i2).xprmat(:,2:end,:,:) );
        h_builder(i1,i2).pin = cat( 2, h_builder(i1,i2).pin(:,1,:), zeros(n_mobiles,1,n_freq), h_builder(i1,i2).pin(:,2:end,:) );
        
        if h_builder(i1,i2).scenpar.GR_enabled == 0
            h_builder(i1,i2).subpath_coupling = cat( 2, h_builder(i1,i2).subpath_coupling(:,1,:), ...
                rand(4,1,n_freq), h_builder(i1,i2).subpath_coupling(:,2:end,:) );
        end
        
        n_freq_taus = size(h_builder(i1,i2).taus,3);
        h_builder(i1,i2).taus = cat( 2, h_builder(i1,i2).taus(:,1,:), repmat(tau_gr',[1,1,n_freq_taus]), h_builder(i1,i2).taus(:,2:end,:)  );
        
        if ~h_builder(i1,i2).simpar(1,1).use_3GPP_baseline
            gen_fbs_lbs( h_builder(i1,i2) );       	% Update FBS / LBS positions
        end
        
        isLOSe = check_los( h_builder(i1,i2) );
        n_exist = h_builder(i1,i2).NumClusters;
        tmp = iB(1,:) == i1 & iB(2,:) == i2;
        iB(3,tmp) = n_exist;%#ok!
        iB(4,tmp) = isLOSe; %#ok!
    end
    
    % Determine the indices of the NLOS paths in "h_bld" and "h_builder"
    if isLOSn > 1.5 % There is a GR path
        % A zero-power GR was already added to the existing builder, so "isLOSe > 1.5" is true as well
        n_add  = n_exist + n_new - h_builder(i1,i2).NumClusters - 2;
        i_clst = 3 : n_new;
        i_path = 3 : sum( h_bld(1,iN).NumSubPaths );
        j_clst = n_exist + 1 : n_exist + n_new - 2;
        j_path = sum( h_builder(i1,i2).NumSubPaths(1:n_exist) );
        j_path = j_path + 1 : j_path + sum(h_bld(1,iN).NumSubPaths(3:end));
    else            % There is no GR path in the new builder
        n_add  = n_exist + n_new - h_builder(i1,i2).NumClusters - 1;
        i_clst = 2 : n_new;
        i_path = 2 : sum( h_bld(1,iN).NumSubPaths );
        j_clst = n_exist + 1 : n_exist + n_new - 1;
        j_path = sum( h_builder(i1,i2).NumSubPaths(1:n_exist) );
        j_path = j_path + 1 : j_path + sum(h_bld(1,iN).NumSubPaths(2:end));
    end
    
    % Enlarge existing data structures
    if n_add > 0
        NumSubPaths = h_bld(1,iN).NumSubPaths( end-n_add+1:end );
        h_builder(i1,i2).NumSubPaths = [ h_builder(i1,i2).NumSubPaths, NumSubPaths ];
        h_builder(i1,i2).NumClusters = numel( h_builder(i1,i2).NumSubPaths );
        NumSubPaths = sum( NumSubPaths );
        h_builder(i1,i2).taus = cat( 2, h_builder(i1,i2).taus, zeros(n_mobiles,n_add,size(h_builder(i1,i2).taus,3)) );
        h_builder(i1,i2).gain = cat( 2, h_builder(i1,i2).gain, zeros(n_mobiles,n_add,n_freq) );
        h_builder(i1,i2).AoD = [ h_builder(i1,i2).AoD, zeros(n_mobiles,n_add) ];
        h_builder(i1,i2).AoA = [ h_builder(i1,i2).AoA, zeros(n_mobiles,n_add) ];
        h_builder(i1,i2).EoD = [ h_builder(i1,i2).EoD, zeros(n_mobiles,n_add) ];
        h_builder(i1,i2).EoA = [ h_builder(i1,i2).EoA, zeros(n_mobiles,n_add) ];
        h_builder(i1,i2).xprmat = cat( 2, h_builder(i1,i2).xprmat, zeros(4,NumSubPaths,n_mobiles,n_freq) );
        h_builder(i1,i2).pin = cat( 2, h_builder(i1,i2).pin, zeros(n_mobiles,NumSubPaths,n_freq) );
        
        % Add entries to "subpath_coupling"
        n_subpath_coupling = size(h_builder(i1,i2).subpath_coupling,2);     % Size of existing "subpath_coupling"
        NumSubPaths = sum( h_builder(i1,i2).NumSubPaths );                  % Number of subpaths in the builder after adding new paths
        if NumSubPaths > n_subpath_coupling
            n_add = NumSubPaths - n_subpath_coupling;                       % Values to be added to "subpath_coupling"
            h_builder(i1,i2).subpath_coupling = cat( 2, h_builder(i1,i2).subpath_coupling, rand(4,n_add,n_freq) );
        end
    end
    
    % Copy NLOS paths from the new builder to the existing builder
    if ~isempty( i_clst )
        h_builder(i1,i2).gain(iMT,j_clst,:) = h_bld(1,iN).gain(1,i_clst,:);
        h_builder(i1,i2).AoD(iMT,j_clst) = h_bld(1,iN).AoD(1,i_clst);
        h_builder(i1,i2).AoA(iMT,j_clst) = h_bld(1,iN).AoA(1,i_clst);
        h_builder(i1,i2).EoD(iMT,j_clst) = h_bld(1,iN).EoD(1,i_clst);
        h_builder(i1,i2).EoA(iMT,j_clst) = h_bld(1,iN).EoA(1,i_clst);
        h_builder(i1,i2).xprmat(:,j_path,iMT,:) = h_bld(1,iN).xprmat(:,i_path,1,:);
        h_builder(i1,i2).pin(iMT,j_path,:) = h_bld(1,iN).pin(1,i_path,:);
        h_builder(i1,i2).subpath_coupling(:,j_path,:) = h_bld(1,iN).subpath_coupling(:,i_path,:);
        if size( h_builder(i1,i2).taus,3 ) == n_freq && size( h_bld(1,iN).taus,3 ) == 1
            h_builder(i1,i2).taus(iMT,j_clst,:) = repmat( h_bld(1,iN).taus(1,i_clst,:), [1,1,n_freq] );
        elseif size( h_builder(i1,i2).taus,3 ) == 1 && size( h_bld(1,iN).taus,3 ) == n_freq
            h_builder(i1,i2).taus = repmat( h_builder(i1,i2).taus, [1,1,n_freq] );
            h_builder(i1,i2).taus(iMT,j_clst,:) = h_bld(1,iN).taus(1,i_clst,:);
        elseif size( h_builder(i1,i2).taus,3 ) ~= size( h_bld(1,iN).taus,3 )
            error('QuaDRiGa:qd_builder:add_clusters','Size of "taus" is inconsistent.');
        else
            h_builder(i1,i2).taus(iMT,j_clst,:) = h_bld(1,iN).taus(1,i_clst,:);
        end
    end
    
    % Update LOS and GR components with the values from the new builder
    if isLOSn == 1 || isLOSn == 2
        % Copy LOS path from new builder and overwrite existing LOS
        % Angles and taus are correct due to calling "check_los" on both builders
        h_builder(i1,i2).gain(iMT,1,:) = h_bld(1,iN).gain(1,1,:);
        h_builder(i1,i2).xprmat(:,1,iMT,:) = h_bld(1,iN).xprmat(:,1,1,:);
        h_builder(i1,i2).pin(iMT,1,:) = h_bld(1,iN).pin(1,1,:);
    end
    if isLOSe == 2 || isLOSe == 3   % Existing builder has a GR path?
        if isLOSn == 1              % New builder has no GR - clear GR path from existing builder
            h_builder(i1,i2).gain(iMT,2,:) = 0;
        elseif isLOSn == 2          % Copy GR path from new builder and overwrite existing GR
            h_builder(i1,i2).gain(iMT,2,:) = h_bld(1,iN).gain(1,2,:);
            h_builder(i1,i2).xprmat(:,2,iMT,:) = h_bld(1,iN).xprmat(:,2,1,:);
            h_builder(i1,i2).pin(iMT,2,:) = h_bld(1,iN).pin(1,2,:);
        end
    end
end

% Calculate the FBS and LBS positions of all new paths
for i_cb = 1 : numel(h_builder)
    [ i1,i2 ] = qf.qind2sub( sic, i_cb );
    if ~h_builder(i1,i2).simpar(1,1).use_3GPP_baseline && h_builder(i1,i2).no_rx_positions ~= 0
        n_exist = iB(3, find( iB(1,:) == i1 & iB(2,:) == i2 , 1 ));
        n_rays_exisit = sum( h_builder(i1,i2).NumSubPaths(1:n_exist) );
        if isempty( h_builder(i1,i2).lbs_pos ) || isempty( h_builder(i1,i2).fbs_pos )
            gen_fbs_lbs( h_builder(i1,i2) );       	% Update FBS / LBS positions
        elseif h_builder(i1,i2).NumClusters > n_exist && size( h_builder(i1,i2).fbs_pos,2 ) >= n_rays_exisit
            fbs_pos = h_builder(i1,i2).fbs_pos;     % Existing FBS-Pos. should not change
            lbs_pos = h_builder(i1,i2).lbs_pos;     % Existing LBS-Pos. should not change
            gen_fbs_lbs( h_builder(i1,i2) );       	% Update FBS / LBS positions
            h_builder(i1,i2).fbs_pos(:,1:n_rays_exisit,:,:) = fbs_pos;
            h_builder(i1,i2).lbs_pos(:,1:n_rays_exisit,:,:) = lbs_pos;
        end
        if h_builder(i1,i2).NumClusters > n_exist && size( h_builder(i1,i2).fbs_pos,2 ) == sum( h_builder(i1,i2).NumSubPaths )
            taus = h_builder(i1,i2).taus(:,1:n_exist,:);    % Store existing
            AoD = h_builder(i1,i2).AoD(:,1:n_exist);
            AoA = h_builder(i1,i2).AoA(:,1:n_exist);
            EoD = h_builder(i1,i2).EoD(:,1:n_exist);
            EoA = h_builder(i1,i2).EoA(:,1:n_exist);
            gen_ssf_from_scatterers( h_builder(i1,i2), 1e-9 );    % Update
            h_builder(i1,i2).taus(:,1:n_exist,:) = taus;    % Restore existing
            h_builder(i1,i2).AoD(:,1:n_exist) = AoD;
            h_builder(i1,i2).AoA(:,1:n_exist) = AoA;
            h_builder(i1,i2).EoD(:,1:n_exist) = EoD;
            h_builder(i1,i2).EoA(:,1:n_exist) = EoA;
        end
    end
end

% Update other parameters
gen_lsf_from_ssf( h_builder );

end
