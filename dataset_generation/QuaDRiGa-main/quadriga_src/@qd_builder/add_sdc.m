function add_sdc( h_builder, pos_rel, pow_dB, pos_reference, pow_reference, xpr_dB,...
    sdc_radius, NumSubPaths, distance, azimuth, elevation )
%ADD_SDC Adds semi-deterministic clusters to the channel builder
%
% Calling object:
%   Object array
%
% Description:
%   Semi-deterministic clusters (SDCs) are clusters that have a specific position relative to a
%   defined reference position and power relative to a defined reference power. The reference
%   position can be selected by the input variable 'pos_reference'. The reference power can be
%   selected by the input variable 'pow_reference'. SDCs are added to all TX-RX links of the
%   calling channel builder object or object array. If you want to add SDCc only to a specific link
%   or set of links, you can use 'split_rx' to obtain separate builders for each link. If you want
%   to combine SDCs with other stochastic paths, you must call 'add_sdc' last, i.e. after calling
%   'add_paths' or 'gen_parameters'. SDCs are added to the builder after the LOS and GR path,
%   before the NLOS clusters.
%
% Position ref.:
%   absolute
%   The SDCs are placed relative to the origin of the global Cartesian coordinate system. As a
%   consequence, all links "see" the same SDCs.
%
%   rx_abs
%   The SDCs are placed relative to each RX position (i.e., the initial position of each track
%   segment). The RX orientation is not taken into account.
%
%   rx_heading
%   The SDCs are placed relative to each RX position (i.e., the initial position of each track
%   segment). The RX orientation is taken into account, but only for the "heading" direction (i.e.,
%   the 3rd row of 'rx_track.orientation'). The SDCs elevation is set relative to the ground.
%
%   rx_full
%   The SDCs are placed relative to each RX position (i.e., the initial position of each track
%   segment). All components (bank-angle, tilt-angle and heading-angle) of the RX orientation
%   defined by 'rx_track.orientation' are taken into account.
%
%   tx_abs
%   The SDCs are placed relative to each TX position (i.e., the initial position of each track
%   segment). The TX orientation is not taken into account.
%
%   tx_heading
%   The SDCs are placed relative to each TX position (i.e., the initial position of each track
%   segment). The TX orientation is taken into account, but only for the "heading" direction (i.e.,
%   the 3rd row of 'tx_track.orientation'). The SDCs elevation is set relative to the ground.
%
%   tx_full
%   The SDCs are placed relative to each TX position (i.e., the initial position of each track
%   segment). All components (bank-angle, tilt-angle and heading-angle) of the TX orientation
%   defined by 'tx_track.orientation' are taken into account.
%
%   los_rx
%   The SDCs are placed relative to the LOS path originating at the RX going towards the TX. Small
%   offset positions place the SDCs closer to the RX.
%
%   los_tx
%   The SDCs are placed relative to the LOS path originating at the TX going towards the RX. Small
%   offset positions place the SDCs closer to the TX.
%
% Power ref.:
%   freespace
%   The SDCs power is defined relative to the free-space-power calculated from the TX-RX distance
%   (in units of [dB]).
%
%   absolute
%   The SDCs power is defined relative to the TX power in [dB].
%
%   pathloss
%   The SDCs power is defined relative  to the path-loss defined by the 'qd_builder' object to
%   which the SDC is added (in units of [dB]).
%
% Input:
%   pos_rel
%   A [ 3 x L ] matrix containing the SDCs Cartesian [x;y;z] positions relative to the reference
%   position in meters. If this variable is empty, the input variables 'distance', 'azimuth' and
%   'elevation' are used to compute the SDC positions from Geographic coordinates. L is the number
%   of SDCs that are added to the builder.
%
%   pow_dB
%   A [ 1 x L ] column vector or scalar variable containing the power values of the SDCs relative
%   to the reference power.
%
%   pos_reference
%   A string defining the reference position of the SDC. Default: 'absolute'.
%
%   pow_reference
%   A string defining the reference power of the SDC. Default: 'freespace'.
%
%   xpr_dB
%   A scalar variable or [ 1 x L ]  column vector containing the XPR values of the SDCs. By default
%   (i.e., if this variable is empty or not given), the XPR values are drawn from distributions
%   defined by the scenario parameters in each builder.
%
%   sdc_radius
%   A scalar variable or [ 1 x L ] column vector containing the effective radius of the SDC in
%   [meters]. SDCs are approximated by a finite number of specular paths. The distribution of the
%   paths is determined by the scenario parameter 'SubpathMethod' which is set by the calling
%   'qd_builder' object.
%
%   NumSubPaths
%   A scalar variable or [ 1 x L ] column vector defining the number of sub-paths that are used to
%   approximate the SDCs by individual paths. By default, one subpath is used if 'sdc_radius' is 0
%   (point source) and 20 sub-paths are used otherwise.
%
%   distance
%   This variable is only used if 'pos_rel' is empty. It is given as a scalar variable or [ 1 x L ]
%   column vector defining the distances between the reference position (RX or TX) and the SDC. For
%   absolute reference positions, the origin of the global Cartesian coordinate system is used as
%   reference.
%
%   azimuth
%   This variable is only used if 'pos_rel' is empty. It is given as a scalar variable or [ 1 x L ]
%   column vector defining the azimuth angles of the SDCc (AoD for TX reference, AoA for RX
%   reference) in [rad]. By default, random angles are drawn from the azimuth angle spread defined
%   in  'qd_builder.scenpar' (if 'pos_rel' is empty, 'distance' is given and 'azimuth' is empty).
%
%   elevation
%   This variable is only used if 'pos_rel' is empty. It is given as a scalar variable or [ 1 x L ]
%   column vector defining the elevation angles of the SDCc (EoD for TX reference, EoA for RX
%   reference) in [rad]. By default, random angles are drawn from the elevation angle spread
%   defined in 'qd_builder.scenpar' (if 'pos_rel' is empty, 'distance' is given and 'elevation' is
%   empty).
%
%
% QuaDRiGa Copyright (C) 2011-2021
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

% Check size of builder array
sic = size( h_builder );
if numel( sic ) > 2
    error('QuaDRiGa:qd_builder:add_sdc','??? Array of qd_builder objects cannot have more that 2 dimensions.');
end

% Check "pos_rel"
if ~exist( 'pos_rel','var' )
    error('QuaDRiGa:qd_builder:add_sdc','??? "pos_rel" is not given.')
elseif isempty( pos_rel )
    % Do nothing - use "distance", "azimuth", "elevation" to create clusters
elseif ~( isnumeric(pos_rel) && isreal(pos_rel) )
    error('QuaDRiGa:qd_builder:add_sdc','??? "pos_rel" must consist of real numbers')
elseif ~all( size(pos_rel,1) == 3 )
    error('QuaDRiGa:qd_builder:add_sdc','??? "pos_rel" must have 3 rows')
end

% If "pos_rel" is empty, we use "distance", "azimuth", "elevation" to create clusters
if isempty( pos_rel )
    
    % Check "distance"
    if ~exist( 'distance','var' ) || isempty( distance )
        error('QuaDRiGa:qd_builder:add_sdc','??? "distance" is not given.')
    elseif ~( isnumeric(distance) && isreal(distance) )
        error('QuaDRiGa:qd_builder:add_sdc','??? "distance" must consist of real numbers.')
    elseif numel( distance ) == 1 && numel( pow_dB ) > 1
        distance = repmat( distance, 1, numel( pow_dB ));
    end
    n_sdc = numel(distance);
    distance = reshape( distance, 1, n_sdc );
    
    % Check "azimuth"
    if ~exist( 'azimuth','var' ) || isempty( azimuth )
        azimuth = [];
    elseif ~( isnumeric(azimuth) && isreal(azimuth) )
        error('QuaDRiGa:qd_builder:add_sdc','??? "azimuth" must consist of real numbers')
    elseif n_sdc > 1 && numel( azimuth ) == 1
        azimuth = repmat( azimuth, 1, n_sdc);
    end
    if ~isempty( azimuth )
        azimuth = reshape( azimuth, 1, n_sdc );
    end
    
    % Check "elevation"
    if ~exist( 'elevation','var' ) || isempty( elevation )
        elevation = [];
    elseif ~( isnumeric(elevation) && isreal(elevation) )
        error('QuaDRiGa:qd_builder:add_sdc','??? "elevation" must consist of real numbers')
    elseif n_sdc > 1 && numel( elevation ) == 1
        elevation = repmat( elevation, 1, n_sdc);
    end
    if ~isempty( elevation )
        elevation = reshape( elevation, 1, n_sdc );
    end
    
else
    n_sdc = size(pos_rel,2);
end

% Check "pow_dB"
if ~exist( 'pow_dB','var' ) || isempty( pow_dB )
    error('QuaDRiGa:qd_builder:add_sdc','??? "pow_dB" is not given.')
elseif ~( isnumeric(pow_dB) && isreal(pow_dB) )
    error('QuaDRiGa:qd_builder:add_sdc','??? "pow_dB" must consist of real numbers.')
elseif n_sdc > 1 && numel( pow_dB ) == 1
    pow_dB = repmat( pow_dB, 1, n_sdc);
end
pow_dB = reshape( pow_dB, 1, n_sdc );

% Check "pos_reference"
supported_types = {'absolute','rx_abs','rx_heading','rx_full','tx_abs','tx_heading','tx_full','los_rx','los_tx'};
if ~exist( 'pos_reference' , 'var' ) || isempty( pos_reference )
    pos_reference = 'absolute';
elseif ~( ischar(pos_reference) && any( strcmpi(pos_reference,supported_types)) )
    str = 'Position reference not found. Supported types are: ';
    no = numel(supported_types);
    for n = 1:no
        str = [str,supported_types{n}];
        if n<no
            str = [str,', '];
        end
    end
    error(str);
end

% Check "pow_reference"
supported_types = {'freespace','absolute','pathloss'};
if ~exist( 'pow_reference' , 'var' ) || isempty( pow_reference )
    pow_reference = 'freespace';
elseif ~( ischar(pow_reference) && any( strcmpi(pow_reference,supported_types)) )
    str = 'Position reference not found. Supported types are: ';
    no = numel(supported_types);
    for n = 1:no
        str = [str,supported_types{n}];
        if n<no
            str = [str,', '];
        end
    end
    error(str);
end

% Check "xpr_dB"
if ~exist( 'xpr_dB','var' ) || isempty( xpr_dB )
    xpr_dB = [];
elseif ~( isnumeric(xpr_dB) && isreal(xpr_dB) )
    error('QuaDRiGa:qd_builder:add_sdc','??? "xpr_dB" must consist of real numbers')
elseif n_sdc > 1 && numel( xpr_dB ) == 1
    xpr_dB = repmat( xpr_dB, 1, n_sdc);
end
if ~isempty( xpr_dB )
    xpr_dB = reshape( xpr_dB, 1, n_sdc );
end

% Check "sdc_radius"
if ~exist( 'sdc_radius','var' ) || isempty( sdc_radius )
    sdc_radius = [];
elseif ~( isnumeric(sdc_radius) && isreal(sdc_radius) )
    error('QuaDRiGa:qd_builder:add_sdc','??? "sdc_radius" must consist of real numbers')
elseif n_sdc > 1 && numel( sdc_radius ) == 1
    sdc_radius = repmat( sdc_radius, 1, n_sdc );
end
if ~isempty( sdc_radius )
    sdc_radius = reshape( sdc_radius, 1, n_sdc );
end

% Check "NumSubPaths"
if ~exist( 'NumSubPaths','var' ) || isempty( NumSubPaths )
    NumSubPaths = ones(1,n_sdc) * 20;
    if ~isempty( sdc_radius )
        NumSubPaths( sdc_radius == 0 ) = 1;
    end
elseif ~( isnumeric(NumSubPaths) && isreal(NumSubPaths) )
    error('QuaDRiGa:qd_builder:add_sdc','??? "NumSubPaths" must consist of real numbers')
elseif n_sdc > 1 && numel( NumSubPaths ) == 1
    NumSubPaths = repmat( NumSubPaths, 1, n_sdc);
end
if ~isempty( NumSubPaths )
    NumSubPaths = reshape( NumSubPaths, 1, n_sdc );
end

% Assemble existing Tx and Rx positions to generate LSF and SSF with the new builder
tx_pos = [];
rx_pos = [];
tx_orientation = [];
rx_orientation = [];
iB = [];
for i_cb = 1 : numel(h_builder)
    [ i1,i2 ] = qf.qind2sub( sic, i_cb );
    if h_builder(i1,i2).dual_mobility == -1 || ...
            any( size( h_builder(i1,i2).tx_position ) ~= size( h_builder(i1,i2).rx_positions ) )
        check_dual_mobility( h_builder(i1,i2), false );
    end
    n_rx_positions = h_builder(i1,i2).no_rx_positions;
    if h_builder(i1,i2).simpar(1,1).use_3GPP_baseline
        error('QuaDRiGa:qd_builder:add_sdc',...
            '??? "Semi-deterministic clusters are not supported for 3GPP baseline simulations.')
    elseif h_builder(i1,i2).check_los > 0 && isempty( h_builder(i1,i2).fbs_pos )
        gen_fbs_lbs( h_builder(i1,i2) );
    end
    if n_rx_positions > 0
        tx_pos = [tx_pos,h_builder(i1,i2).tx_position]; %#ok!
        rx_pos = [rx_pos,h_builder(i1,i2).rx_positions]; %#ok!
        for i3 = 1 : n_rx_positions
            initial_pos = h_builder(i1,i2).rx_track(1,i3).segment_index(...
                min( [h_builder(i1,i2).rx_track(1,i3).no_segments,2] ));
            rx_orientation = [ rx_orientation, h_builder(i1,i2).rx_track(1,i3).orientation(:,initial_pos) ]; %#ok!
            if h_builder(i1,i2).dual_mobility
                tx_orientation = [ tx_orientation, h_builder(i1,i2).tx_track(1,i3).orientation(:,initial_pos) ]; %#ok!
            else
                tx_orientation = [ tx_orientation, h_builder(i1,i2).tx_track(1,i3).orientation(:,1) ]; %#ok!
            end
        end
        
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

% Add new clusters to builders
for iP = 1 : size( iB,2 )
    
    % Get the indices
    i1 = iB(1,iP);                                  % First Builder Index
    i2 = iB(2,iP);                                  % Second Builder Index
    iMT = iB(5,iP);                                 % Position index within builder
    isLOSe = iB(4,iP);                              % LOS state of existing builder
    n_exist = iB(3,iP);                             % Number of existing clusters
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
    
    % Determine the indices of the SDC paths in "h_builder"
    n_add  = n_exist + n_sdc - h_builder(i1,i2).NumClusters;
    if isLOSe > 1.5 % Ground reflection
        i_los  = 2;
        i_nlos = 3;
    else
        i_los  = 1;
        i_nlos = 2;
    end
    j_clst = i_nlos : i_los + n_sdc;
    j_path = i_nlos : i_los + sum( NumSubPaths );
        
    % Enlarge existing data structures
    if n_add > 0
        h_builder(i1,i2).NumSubPaths = [ h_builder(i1,i2).NumSubPaths(1:i_los), NumSubPaths, h_builder(i1,i2).NumSubPaths(i_nlos:end) ];
        h_builder(i1,i2).NumClusters = numel( h_builder(i1,i2).NumSubPaths );
        NumSubPathsS = sum( NumSubPaths );
        h_builder(i1,i2).taus = cat( 2, h_builder(i1,i2).taus(:,1:i_los,:),...
            zeros(n_mobiles,n_add,size(h_builder(i1,i2).taus,3)), h_builder(i1,i2).taus(:,i_nlos:end,:) );
        h_builder(i1,i2).gain = cat( 2, h_builder(i1,i2).gain(:,1:i_los,:),...
            zeros(n_mobiles,n_add,n_freq), h_builder(i1,i2).gain(:,i_nlos:end,:) );
        h_builder(i1,i2).AoD = [ h_builder(i1,i2).AoD(:,1:i_los), zeros(n_mobiles,n_add), h_builder(i1,i2).AoD(:,i_nlos:end) ];
        h_builder(i1,i2).AoA = [ h_builder(i1,i2).AoA(:,1:i_los), zeros(n_mobiles,n_add), h_builder(i1,i2).AoA(:,i_nlos:end) ];
        h_builder(i1,i2).EoD = [ h_builder(i1,i2).EoD(:,1:i_los), zeros(n_mobiles,n_add), h_builder(i1,i2).EoD(:,i_nlos:end) ];
        h_builder(i1,i2).EoA = [ h_builder(i1,i2).EoA(:,1:i_los), zeros(n_mobiles,n_add), h_builder(i1,i2).EoA(:,i_nlos:end) ];
        h_builder(i1,i2).xprmat = cat( 2, h_builder(i1,i2).xprmat(:,1:i_los,:,:),...
            zeros(4,NumSubPathsS,n_mobiles,n_freq), h_builder(i1,i2).xprmat(:,i_nlos:end,:,:) );
        h_builder(i1,i2).pin = cat( 2, h_builder(i1,i2).pin(:,1:i_los,:), zeros(n_mobiles,NumSubPathsS,n_freq),...
            h_builder(i1,i2).pin(:,i_nlos:end,:) );
        
        h_builder(i1,i2).fbs_pos = cat( 2, h_builder(i1,i2).fbs_pos(:,1:i_los,:,:),...
            zeros(3,NumSubPathsS,n_mobiles,n_freq), h_builder(i1,i2).fbs_pos(:,i_nlos:end,:,:) );
        h_builder(i1,i2).lbs_pos = cat( 2, h_builder(i1,i2).lbs_pos(:,1:i_los,:,:),...
            zeros(3,NumSubPathsS,n_mobiles,n_freq), h_builder(i1,i2).lbs_pos(:,i_nlos:end,:,:) );
        
        % Add entries to "subpath_coupling"
        n_subpath_coupling = size(h_builder(i1,i2).subpath_coupling,2);      % Size of existing "subpath_coupling"
        NumSubPathsS = sum( h_builder(i1,i2).NumSubPaths );                  % Number of subpaths in the builder after adding new paths
        if NumSubPathsS > n_subpath_coupling
            n_add = NumSubPathsS - n_subpath_coupling;                       % Values to be added to "subpath_coupling"
            h_builder(i1,i2).subpath_coupling = cat( 2, h_builder(i1,i2).subpath_coupling(:,1:i_los,:),...
                rand(4,n_add,n_freq), h_builder(i1,i2).subpath_coupling(:,i_nlos:end,:) );
        end
    end
    
    % Create a builder that generates the SDCs and initialize it with the LOS component
    h_bld = qd_builder('Freespace');
    if strcmp( pow_reference, 'pathloss' )
        h_bld.plpar = h_builder(i1,i2).plpar;
    end
    h_bld.simpar = h_builder(i1,i2).simpar;
    h_bld.tx_position = tx_pos(:,iP);
    h_bld.rx_positions = rx_pos(:,iP);
    h_bld.check_dual_mobility( false );
    h_bld.gen_parameters;
    h_bld.sos = [];
    
    if isempty( pos_rel )    % Create relative SDC positions
        
        switch pos_reference
            case { 'absolute','rx_abs','rx_heading','rx_full' }
                if isempty( azimuth )
                    mu  = 10^(mean( h_builder(i1,i2).lsp_vals(5,1,:) ));    % Mean ASA in [deg]
                    tmp = 3.465 * (rand( 1,n_sdc )-0.5);                    % Random variable with zero-mean, unit STD
                    azimuth = mu * tmp * pi/180;                            % Azimuth angle in [rad]
                end
                if isempty( elevation )
                    mu  = 10^(mean( h_builder(i1,i2).lsp_vals(7,1,:) ));    % Mean ESA in [deg]
                    tmp = 3.465 * (rand( 1,n_sdc )-0.5);                    % Random variable with zero-mean, unit STD
                    elevation = mu * tmp * pi/180;                          % Elevation angle in [rad]
                end
            otherwise % Tx is reference
                if isempty( azimuth )
                    mu  = 10^(mean( h_builder(i1,i2).lsp_vals(4,1,:) ));    % Mean ASD in [deg]
                    tmp = 3.465 * (rand( 1,n_sdc )-0.5);                    % Random variable with zero-mean, unit STD
                    azimuth = mu * tmp * pi/180;                            % Azimuth angle in [rad]
                end
                if isempty( elevation )
                    mu  = 10^(mean( h_builder(i1,i2).lsp_vals(6,1,:) ));    % Mean ESD in [deg]
                    tmp = 3.465 * (rand( 1,n_sdc )-0.5);                    % Random variable with zero-mean, unit STD
                    elevation = mu * tmp * pi/180;                          % Elevation angle in [rad]
                end
        end
        
        % Transform from geographic coordinated to Cartesian coordinates
        pos_rel_sdc = zeros( 3,n_sdc );
        pos_rel_sdc(1,:) = distance .* cos( azimuth ) .* cos( elevation );
        pos_rel_sdc(2,:) = distance .* sin( azimuth ) .* cos( elevation );
        pos_rel_sdc(3,:) = distance .* sin( elevation );

    else % Use absolute SDC position offset
        pos_rel_sdc = pos_rel;
    end
    
    % Set the SDC center position
    switch pos_reference
        case 'absolute'
            sdc_pos = pos_rel_sdc;
            
        case 'rx_abs'
            sdc_pos = rx_pos(:,iP) + pos_rel_sdc;
            
        case 'rx_heading'
            R = qf.calc_ant_rotation( rx_orientation(3,iP) );
            sdc_pos = rx_pos(:,iP) + R*pos_rel_sdc;
            
        case 'rx_full'
            R = qf.calc_ant_rotation( rx_orientation(3,iP), rx_orientation(2,iP), rx_orientation(1,iP) );
            sdc_pos = rx_pos(:,iP) + R*pos_rel_sdc;
            
        case 'tx_abs'
            sdc_pos = tx_pos(:,iP) + pos_rel_sdc;
            
        case  'tx_heading'
            R = qf.calc_ant_rotation( tx_orientation(3,iP) );
            sdc_pos = tx_pos(:,iP) + R*pos_rel_sdc;

        case 'tx_full'
            R = qf.calc_ant_rotation( tx_orientation(3,iP), tx_orientation(2,iP), tx_orientation(1,iP) );
            sdc_pos = tx_pos(:,iP) + R*pos_rel_sdc;
            
        case 'los_rx'
            los = tx_pos(:,iP) - rx_pos(:,iP);
            los = los ./ norm(los);
            los_az = atan2(los(2),los(1));
            los_el = atan2(los(3),hypot( los(1),los(2) ));
            R = qf.calc_ant_rotation( los_az, -los_el );
            sdc_pos = repmat( rx_pos(:,iP), 1,n_sdc) + R*pos_rel_sdc;
            
        case 'los_tx'
            los = rx_pos(:,iP) - tx_pos(:,iP);
            los = los ./ norm(los);
            los_az = atan2(los(2),los(1));
            los_el = atan2(los(3),hypot( los(1),los(2) ));
            R = qf.calc_ant_rotation( los_az, -los_el );
            sdc_pos = repmat( tx_pos(:,iP), 1,n_sdc) + R*pos_rel_sdc;
    end
    h_bld.fbs_pos = cat( 2, h_bld.fbs_pos, repmat( sdc_pos, [1,1,1,n_freq] ) );
    h_bld.lbs_pos = cat( 2, h_bld.lbs_pos, repmat( sdc_pos, [1,1,1,n_freq] ) );
    
    % Calculate the distance between TX, RX and SDC
    dist_rx_sdc = sqrt( sum( abs( sdc_pos - repmat( rx_pos(:,iP),1,n_sdc) ) .^2,1) );
    dist_tx_sdc = sqrt( sum( abs( repmat( tx_pos(:,iP),1,n_sdc) - sdc_pos ) .^2,1) );
    
    % Set the SDC power
    if strcmp( pow_reference, 'absolute' )
        sdc_gain = repmat( 10.^(0.1*pow_dB ), [1,1,n_freq] );
    else % 'pathloss' or 'freespace'
        d3d = dist_rx_sdc + dist_tx_sdc;            % Total path length including the SDC
        d2d = d3d;
        hUE = repmat( rx_pos(3,iP),1,n_sdc);        % UE height
        hBS = repmat( tx_pos(3,iP),1,n_sdc);        % BS height
        ii = abs(hBS-hUE) > 0.1;                    % BS and UE are on different heights
        alpha = asin( (hBS(ii)-hUE(ii))./d3d(ii) );
        d2d(ii) = (hBS(ii)-hUE(ii))./tan(alpha);                % 2D distance
        sdc_gain = -h_bld.get_pl( [ d2d; zeros(1,n_sdc); hUE ], [], [ zeros(2,n_sdc); hBS ] );  % PG
        sdc_gain = sdc_gain + repmat( pow_dB, n_freq, 1 );      % Scales PG by SDC power loss
        sdc_gain = permute( 10.^(0.1*sdc_gain ), [3,2,1] );     % Reshape
    end
    h_bld.gain = cat( 2, h_bld.gain, sdc_gain );
    h_bld.xprmat = repmat( h_bld.xprmat, [1,1+n_sdc,1,1] ); % Temporary XPR
    h_bld.NumSubPaths = ones(1,1+n_sdc);
    
    % Update the SSF parameters
    h_bld.gen_ssf_from_scatterers;
  
    % Set the number of SubPaths and apply random coupling between sub-paths
    h_bld.NumSubPaths(2:end) = NumSubPaths;
    h_bld.subpath_coupling = rand(4,sum(NumSubPaths)+1,n_freq);
    
    % Calculate the per-cluster angular spread for each SDC
    PerClusterAS = zeros( 4,1+n_sdc ); % [ ASD; ASA; ESD; ESA ]
    if isempty( sdc_radius ) && any( NumSubPaths > 1 )
        PerClusterAS( 1,2:end ) = h_builder(i1,i2).scenpar.PerClusterAS_D;
        PerClusterAS( 2,2:end ) = h_builder(i1,i2).scenpar.PerClusterAS_A;
        PerClusterAS( 3,2:end ) = h_builder(i1,i2).scenpar.PerClusterES_D;
        PerClusterAS( 4,2:end ) = h_builder(i1,i2).scenpar.PerClusterES_A;
    elseif ~isempty( sdc_radius ) 
        for iS = 1 : n_sdc
            dist_rx_sdc = norm( sdc_pos(:,iS) - rx_pos(:,iP) );
            dist_tx_sdc = norm( tx_pos(:,iP) - sdc_pos(:,iS) );
            PerClusterAS( 1,1+iS ) = mean( abs( atand( sdc_radius(iS) ./ dist_tx_sdc ) ) );
            PerClusterAS( 2,1+iS ) = mean( abs( atand( sdc_radius(iS) ./ dist_rx_sdc ) ) );
            PerClusterAS( 3,1+iS ) = mean( abs( atand( sdc_radius(iS) ./ dist_tx_sdc ) ) );
            PerClusterAS( 4,1+iS ) = mean( abs( atand( sdc_radius(iS) ./ dist_rx_sdc ) ) );
        end
    end
    
    % Calculate the positions of the FBS and LBS using the per-cluster angular spread as input
    % "gen_fbs_lbs" cannot operate with very small delay offsets 
    fbs_pos = h_bld.fbs_pos;
    lbs_pos = h_bld.lbs_pos;
    valid = h_bld.taus > 1e-11 & h_bld.NumSubPaths > 1.5;   % 3 cm path length difference
    
    % This maps scatterers to sub-paths
    h_bld.gen_fbs_lbs(1,PerClusterAS);
    
    % Replace invalid data with the original data
    valid( 1 ) = true; % LOS
    invalid = clst_expand( ~valid, h_bld.NumSubPaths );
    if any( invalid )
        for iF = 1 : n_freq
            tmp = clst_expand( fbs_pos(:,:,1,iF), h_bld.NumSubPaths );
            h_bld.fbs_pos(:,invalid,1,iF) = tmp(:,invalid);
            tmp = clst_expand( lbs_pos(:,:,1,iF), h_bld.NumSubPaths );
            h_bld.lbs_pos(:,invalid,1,iF) = tmp(:,invalid);
        end
    end
    
    % Obtain the XPR for each SDC in the current builder
    if isempty( xpr_dB ) % draw from distribution in "h_builder"
        xpr_mu = reshape( h_builder(i1,i2).lsp_vals(8,1,:), n_freq, 1 );
        xpr_sigma = reshape( h_builder(i1,i2).lsp_vals(8,2,:), n_freq, 1 );
        xpr_bld = repmat( randn(1,n_sdc), n_freq,1 ) .* repmat(xpr_sigma,1,n_sdc) + repmat( xpr_mu,1,n_sdc);
    else % get from input variables
        xpr_bld = repmat( xpr_dB, n_freq, 1 );
    end
    
    % Calculate the polarization transfer matrix
    xprmat = zeros( 4, sum(NumSubPaths), n_freq );
    for iF = 1 : n_freq
        xpr_sigma = 0.1 * h_builder(i1,i2).lsp_vals(8,2,iF);                % Random XPR variations
        xpr_mu = clst_expand( xpr_bld(iF,:) , NumSubPaths );                % Average XPR
        
        % Calculate the linear XPR
        xpr_clst = randn( 1,sum(NumSubPaths) ) * xpr_sigma + xpr_mu;
        xpr_clst = 10.^(0.1*xpr_clst);
        tmp = clst_avg( xpr_clst, NumSubPaths ) ./ 10.^(0.1*xpr_bld(iF,:));
        xpr_clst = xpr_clst ./ clst_expand( tmp , NumSubPaths );
        gamma = acot( sqrt( xpr_clst ) );
        
        % Calculate the circular XPR
        xpr_clst = randn( 1,sum(NumSubPaths) ) * xpr_sigma + xpr_mu;
        xpr_clst = 10.^(0.1*xpr_clst);
        tmp = clst_avg( xpr_clst, NumSubPaths ) ./ 10.^(0.1*xpr_bld(iF,:));
        xpr_clst = xpr_clst ./ clst_expand( tmp , NumSubPaths );
        kappa = acot( sqrt( xpr_clst ) );
        kappa = exp(-1j*kappa);
        
        % Linear XPR
        xprmat( 1,:,iF ) = cos( gamma );       % M(1,1)
        xprmat( 2,:,iF ) = sin( gamma );       % M(2,1)
        xprmat( 3,:,iF ) = -xprmat( 2,:,iF );  % M(1,2)
        xprmat( 4,:,iF ) =  xprmat( 1,:,iF );  % M(2,2)
        
        % Circular XPR
        xprmat( 1,:,iF ) = xprmat( 1,:,iF ) .* kappa;   % M(1,1)
        xprmat( 2,:,iF ) = xprmat( 2,:,iF ) .* kappa;   % M(2,1)
        kappa = -conj( kappa );
        xprmat( 3,:,iF ) = xprmat( 3,:,iF ) .* kappa;   % M(1,2)
        xprmat( 4,:,iF ) = xprmat( 4,:,iF ) .* kappa;   % M(2,2)
    end
    h_bld.xprmat = cat( 2, h_bld.xprmat(:,1,:,:), permute( xprmat,[1,2,4,3] ) );
    
    % Generate the random initial phases
    if h_builder(i1,i2).simpar(1,1).use_random_initial_phase
        h_bld.pin = rand(1,sum(NumSubPaths)+1,n_freq)*2*pi - pi;
    else
        h_bld.pin = zeros(1,sum(NumSubPaths)+1,n_freq);
    end
    
    % Update the SSF parameters
    h_bld.gen_ssf_from_scatterers;
    
    % Copy SSF parameters from new builder to the existing builder
    h_builder(i1,i2).gain(iMT,j_clst,:) = h_bld.gain(1,2:end,:);
    h_builder(i1,i2).AoD(iMT,j_clst) = h_bld.AoD(1,2:end);
    h_builder(i1,i2).AoA(iMT,j_clst) = h_bld.AoA(1,2:end);
    h_builder(i1,i2).EoD(iMT,j_clst) = h_bld.EoD(1,2:end);
    h_builder(i1,i2).EoA(iMT,j_clst) = h_bld.EoA(1,2:end);
    h_builder(i1,i2).xprmat(:,j_path,iMT,:) = h_bld.xprmat(:,2:end,1,:);
    h_builder(i1,i2).pin(iMT,j_path,:) = h_bld.pin(1,2:end,:);
    h_builder(i1,i2).subpath_coupling(:,j_path,:) = h_bld.subpath_coupling(:,2:end,:);
    if size( h_builder(i1,i2).taus,3 ) == n_freq && size( h_bld.taus,3 ) == 1
        h_builder(i1,i2).taus(iMT,j_clst,:) = repmat( h_bld.taus(1,2:end,:), [1,1,n_freq] );
    elseif size( h_builder(i1,i2).taus,3 ) == 1 && size( h_bld.taus,3 ) == n_freq
        h_builder(i1,i2).taus = repmat( h_builder(i1,i2).taus, [1,1,n_freq] );
        h_builder(i1,i2).taus(iMT,j_clst,:) = h_bld.taus(1,2:end,:);
    elseif size( h_builder(i1,i2).taus,3 ) ~= size( h_bld.taus,3 )
        error('QuaDRiGa:qd_builder:add_clusters','Size of "taus" is inconsistent.');
    else
        h_builder(i1,i2).taus(iMT,j_clst,:) = h_bld.taus(1,2:end,:);
    end
    h_builder(i1,i2).fbs_pos(:,j_path,iMT,:) = h_bld.fbs_pos(:,2:end,1,:);
    h_builder(i1,i2).lbs_pos(:,j_path,iMT,:) = h_bld.lbs_pos(:,2:end,1,:);

end

% Update other parameters
gen_lsf_from_ssf( h_builder );

end
