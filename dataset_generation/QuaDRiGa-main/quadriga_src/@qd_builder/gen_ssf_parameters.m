function gen_ssf_parameters( h_builder, usage, show_warnings )
%GEN_SSF_PARAMETERS Generates the small-scale-fading parameters for all terminals and frequencies
%
% Calling object:
%   Object array
%
% Description:
%   This method generates the small-scale-fading parameters for each user position. Those
%   parameters are needed by the channel builder to calculate the positions of the scattering
%   clusters for each track or segment which are then evolved into time varying channels. The SSF
%   parameters require that the SOS random generators are initialized first and that the LSF
%   parameters are initialized. Hence, you need to call 'qd_builder.init_sos' and
%   'qd_builder.gen_lsf_parameters' before calling 'qd_builder.gen_ssf_parameters'. The output of
%   this method is written to the object properties.
%
% SSF parameters:
%   NumClusters     The number of clusters, including LOS and ground-reflection
%   NumSubPaths     The number of sub-paths for each cluster. Dimensions: [ 1, NumClusters ]
%   taus            The delays for each cluster in [s] relative to the LOS delay
%   gain            The absolute path-gain for each cluster (linear values)
%   pow             The normalized cluster-powers (squared average amplitude) for each cluster
%   AoD             The azimuth of departure angles for each path in [rad]
%   AoA             The azimuth of arrival angles for each path in [rad]
%   EoD             The elevation of departure angles for each path in [rad]
%   EoA             The elevation of departure angles for each path in [rad]
%   xprmat          The complex-valued polarization transfer matrix for each sub-path
%   pin             The initial phases in [rad] for each sub-path
%
% Input:
%   usage
%   Controls the behavior of the method: If set to 0, all existing SSF parameters will be discarded
%   and the method exits. By default (1), new SSF parameters will be created, existing ones will be
%   replaced. It set to 2, existing SSF parameters will be reused and missing ones will be created.
%
%   show_warnings
%   If set to true (default), 'qd_builder.gen_ssf_parameters' performs a set of tests to determine
%   if all provided input variables are correctly initialized. This can be disabled by setting
%   'show_warnings = 0'.
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

% Parse input variables.
if ~exist( 'usage','var' ) || isempty( usage )
    usage = 1;
end
if ~exist( 'show_warnings','var' ) || isempty( show_warnings )
    show_warnings = true;
end

if numel(h_builder) > 1
    
    % Recursive call for all objects in the array
    sic = size( h_builder );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        gen_ssf_parameters( h_builder( i1,i2,i3,i4 ), usage, show_warnings );
    end
    
else
    
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Remove existing SSF parameters
    if usage == 0 || usage == 6
        h_builder.NumClusters = [];
        h_builder.NumSubPaths = [];
        h_builder.taus = [];
        h_builder.gain = [];
        h_builder.AoD = [];
        h_builder.AoA = [];
        h_builder.EoD = [];
        h_builder.EoA = [];
        h_builder.xprmat = [];
        h_builder.pin = [];
    end
    
    % Check if we need to do anything else
    if usage == 0 || h_builder.no_rx_positions == 0
        return
    end
    
    % Get required variables
    n_clusters      = h_builder.scenpar.NumClusters;
    n_subpaths      = h_builder.scenpar.NumSubPaths;
    n_freq          = numel( h_builder.simpar(1,1).center_frequency );
    n_mobiles       = h_builder.no_rx_positions;
    tx_pos          = h_builder.tx_position;
    rx_pos          = h_builder.rx_positions;
    use_3GPP_baseline = h_builder.simpar(1,1).use_3GPP_baseline;
    use_ground_reflection = logical( h_builder.scenpar.GR_enabled );
    speed_of_light  = qd_simulation_parameters.speed_of_light;
    r_DS            = h_builder.scenpar.r_DS;

    if use_ground_reflection
        gr_epsilon_r = h_builder.gr_epsilon_r.';
    end

    % Get the total number of paths
    if use_ground_reflection
        n_los_clusters = 2;
        n_nlos_clusters = n_clusters-2;
    else
        n_los_clusters = 1;
        n_nlos_clusters = n_clusters-1;
    end
    n_paths = n_los_clusters + n_nlos_clusters*n_subpaths;      % Total number of paths
    n_nlos_paths = n_nlos_clusters * h_builder.scenpar.NumSubPaths;    % Total number of NLOS paths
    
    o_nlos_clusters = ones(1,n_nlos_clusters);
    o_los_clusters = ones(1,n_los_clusters);
    o_clusters  = ones(1,n_clusters);
    o_subpaths = ones(1,n_subpaths);
    o_freq  = ones(1,n_freq);
    o_mobiles = ones(1,n_mobiles);
    o_nlos_paths = ones(1,n_nlos_paths);
    
    i_nlos_clusters = [ false(1,n_los_clusters), true(1,n_nlos_clusters) ];
    
    % Check Path-SOS random generator
    path_sos = h_builder.path_sos;
    if use_3GPP_baseline || isempty( path_sos )
        path_sos = [];
        
    elseif ~isa( path_sos, 'qd_sos' ) || size(path_sos,1) ~= n_nlos_clusters || size(path_sos,2) ~= 5
        error('QuaDRiGa:qd_builder:gen_ssf_parameters','path_sos has invalid format.');
        
    elseif show_warnings
        % Test if all generators are set to uniform distribution
        for n = 1 : size( path_sos, 1)
            for m = 1 : size( path_sos, 2)
                if ~strcmp( path_sos(n,m).distribution , 'Uniform' )
                    warning('QuaDRiGa:qd_builder:generate_paths:Uniform',...
                        'All random generators should be set to uniform distribution.');
                    warning ('off','QuaDRiGa:qd_builder:generate_paths:Uniform');
                end
            end
        end
        warning ('on','QuaDRiGa:qd_builder:generate_paths:Uniform');
        
        % Test if delays are reciprocal
        tmp = cat(3,path_sos(:,1).sos_phase);
        tmp = tmp(:,1,:) - tmp(:,2,:);
        if any(abs(tmp(:))>1e-6)
            warning('QuaDRiGa:qd_builder:generate_paths','Cluster delays are not reciprocal.');
        end
        
        % Test if azimuth angles are reciprocal
        tmp1 = cat(3,path_sos(:,2).sos_phase);
        tmp2 = cat(3,path_sos(:,3).sos_phase);
        tmp3 = tmp1(:,1,:) - tmp2(:,2,:);
        tmp4 = tmp2(:,1,:) - tmp1(:,2,:);
        if any( abs( [tmp3(:);tmp4(:)] ) >1e-6 )
            warning('QuaDRiGa:qd_builder:generate_paths','Azimuth angles are not reciprocal.');
        end
        
        % Test if elevation angles are reciprocal
        tmp1 = cat(3,path_sos(:,4).sos_phase);
        tmp2 = cat(3,path_sos(:,5).sos_phase);
        tmp3 = tmp1(:,1,:) - tmp2(:,2,:);
        tmp4 = tmp2(:,1,:) - tmp1(:,2,:);
        if any( abs( [tmp3(:);tmp4(:)] ) >1e-6 )
            warning('QuaDRiGa:qd_builder:generate_paths','Elevation angles are not reciprocal.');
        end
    end
    
    % Check XPR-SOS random generator
    xpr_sos = h_builder.xpr_sos;
    if use_3GPP_baseline || isempty( xpr_sos ) || n_nlos_clusters == 0
        xpr_sos = [];
        
    elseif ~isa( xpr_sos, 'qd_sos' ) || size(xpr_sos,1) ~= n_nlos_clusters ||...
            size(xpr_sos,2) ~= n_subpaths+1 || ...
            ( n_freq ~=1 && size(xpr_sos,3) ~= n_freq )
        error('QuaDRiGa:qd_builder:gen_ssf_parameters','xpr_sos has invalid format.');
        
    elseif show_warnings
        % Test if all generators are set to uniform distribution
        for n = 1 : size( xpr_sos, 1)
            for m = 1 : size( xpr_sos, 2)
                if ~strcmp( xpr_sos(n,m).distribution , 'Normal' )
                    warning('QuaDRiGa:qd_builder:gen_ssf_parameters:Normal',...
                        'All random generators should be set to uniform distribution.');
                    warning ('off','QuaDRiGa:qd_builder:generate_pol_rot:Normal');
                end
                tmp = xpr_sos(n,m).sos_phase;
                tmp = tmp(:,1,:) - tmp(:,2,:);
                if any(abs(tmp(:))>1e-6)
                    warning('QuaDRiGa:qd_builder:gen_ssf_parameters:reciprocal','XPR SOS generators are not reciprocal.');
                    warning ('off','QuaDRiGa:qd_builder:generate_pol_rot:reciprocal');
                end
            end
        end
        warning ('on','QuaDRiGa:qd_builder:gen_ssf_parameters:Normal');
        warning ('on','QuaDRiGa:qd_builder:gen_ssf_parameters:reciprocal');
    end
    
    if h_builder.dual_mobility
        tx_pos_SOS = tx_pos;
    else
        tx_pos_SOS = [];
    end
    
    % Set the number of sub-paths per cluster
    if use_ground_reflection
        NumSubPaths = [1,1,ones(1,n_nlos_clusters) * n_subpaths];
    else
        NumSubPaths = [1,ones(1,n_nlos_clusters) * n_subpaths];
    end
    
    % distances between BS and MT
    d_2d = hypot( tx_pos(1,:) - rx_pos(1,:), tx_pos(2,:) - rx_pos(2,:) );
    d_3d = sqrt( sum((rx_pos - tx_pos).^2,1) );
    d_2d( d_2d<1e-5 ) = 1e-5;
    d_3d( d_3d<1e-5 ) = 1e-5;
    
    % Calculate angles between BS and MT
    angles = zeros( 5,n_mobiles );
    angles(1,:) = atan2( rx_pos(2,:) - tx_pos(2,:) , rx_pos(1,:) - tx_pos(1,:) );           % Azimuth at BS
    angles(2,:) = mod( pi + angles(1,:) + 3.141592653589792, 2*pi ) - 3.141592653589792;    % Azimuth at MT
    angles(3,:) = atan( ( rx_pos(3,:) - tx_pos(3,:) ) ./ d_2d );                            % Elevation at BS
    angles(4,:) = -angles(3,:);                                                             % Elevation at MT
    angles(5,:) = -atan( ( rx_pos(3,:) + tx_pos(3,:) ) ./ d_2d );                           % Ground Reflection Elevation at BS and MT
    angles = angles.';
    
    % Get spread variables and convert from [deg] to [rad]
    DS  = h_builder.ds';
    ASD = h_builder.asD' * pi/180;
    ASA = h_builder.asA' * pi/180;
    ESD = h_builder.esD' * pi/180;
    ESA = h_builder.esA' * pi/180;
    KF  = h_builder.kf';
    
    % Process ground reflection parameters
    if use_ground_reflection
        % GR path length
        d_gf = sqrt( sum(([rx_pos(1:2,:);-rx_pos(3,:)] - tx_pos).^2,1) );
        
        % Incident angle of the GR
        theta_r = -angles(:,5);
        
        % Delay of the ground reflection relative to the LOS component
        tau_gr = ( d_gf-d_3d ).' / speed_of_light;
        
        % The reflection coefficient
        Z         = sqrt( gr_epsilon_r - (cos(theta_r)).^2 * o_freq );
        R_par     = (gr_epsilon_r .* (sin(theta_r)*o_freq) - Z) ./ (gr_epsilon_r .* (sin(theta_r)*o_freq) + Z);
        R_per     = ( sin(theta_r)*o_freq - Z) ./ ( sin(theta_r)*o_freq + Z);
        Rsq       = ( 0.5*(abs(R_par).^2 + abs(R_per).^2) );
    end
    
    % Generate paths
    if n_clusters == 1   % Only LOS component is present
        
        pow  = ones(n_mobiles,1,n_freq);
        taus = zeros(n_mobiles,1);
        AoD  = angles(:,1);
        AoA  = angles(:,2);
        EoD  = angles(:,3);
        EoA  = angles(:,4);
        
    elseif use_ground_reflection && n_clusters == 2   % Only LOS and GR component
        
        pow  = zeros( n_mobiles, n_clusters, n_freq );
        pow( :,1,: ) = (1-Rsq);
        pow( :,2,: ) = Rsq;
        taus = [ zeros(n_mobiles,1), tau_gr ];
        AoD  = angles(:,1) * [1 1];
        AoA  = angles(:,2) * [1 1];
        EoD  = [ angles(:,3), angles(:,5) ];
        EoA  = [ angles(:,4), angles(:,5) ];
        
    elseif use_3GPP_baseline % 3GPP Baseline method for NLOS path generation
        
        if n_freq > 1
            error('QuaDRiGa:qd_builder:gen_ssf_parameters',...
                '3GPP Baseline method does not support multiple frequencies.');
        end
        
        % If we have a ground reflection, we perform the calculations as if the GR was not there,
        % but with 1 cluster less
        if use_ground_reflection
            n_clusters_baseline = n_clusters - 1;
            o_clusters_baseline = ones( 1,n_clusters_baseline );
        else
            n_clusters_baseline = n_clusters;
            o_clusters_baseline = o_clusters;
        end
            
        % Alternative method leading to correct calibration results
        alt_AS_mapping = true;
        
        % Step 5: Generate cluster delays
        % Generate initial delays following an exponential distribution; eq. (7.5-1)
        X = rand( n_mobiles,n_clusters_baseline );
        taus = -r_DS * DS(:,o_clusters_baseline) .* log( X );
        
        % Subtract the minimum delay and sort the normalized delays to ascending order; eq. (7.5-2)
        taus = sort( taus - min(taus,[],2)*o_clusters_baseline , 2 );
        
        % Additional scaling of delays is required to compensate for the effect of LOS peak; eq. (7.5-3)
        KF_dB = 10*log10( KF );  %#ok!
        
        % Scaled delays (not to be used in cluster power generation); eq. (7.5-4)
        % C_tau = 0.7705 - 0.0433*KF_dB + 0.0002*KF_dB.^2 + 0.000017*KF_dB.^3;
        % C_tau( C_tau > 1 ) = 1;
        % taus_LOS = taus ./ C_tau(:,o_los_clusters);
        
        % Step 6: Generate cluster powers
        % Cluster powers are calculated assuming a single slope exponential power delay profile; eq. (7.5-5)
        LNS_ksi = h_builder.scenpar.LNS_ksi;
        Z = randn( n_mobiles,n_clusters_baseline )*LNS_ksi;
        pow = exp( -taus .* (((r_DS-1)./( r_DS*DS ))*o_clusters_baseline) ) .* 10.^(Z/10);
        
        % Power of the single LOS ray; eq. (7.5-7)
        pow(:,1) = KF./(KF+1);
        
        % Normalize the cluster powers so that the sum of all cluster powers is equal to one; eq. (7.5-8)
        pow(:,2:end) = (1./(KF+1))*o_nlos_clusters .* pow(:,2:end) ./ (sum( pow(:,2:end),2 )*o_nlos_clusters);
        
        if alt_AS_mapping % Alternative method leading to correct calibration results
            
            AoA = 2*rand(size(pow))-1;                          % Random angles in range -57 ... 57 deg
            AoA = AoA .* ceil( -pow+(max( pow,[],2)*o_clusters_baseline) );      % Set angle of strongest path to 0 deg
            as  = qf.calc_angular_spreads( AoA , pow );         % Calculate AS
            AoA = AoA .* (ASA./as * o_clusters_baseline);
            
            AoA_LOS = angle( exp( 1j*angles(:,2) ) );           % Apply LOS angle
            AoA = AoA - ( AoA(:,1) - AoA_LOS )*o_clusters_baseline;
            
            AoD = 2*rand(size(pow))-1;                          % Random angles in range -57 ... 57 deg
            AoD = AoD .* ceil( -pow+(max( pow,[],2)*o_clusters_baseline) );      % Set angle of strongest path to 0 deg
            as  = qf.calc_angular_spreads( AoD , pow );         % Calculate AS
            AoD = AoD .* (ASD./as * o_clusters_baseline);
            
            AoD_LOS = angle( exp( 1j*angles(:,1) ) );           % Apply LOS angle
            AoD = AoD - ( AoD(:,1) - AoD_LOS )*o_clusters_baseline;
            
        else % Strict method as defined in the TR
            
            % Step 7: Generate arrival angles and departure angles for both azimuth and elevation.
            % C_phi_NLOS is defined as a scaling factor related to the total number of clusters.
            % The "-1" accounts for the LOS component which is always present in QuaDRiGa.
            if n_clusters_baseline <= 5 %#ok!
                C_phi_NLOS = 0.779;
            elseif n_clusters_baseline >= 21
                C_phi_NLOS = 1.289;
            else
                L_phi_NLOS = [     4,     5,     8,    10,    11,    12,    14,    15,    16,    19,    20 ];
                C_phi_NLOS = [ 0.779, 0.860, 1.018, 1.090, 1.123, 1.146, 1.190, 1.211, 1.226, 1.273, 1.289 ];
                C_phi_NLOS = qf.interp( L_phi_NLOS, 0, C_phi_NLOS, n_clusters_baseline-1 );
            end
            
            % C_phi_LOS depends on the K-Factor, eq. (7.5-10)
            C_phi_LOS = 1.1035 - 0.028*KF_dB - 0.002*KF_dB.^2 + 0.0001*KF_dB.^3;
            C_phi_LOS( KF_dB < 0 & C_phi_LOS < 1 ) = 1;
            C_phi = C_phi_NLOS * C_phi_LOS;
            
            % The AOAs are determined by applying the inverse Gaussian function with input parameters pow
            % and RMS angle spread ASA, eq. (7.5-9)
            AoA = 2*(ASA/1.4)*o_clusters_baseline .* sqrt( -log( pow ./ (max( pow,[],2)*o_clusters_baseline) ) ) ./ (C_phi*o_clusters_baseline);
            
            % Assign positive or negative sign and introduce random variation, enforce the first cluster to
            % the LOS direction, eq. (7.5-12)
            X = 2*(randi(2,n_mobiles,n_clusters_baseline)-1.5);
            Y = randn( n_mobiles,n_clusters_baseline ).*((ASA/7)*o_clusters_baseline);
            AoA_LOS = angle( exp( 1j*angles(:,2) ) );
            AoA = ( X.*AoA + Y ) - ( X(:,1).*AoA(:,1) + Y(:,1) - AoA_LOS )*o_clusters_baseline;
            
            % The generation of AOD follows a procedure similar to AOA
            AoD = 2*(ASD/1.4)*o_clusters_baseline .* sqrt( -log( pow ./ (max( pow,[],2)*o_clusters_baseline) ) ) ./ (C_phi*o_clusters_baseline);
            X = 2*(randi(2,n_mobiles,n_clusters_baseline)-1.5);
            Y = randn( n_mobiles,n_clusters_baseline ).*((ASD/7)*o_clusters_baseline);
            AoD_LOS = angle( exp( 1j*angles(:,1) ) );
            AoD = ( X.*AoD + Y ) - ( X(:,1).*AoD(:,1) + Y(:,1) - AoD_LOS )*o_clusters_baseline;
        end
        
        % Wrap AoAs around the unit circle
        AoA = angle( exp( 1j*AoA ) );
        AoD = angle( exp( 1j*AoD ) );
        
        if alt_AS_mapping % Alternative method leading to correct calibration results
            
            EoA = 2*rand(size(pow))-1;                          % Random angles in range -57 ... 57 deg
            EoA = EoA .* ceil( -pow+(max( pow,[],2)*o_clusters_baseline) );      % Set angle of strongest path to 0 deg
            as  = qf.calc_angular_spreads( EoA , pow );         % Calculate AS
            EoA = EoA .* (ESA./as * o_clusters_baseline);
            
            EoA_LOS = angle( exp( 1j*angles(:,4) ) );
            EoA = EoA - ( EoA(:,1) - EoA_LOS )*o_clusters_baseline;
            
            EoD = 2*rand(size(pow))-1;                          % Random angles in range -57 ... 57 deg
            EoD = EoD .* ceil( -pow+(max( pow,[],2)*o_clusters_baseline) );      % Set angle of strongest path to 0 deg
            as  = qf.calc_angular_spreads( EoD , pow );         % Calculate AS
            EoD = EoD .* (ESD./as * o_clusters_baseline);
            
            EoD_LOS = angle( exp( 1j*angles(:,3) ) );
            EoD = EoD - ( EoD(:,1) - EoD_LOS )*o_clusters_baseline;
            
        else % Strict method as defined in the TR
            
            % C_phi_NLOS is defined as a scaling factor related to the total number of clusters.
            % The "-1" accounts for the LOS component which is always present in QuaDRiGa.
            if n_clusters_baseline <= 9 %#ok!
                C_phi_NLOS = 0.889;
            elseif n_clusters_baseline >= 21
                C_phi_NLOS = 1.178;
            else
                L_phi_NLOS = [     8,    10,    11,    12,     15,    19,    20 ];
                C_phi_NLOS = [ 0.889, 0.957, 1.031, 1.104, 1.1088, 1.184, 1.178 ];
                C_phi_NLOS = qf.interp( L_phi_NLOS, 0, C_phi_NLOS, n_clusters_baseline-1 );
            end
            
            % C_phi_LOS depends on the K-Factor, eq. (7.5-15)
            C_phi_LOS = 1.3086 + 0.0339*KF_dB - 0.0077*KF_dB.^2 + 0.0002*KF_dB.^3;
            C_phi_LOS( KF_dB < 0 & C_phi_LOS < 1 ) = 1;
            C_phi = C_phi_NLOS * C_phi_LOS;
            
            % The generation of ZOA assumes that the composite PAS in the zenith dimension of all clusters
            % is Laplacian. The ZOAs are determined by applying the inverse Laplacian function, eq. (7.5-14)
            EoA = -(ESA*o_clusters_baseline) .* log( pow ./ (max( pow,[],2)*o_clusters_baseline) ) ./ (C_phi*o_clusters_baseline);
            
            % Assign positive or negative sign and introduce random variation, enforce the first cluster to
            % the LOS direction, eq. (7.5-12)
            X = 2*(randi(2,n_mobiles,n_clusters_baseline)-1.5);
            Y = randn( n_mobiles,n_clusters_baseline ).*((ESA/7)*o_clusters_baseline);
            EoA_LOS = angle( exp( 1j*angles(:,4) ) );
            EoA = ( X.*EoA + Y ) - ( X(:,1).*EoA(:,1) + Y(:,1) - EoA_LOS )*o_clusters_baseline;
            
            EoD = -(ESD*o_clusters_baseline) .* log( pow ./ (max( pow,[],2)*o_clusters_baseline) ) ./ (C_phi*o_clusters_baseline);
            X = 2*(randi(2,n_mobiles,n_clusters_baseline)-1.5);
            Y = randn( n_mobiles,n_clusters_baseline ).*((ESD/7)*o_clusters_baseline);
            EoD_LOS = angle( exp( 1j*angles(:,3) ) );
            EoD = ( X.*EoD + Y ) - ( X(:,1).*EoD(:,1) + Y(:,1) - EoD_LOS )*o_clusters_baseline;
        end
        
        % If ZOA is is wrapped within [0, 360°], and ZOA is in [ 180° , 360° ], then set to ( 360° - ZOA )
        % QuaDRiGa uses EOA. The wrapping is equivalent to inverting the real part.
        C = exp( 1j*EoA );
        ii = real(C) < 0;
        C(ii) = -real(C(ii)) + 1j*imag(C(ii));
        EoA = angle( C );
        
        C = exp( 1j*EoD );
        ii = real(C) < 0;
        C(ii) = -real(C(ii)) + 1j*imag(C(ii));
        EoD = angle( C );
        
        % If we have a ground reflection, we now add its deterministic values to the path list
        if use_ground_reflection
            p1 = pow( :,1 );
            pow = [ zeros(n_mobiles,2), pow(:,2:end) ];
            pow( :,1 ) = (1-Rsq).*p1;
            pow( :,2 ) = Rsq.*p1;
            taus = [ zeros(n_mobiles,1), tau_gr, taus(:,2:end) ];
            AoD  = [ repmat( angles(:,1),1,2), AoD(:,2:end) ];
            AoA  = [ repmat( angles(:,2),1,2), AoA(:,2:end) ];
            EoD  = [ angles(:,3), angles(:,5), EoD(:,2:end) ];
            EoA  = [ angles(:,4), angles(:,5), EoA(:,2:end) ];
        end
        
    else % QuaDRiGa method for NLOS path generation
        
        % Placeholder for the output variables
        pow  = zeros( n_mobiles, n_clusters, n_freq );
        taus = zeros( n_mobiles, n_clusters );
        
        % Generate initial delays following an exponential distribution
        if isempty( path_sos )
            randC = rand( n_mobiles,n_nlos_clusters );
        else
            randC = val( path_sos( 1:n_nlos_clusters,1 ), rx_pos,tx_pos_SOS ).';  % Uniform distribution
        end
        taus(:,i_nlos_clusters) = -log( randC );
        
        % Add ground reflection delay and calculate split the LOS power in a LOS and GR part
        if use_ground_reflection
            taus(:,2) = tau_gr;
            Pf = [ (1-mean(Rsq,2)) mean(Rsq,2) ];
        else % only LOS
            Pf = o_mobiles';
        end
        
        % Generate correlated random variables for the angles
        if n_nlos_clusters > 2
            if isempty( path_sos )
                randI = rand( n_mobiles,n_nlos_clusters,4 );
            else
                % Generate Normal-distribute spatially correlated random variables
                randI = zeros( n_mobiles,n_nlos_clusters,4 );
                for i_ang = 1:4                                             % Uniform distribution
                    randI(:,:,i_ang) = val( path_sos( 1:n_nlos_clusters,i_ang+1 ), rx_pos,tx_pos_SOS ).';
                end
            end
        end
        
        % Generate angles for the average angular spreads
        mu_fixed = zeros(n_mobiles,4);  % The fixed angle for LOS and GR
        path_angles = zeros( n_mobiles,n_clusters,4 );
        for i_ang = 1 : 4
            
            % The LOS angle and the angle for the GR are deterministic and know from the BS and MT
            % positions. They are assembled before generating the NLOS angles. NLOS angles are
            % distributed around the average fixed angle.
            
            if use_ground_reflection
                if i_ang == 3 % EoD
                    ang_fixed = [ angles(:,3) angles(:,5) ];
                elseif i_ang == 4 % EoA
                    ang_fixed = [ angles(:,4) angles(:,5) ];
                else % AoD and AoA
                    ang_fixed = angles(:,i_ang) * [1 1];
                end
            else % LOS only
                ang_fixed = angles( :,i_ang );
            end
            [ ~, mu_fixed(:,i_ang) ] = qf.calc_angular_spreads( ang_fixed, Pf, 0 );
            
            if i_ang == 4 && ~use_ground_reflection    % Do not apply NLOS EaA rotation (e.g. for satellites)
                % We must maintain spatial consistency and channel reciprocity in dual-mobility
                % scenarios. Hence, if Tx and Rx are at the same height, we must apply the elevation
                % angle rotation. If the height difference is large (e.g. for a satellite in the sky),
                % the elevation angle rotation can be disabled. However, we must avoid decision
                % boundaries to maintain spatial consistency. This is done by a smooth transition.
                
                i_kf = all(KF<1e-9,2);                      % KF must be < -90 dB (no LOS component)
                d_height = tx_pos(3,:) - rx_pos(3,:);       % Height difference between Tx and Rx
                
                % Calculate the scaling weights
                w_height = (abs(d_height.')-10)/10;         % Smooth transitions from 10 to 20 meters
                w_height(w_height<0) = 0;                   % EoA rotation from -10 to 10 meters
                w_height(w_height>1) = 1;                   % No EoA rotation for more then 20 meters difference
                w_height = cos(pi/2*w_height).^2;           % Continuous transition function
                w_height(~i_kf) = 1;
                
                % Apply scaling
                if any( i_kf )
                    mu_fixed( i_kf ,4 ) = w_height( i_kf ) .* mu_fixed( i_kf ,4 );
                end
            end
            
            % Scale angles within +/- pi
            if n_nlos_clusters == 1
                randC = ones(n_mobiles,1) / sqrt(2);    % 40.5 degree
            elseif n_nlos_clusters == 2
                randC = ones(n_mobiles,1) * [-1 1] / sqrt(2);       % +/- 40.5 degree
            else % n_nlos_clusters > 2
                % Read variables from previously generated random numbers
                randC = (randI(:,:,i_ang)-0.5)*pi;
            end
            
            % Set the fixed angles (LOS and GR)
            path_angles(:,:,i_ang) = [ ang_fixed - mu_fixed(:,o_los_clusters*i_ang), randC ];
        end
        
        % Generate initial path powers
        % Calculate DS scaling coefficient
        if n_freq > 1
            gDS = DS ./ (( max(DS,[],2) + min(DS,[],2) ) *o_freq);
            gDS( gDS < 0.15 ) = 0.15;
            gDS( gDS > 0.85 ) = 0.85;
            gDS = -1.5 * log( 1.2 * gDS - 0.15 );
            
            % If we have identical DS values for multiple frequencies, we use the scaling factor
            ind = abs( max(DS,[],2) ./ min(DS,[],2) - 1 ) < 1e-6;
            gDS(ind,:) = r_DS-1;
        else
            gDS = r_DS(o_mobiles,1) - 1;
        end
        
        % Calculate ASD scaling coefficient
        if n_freq > 1
            gASD = 0.75 * ASD./(max(ASD,[],2)*o_freq);
            gASD( gASD < 0.25 ) = 0.25;
            gASD = -2.2 * log( 1.5 * gASD - 0.35 );
        else
            gASD = 0.56*o_mobiles';
        end
        
        % Calculate ASA scaling coefficient
        if n_freq > 1
            gASA = 0.75 * ASA./(max(ASA,[],2)*o_freq);
            gASA( gASA < 0.25 ) = 0.25;
            gASA = -2.2 * log( 1.5 * gASA - 0.35 );
        else
            gASA = 0.56*o_mobiles';
        end
        
        % Calculate ESD scaling coefficient
        if n_freq > 1
            gESD = 0.75 * ESD./(max(ESD,[],2)*o_freq);
            gESD( gESD < 0.25 ) = 0.25;
            gESD = -3.4 * log( 1.2 * gESD - 0.1 );
        else
            gESD = 0.76*o_mobiles';
        end
        
        % Calculate ESA scaling coefficient
        if n_freq > 1
            gESA = 0.75 * ESA./(max(ESA,[],2)*o_freq);
            gESA( gESA < 0.25 ) = 0.25;
            gESA = -3.4 * log( 1.2 * gESA - 0.1 );
        else
            gESA = 0.76*o_mobiles';
        end
        
        for i_freq = 1 : n_freq
            % NLOS powers
            pow(:,:,i_freq) = exp( -taus .* gDS(:,o_clusters*i_freq) ) .* ...
                exp( -path_angles(:,:,1).^2 .* gASD(:,o_clusters*i_freq) ) .* ...
                exp( -path_angles(:,:,2).^2 .* gASA(:,o_clusters*i_freq) ) .* ...
                exp( -abs(path_angles(:,:,3)) .* gESD(:,o_clusters*i_freq) ) .* ...
                exp( -abs(path_angles(:,:,4)) .* gESA(:,o_clusters*i_freq) );
            
            % Split LOS power into LOS and GR power
            if use_ground_reflection
                Pf = [ 1-Rsq(:,i_freq), Rsq(:,i_freq) ];
            else
                Pf = o_mobiles';
            end
            
            % Apply K-Factor
            Pf = Pf .* ((KF(:,i_freq) ./ ( 1 + KF(:,i_freq) ))*o_los_clusters);
            Pn = 1-sum(Pf,2);
            
            pow(:,1:n_los_clusters,i_freq) = 0;
            pow(:,:,i_freq)    = pow(:,:,i_freq) .* (Pn ./ sum(pow(:,:,i_freq),2) * o_clusters);
            pow(:,1:n_los_clusters,i_freq) = Pf;
        end
        
        % Applying the Delay Spread
        % Calculate the delay scaling factor (has analytic solution)
        s = zeros( n_mobiles,n_freq );   % Scaling factor
        for i_freq = 1 : n_freq
            if use_ground_reflection
                a = DS(:,i_freq).^2;
                b = pow(:,2,i_freq) .* tau_gr.^2;
                c = sum( pow(:,3:end,i_freq) .* taus(:,3:end).^2 , 2 );
                d = pow(:,2,i_freq) .* tau_gr;
                e = sum( pow(:,3:end,i_freq) .* taus(:,3:end) , 2 );
                tmp = a.*c - a.*e.^2 - b.*c + b.*e.^2 + c.*d.^2;
                tmp( tmp<0 ) = 0;   % Minimum DS is given by GR, cannot get smaller
                s(:,i_freq) = ( sqrt( tmp ) + d.*e ) ./ (c-e.^2);
            else
                a = DS(:,i_freq).^2;
                b = sum( pow(:,2:end,i_freq) .* taus(:,2:end) , 2 );
                c = sum( pow(:,2:end,i_freq) .* taus(:,2:end).^2 , 2 );
                s(:,i_freq) = sqrt( a ./ (c-b.^2) );
            end
        end
        if n_freq > 1 % Average scaling coefficient for multiple frequencies
            s = mean(s,2);
        end
        taus(:,i_nlos_clusters) = taus(:,i_nlos_clusters) .* ( s*o_nlos_clusters );  % Scale the delays
        
        
        % Applying the Angular Spread
        % Combine all angular spread values to a single variable
        AS = cat( 3, ASD,ASA,ESD,ESA );
        
        for i_ang = 1 : 4
            
            s = zeros( n_mobiles,n_freq ); % Scaling factor
            for i_freq = 1 : n_freq
                
                % Current angular spread
                ang  = path_angles(:,:,i_ang);
                as   = qf.calc_angular_spreads( ang , pow(:,:,i_freq), 1 );
                
                % The following iterative optimization tries to find the optimal STD of the NLOS angles
                % such that the given angular spread is reached. This iteration is needed to correctly
                % incorporate the ground reflection.
                
                % Calculate the scaling coefficient as start values for an iterative optimization
                sL = 10*log10( AS(:,i_freq,i_ang) ./ as  );
                
                lp = 1;
                upd  = true( n_mobiles,1 );         % The values that need to be updated
                while any( upd ) && lp < 10
                    % Apply the scaling coefficients to the angles and update the angles and the angular spread
                    ang( upd, i_nlos_clusters ) = (10.^(0.1*sL(upd))*o_nlos_clusters) .*...
                        path_angles( upd, i_nlos_clusters, i_ang );
                    as = qf.calc_angular_spreads( ang(upd,:), pow(upd,:,i_freq),0 );
                    
                    % Calculate the new scaling coefficient
                    sN = zeros( size( sL ) );
                    sN(upd)  = 10*log10( AS(upd,i_freq,i_ang) ./ as  );
                    
                    upd = abs( sN ) > 1e-4;
                    sL(upd) = sL(upd) + sN(upd);
                    lp = lp + 1;
                end
                
                s(:,i_freq) = 10.^( 0.1*sL );
            end
            
            if n_freq > 1 % Average scaling coefficient for multiple frequencies
                s = mean(s,2);
            end
            
            % The initial angles range from -pi/2 to pi/2, the initial angular spread is 30°
            % Due to the wrapping of the angles, they cannot exceed
            if i_ang < 2.5 % Azimuth angles
                s( s > 3 ) = 3;
            else % Elevation angles
                s( s > 1.5 ) = 1.5;
            end
            
            % Scale NLOS angles
            path_angles(:,i_nlos_clusters,i_ang) = ( s*o_nlos_clusters ) .* path_angles(:,i_nlos_clusters,i_ang);
        end
        
        % Elevation angles outside the +/- pi/2 range would change the azimuth angles as well
        % We therefore mirror the elevation angles at the poles
        tmp = path_angles(:,i_nlos_clusters,[3,4]);
        ind = tmp > pi/2;
        tmp(ind) = pi-tmp(ind);
        ind = tmp < -pi/2;
        tmp(ind) = -pi-tmp(ind);
        path_angles(:,i_nlos_clusters,[3,4]) = tmp;
        
        % Apply the LOS angles
        % Wrap angles around the unit circle (does not change AS)
        path_angles  = mod( real(path_angles) + 3.141592653589792, 2*pi) - 3.141592653589792;
        
        % Transform departure angles to Cartesian coordinates
        path_angles = permute( path_angles, [3,2,1] );
        C = zeros( 3,n_clusters,n_mobiles );
        C(1,:,:) = cos( path_angles(1,:,:) ) .* cos( path_angles(3,:,:) );
        C(2,:,:) = sin( path_angles(1,:,:) ) .* cos( path_angles(3,:,:) );
        C(3,:,:) = sin( path_angles(3,:,:) );
        
        % Build rotation matrix
        R = zeros( 3,3,n_mobiles );
        ce = cos( mu_fixed(:,3) );
        se = sin( mu_fixed(:,3) );
        ca = cos( mu_fixed(:,1) );
        sa = sin( mu_fixed(:,1) );
        R(1,1,:) = ce.*ca;
        R(1,2,:) = -sa;
        R(1,3,:) = -se.*ca;
        R(2,1,:) = ce.*sa;
        R(2,2,:) = ca;
        R(2,3,:) = -se.*sa;
        R(3,1,:) = se;
        R(3,2,:) = 0;
        R(3,3,:) = ce;
        
        % Apply rotation in Cartesian coordinates
        for n = 1 : n_mobiles
            C(:,:,n) = R(:,:,n) * C(:,:,n);
        end
        
        % Calculate rotated departure angles
        hypotxy = hypot( C(1,:,:),C(2,:,:) );
        EoD = atan2(C(3,:,:),hypotxy);
        AoD = atan2(C(2,:,:),C(1,:,:));
        EoD = permute( EoD, [3,2,1] );
        AoD = permute( AoD, [3,2,1] );
        
        % Transform arrival angles to Cartesian coordinates
        C = zeros( 3,n_clusters,n_mobiles );
        C(1,:,:) = cos( path_angles(2,:,:) ) .* cos( path_angles(4,:,:) );
        C(2,:,:) = sin( path_angles(2,:,:) ) .* cos( path_angles(4,:,:) );
        C(3,:,:) = sin( path_angles(4,:,:) );
        
        % Build rotation matrix
        R = zeros( 3,3,n_mobiles );
        ce = cos( mu_fixed(:,4) );
        se = sin( mu_fixed(:,4) );
        ca = cos( mu_fixed(:,2) );
        sa = sin( mu_fixed(:,2) );
        R(1,1,:) = ce.*ca;
        R(1,2,:) = -sa;
        R(1,3,:) = -se.*ca;
        R(2,1,:) = ce.*sa;
        R(2,2,:) = ca;
        R(2,3,:) = -se.*sa;
        R(3,1,:) = se;
        R(3,2,:) = 0;
        R(3,3,:) = ce;
        
        % Apply rotation in Cartesian coordinates
        for n = 1 : n_mobiles
            C(:,:,n) = R(:,:,n) * C(:,:,n);
        end
        
        % Calculate rotated departure angles
        hypotxy = hypot( C(1,:,:),C(2,:,:) );
        EoA = atan2(C(3,:,:),hypotxy);
        AoA = atan2(C(2,:,:),C(1,:,:));
        EoA = permute( EoA, [3,2,1] );
        AoA = permute( AoA, [3,2,1] );
    end
    
    % Apply absolute time-of-arrival offset to NLOS delays
    if ~isempty( h_builder.absTOA_offset )
        min_nlos_delay = min( taus( :, i_nlos_clusters ),[],2);
        taus( :, i_nlos_clusters ) = taus( :, i_nlos_clusters ) - ...
            repmat( min_nlos_delay, 1,n_nlos_clusters ) + ...
            repmat( h_builder.absTOA_offset.', 1, n_nlos_clusters );
    end
    
    % If we do not use spatial consistency, we sort the clusters by powers in descending
    % order. A subsequent call of "generate_subpaths" then splits only the first two
    % clusters into sub-clusters. If spatial consistency is used, all clusters are split
    % and no sorting operations are allowed.
    if use_3GPP_baseline || strcmp( h_builder.simpar(1,1).autocorrelation_function, 'Disable' ) || ...
            h_builder.scenpar.SC_lambda == 0
        
        tmp = sum( pow,3 );     % Sum over all frequencies
        iNLOS =  n_los_clusters + 1 : n_clusters;
        [ ~,iS ] = sort(  tmp( :,iNLOS ), 2, 'descend' );
        iS = [ repmat( 1:n_los_clusters, n_mobiles, 1 ), iS + n_los_clusters  ];
        for n = 1 : n_mobiles
            pow( n,:,: )= pow(n,iS(n,:),:);
            taus( n,: ) = taus( n,iS(n,:));
            AoD( n,: )  = AoD( n,iS(n,:) );
            AoA( n,: )  = AoA( n,iS(n,:) );
            EoD( n,: )  = EoD( n,iS(n,:) );
            EoA( n,: )  = EoA( n,iS(n,:) );
        end
    end
    
    % Get the per-cluster DS in seconds (it is given in ns in the config files)
    if h_builder.scenpar.PerClusterDS_gamma ~= 0
        f_GHz = h_builder.simpar(1,1).center_frequency / 1e9;
        PerClusterDS = h_builder.scenpar.PerClusterDS_gamma * log10( f_GHz ) + h_builder.scenpar.PerClusterDS;
        PerClusterDS = max( PerClusterDS, h_builder.scenpar.PerClusterDS_min ) / 1e9;
    else
        PerClusterDS = h_builder.scenpar.PerClusterDS / 1e9;
    end
    
    % Apply PerClusterDS and AS
    switch h_builder.scenpar.SubpathMethod
        
        case {'legacy','Laplacian'}
            % This is the legacy subpath generation method for bandwidths below 100 MHz.
            % See: 3GPP TR 38.901 V16.1.0 (2019-12), pp39
            
            % Determine if clusters will be split in sub-clusters with different delays
            % This depends on 3 conditions:
            %   1. The per-cluster delay spread is set to values > 0 in the scenario definition
            %   2. The number of sub-paths is set to 20
            %   3. There is at least 1 NLOS path (excluding ground reflection)
            use_cluster_DS = h_builder.scenpar.PerClusterDS ~= 0 &...
                h_builder.scenpar.NumSubPaths == 20 &...
                n_clusters > n_los_clusters;
            
            % Plot a warning if NumSubPaths is not 20
            if h_builder.scenpar.PerClusterDS ~= 0 && h_builder.scenpar.NumSubPaths ~= 20 && show_warnings
                warning('QuaDRiGa:builder:generate_subpaths:wrongNumSubPaths',...
                    'Number of subpaths must be 20 for legacy cluster splitting. PerClusterDS was not applied.');
                
            end
            
            % According the 3GPP 38.901 16.1.0, Table 7.5-5, p40, the 2 strongest clusters should
            % be split in sub-clusters with a delay-offset that is determined by the
            % "PerClusterDS". However, this would break the spatial consistency since the strongest
            % clusters change with locations. To solve this, ALL clusters are split into sub-clusters.
            
            no_SC = strcmp( h_builder.simpar(1,1).autocorrelation_function, 'Disable' ) | ...
                h_builder.scenpar.SC_lambda == 0 | use_3GPP_baseline ;
            
            if use_cluster_DS
                
                % Get the number of clusters that need to be split
                if no_SC         % Split 2 first clusters
                    nSPL = min( n_clusters-n_los_clusters , 2 );
                else
                    nSPL = n_clusters-n_los_clusters;
                end
                splt = sort( [ 1:n_los_clusters, (1:nSPL)+n_los_clusters, (1:nSPL)+n_los_clusters,...
                    (1:nSPL)+n_los_clusters, n_los_clusters+nSPL+1:n_clusters ] );
                
                if numel( PerClusterDS ) > 1 % Frequency-dependent !!!
                    taus = taus( :,splt, ones(1,n_freq) );
                else
                    taus = taus( :,splt );
                end
                
                pow  = pow( :,splt,: );
                nsplt = [ 10 , 6 , 4 ];
                NumSubPaths_new = ones(1,n_los_clusters);
                for n = 0:nSPL-1
                    NumSubPaths_new = [ NumSubPaths_new, nsplt ]; %#ok!
                    for m = 1:3
                        ii = n_los_clusters+n*3+m;
                        if numel( PerClusterDS ) > 1 % Frequency-dependent !!!
                            for iF = 1:n_freq
                                taus( :,ii,iF ) = taus( :,ii,iF ) + PerClusterDS(iF)*1.28*(m-1);
                            end
                        else
                            taus( :,ii ) = taus( :,ii ) + PerClusterDS*1.28*(m-1);
                        end
                        pow( :,ii,: )  = pow( :,ii,: ).*nsplt(m)/20;
                    end
                end
                NumSubPaths = [ NumSubPaths_new, NumSubPaths( n_los_clusters+nSPL+1:n_clusters ) ];
                
                AoD = AoD( :,splt );
                AoA = AoA( :,splt );
                EoD = EoD( :,splt );
                EoA = EoA( :,splt );
                n_clusters = numel( splt );
            end
            
        case 'mmMAGIC'
            % This is the mmMAGIC subpath generation method.
            % See: H2020-ICT-671650-mmMAGIC/D2.2, pp86, Section 4.5.3 Mapping of paths to sub-paths
            
            % Delay scaling coefficient
            r_DS = h_builder.scenpar.r_DS;
            
            % Rename existing variables
            Cpow  = pow;
            Ctaus = taus;
            CAoD  = AoD;
            CAoA  = AoA;
            CEoD  = EoD;
            CEoA  = EoA;
            
            % Reserve Memory
            pow  = zeros( n_mobiles, n_paths, n_freq );
            if size( PerClusterDS,2) == 1
                taus = zeros( n_mobiles, n_paths );
            else
                taus = zeros( n_mobiles, n_paths, n_freq );
            end
            AoD = zeros( n_mobiles, n_paths );
            AoA = zeros( n_mobiles, n_paths );
            EoD = zeros( n_mobiles, n_paths );
            EoA = zeros( n_mobiles, n_paths );
            
            % Copy LOS and GR data
            pow(:,1:n_los_clusters,:) = Cpow(:,1:n_los_clusters,:);
            for n = 1:size( PerClusterDS,2)
                taus(:,1:n_los_clusters,n) = Ctaus(:,1:n_los_clusters);
            end
            AoD(:,1:n_los_clusters) = CAoD(:,1:n_los_clusters);
            AoA(:,1:n_los_clusters) = CAoA(:,1:n_los_clusters);
            EoD(:,1:n_los_clusters) = CEoD(:,1:n_los_clusters);
            EoA(:,1:n_los_clusters) = CEoA(:,1:n_los_clusters);
            
            st = n_los_clusters + 1;
            for n = n_los_clusters+1 : n_clusters
                M = NumSubPaths(n);
                oM = ones( 1,M );
                
                % Generate spatially correlated random numbers for the delay offsets
                if isempty( h_builder.clst_dl_sos )
                    randC = rand( n_mobiles, M );
                else
                    randC = val( h_builder.clst_dl_sos( n-n_los_clusters,: ), rx_pos, tx_pos_SOS ).';
                end
                
                % Generate delay offsets
                powers = zeros( n_mobiles, M, size( PerClusterDS,2) );
                for iF = 1 : size( PerClusterDS,2)
                    delays = -r_DS * PerClusterDS(1,iF) * log( randC );                     % Calc delays
                    delays = delays - min( delays,[],2 ) * oM;                              % Normalize delays
                    if PerClusterDS(1,iF) > 1e-10
                        powers(:,:,iF) = exp( -delays * (r_DS-1)/(r_DS*PerClusterDS(1,iF)) );   % Calc powers
                    else
                        powers(:,:,iF) = ones( n_mobiles,M );
                    end
                    powers(:,:,iF) = powers(:,:,iF) ./ ( sum( powers(:,:,iF) , 2 ) * oM );  % Normalize powers
                    ds = sqrt( sum(powers(:,:,iF).*delays.^2,2) - sum( powers(:,:,iF).*delays,2).^2 );
                    if PerClusterDS(1,iF) > 1e-10
                        delays = (PerClusterDS(1,iF)./ds) * oM .*  delays;                  % Scale delays
                    end
                    taus(:,st:st+M-1,iF) = delays + Ctaus(:,n)*oM;                 % Add cluster delay
                end
                
                % Duplicate powers if cluster-DS is not freq.-dependent
                if size( PerClusterDS,2) < n_freq
                    powers = powers(:,:,ones(1,n_freq));
                end
                
                % Apply cluster powers
                for iF = 1:n_freq
                    pow(:,st:st+M-1,iF) = powers(:,:,iF) .* (Cpow(:,n,iF)*oM);
                end
                
                % Generate angle offsets (Laplacian angular offsets)
                ao = 1:floor( (M+0.1)/2 );
                if ao(end) > M/2-0.1
                    ao = [ log(2*ao/(M+1)), -log(2*ao(end:-1:1)/(M+1)) ];   % Even number of subpaths
                else
                    ao = [ log(2*ao/M), 0, -log(2*ao(end:-1:1)/M) ];        % Odd number of subpaths
                end
                ao = ones(n_mobiles,1) * ao;                        % Per MT angle offsets in [deg]
                P = sum( powers,3 );                                % Average power offsets for all frequencies
                P = P ./ ( sum( P , 2 ) * oM );                     % Normalize powers
                as = sqrt( sum(P.*ao.^2,2) - sum( P.*ao,2).^2 );    % angular spread in [deg]
                
                AoD(:,st:st+M-1) = ((h_builder.scenpar.PerClusterAS_D./as) * oM) .* ao * pi/180 + CAoD(:,n)*oM;
                AoA(:,st:st+M-1) = ((h_builder.scenpar.PerClusterAS_A./as) * oM) .* ao * pi/180 + CAoA(:,n)*oM;
                EoD(:,st:st+M-1) = ((h_builder.scenpar.PerClusterES_D./as) * oM) .* ao * pi/180 + CEoD(:,n)*oM;
                EoA(:,st:st+M-1) = ((h_builder.scenpar.PerClusterES_A./as) * oM) .* ao * pi/180 + CEoA(:,n)*oM;
                st = st + M;
            end
            
            % Restrict angle ranges
            AoD = angle(exp(1j*AoD));
            AoA = angle(exp(1j*AoA));
            EoD( EoD >  pi/2 ) =  pi - EoD( EoD >  pi/2 );
            EoD( EoD < -pi/2 ) = -pi - EoD( EoD < -pi/2 );
            EoA( EoA >  pi/2 ) =  pi - EoA( EoA >  pi/2 );
            EoA( EoA < -pi/2 ) = -pi - EoA( EoA < -pi/2 );
            
            n_clusters = n_paths;
            NumSubPaths = ones(1,n_paths);
            
        otherwise
            error('QuaDRiGa:builder:generate_subpaths:notSupported',...
                ['Subpath generation method "',h_builder.scenpar.SubpathMethod,'" is not supported.']);
            
    end
    
    % Generate polarization transfer matrix
    if use_3GPP_baseline
        
        % LOS polarization transfer matrix
        xprmat = zeros( 4, n_paths, n_mobiles );
        xprmat(1,1,:) = 1;                                                  % LOS
        xprmat(4,1,:) = -1;                                                 % LOS
        
        if n_nlos_clusters > 0
            % Generate XPR, 3GPP TR 38.901, Step 9
            randC       = randn( n_mobiles, n_nlos_paths  );
            xpr_sigma   = h_builder.lsp_vals(8,2);
            xpr_mu      = h_builder.lsp_vals(8,1);
            xpr_nlos    = randC * xpr_sigma + xpr_mu; % dB
            
            % Draw initial random phases, 3GPP TR 38.901, Step 10
            randC       = rand( n_mobiles, n_nlos_paths, 4 );
            
            % Calculate the polarization transfer matrix
            xpr_nlos = sqrt( 1./10.^(0.1*permute(xpr_nlos,[3,2,1])) );
            i_nlos_paths = [ false(1,n_los_clusters), true(1,n_paths-n_los_clusters) ];
            xprmat(:,i_nlos_paths,:) = exp(1j*( 2*pi* permute(randC,[3,2,1]) - pi ));  % NLOS phases
            xprmat(2,i_nlos_paths,:) = xprmat(2,i_nlos_paths,:) .* xpr_nlos;                  % NLOS XPR
            xprmat(3,i_nlos_paths,:) = xprmat(3,i_nlos_paths,:) .* xpr_nlos;                  % NLOS XPR
        end
        
    else
        % Use spatially consistent polarization rotation
        xpr_sigma = 0.1 * reshape( h_builder.lsp_vals(8,2,:), n_freq, 1 ) * ones(1,n_mobiles);
        xpr_mu = 10*log10(h_builder.xpr);
        
        gamma = zeros( n_mobiles, n_nlos_paths, n_freq );
        kappa = zeros( n_mobiles, n_nlos_paths, n_freq );
        
        % Polarization rotation is independently generated for different frequencies
        for iF = 1 : n_freq
            
            % Generate spatially correlated random variables for the linear polarization offset
            if isempty( xpr_sos )
                randC = randn( n_mobiles,n_nlos_paths );
            else
                randC = val( xpr_sos(:,1:n_subpaths,iF), rx_pos, tx_pos_SOS ).';   % Normal distribution
                randC = reshape( randC, n_mobiles, n_nlos_clusters, n_subpaths );
                randC = reshape( permute( randC, [1,3,2] ) , n_mobiles,n_nlos_paths );
            end
            
            xpr_linear = randC .* ( xpr_sigma(iF,:)' * o_nlos_paths ) + xpr_mu(iF,:).' * o_nlos_paths; % dB
            gamma(:,:,iF) = acot( sqrt( 10.^(0.1*xpr_linear) ) );
            
            % Generate spatially correlated random variables for the circular polarization offset
            if isempty( xpr_sos )
                randC = randn( n_mobiles,n_nlos_paths );
            else
                randC = val( xpr_sos(:,n_subpaths+1,iF), rx_pos, tx_pos_SOS ).';   % Normal distribution
                randC = randC( :,:,o_subpaths );
                randC = reshape( permute( randC, [1,3,2] ) , n_mobiles,n_nlos_paths);
            end
            
            xpr_circular = randC .* ( xpr_sigma(iF,:)' * o_nlos_paths ) + xpr_mu(iF,:).' * o_nlos_paths; % dB
            kappa(:,:,iF) = acot( sqrt( 10.^(0.1*xpr_circular) ) );
        end
        
        los_angle = zeros(n_mobiles,n_los_clusters,n_freq);
        gamma = cat( 2, los_angle, gamma );
        kappa = cat( 2, los_angle, kappa );
        
        % Calculate the polarization transfer matrix
        gamma  = permute( gamma, [4,2,1,3] );
        kappa  = permute( exp(-1j*kappa), [4,2,1,3] );
        xprmat = zeros( 4, n_paths, n_mobiles, n_freq );
        
        % Linear XPR
        xprmat( 1,:,:,: ) = cos( gamma );       % M(1,1)
        xprmat( 2,:,:,: ) = sin( gamma );       % M(2,1)
        xprmat( 3,:,:,: ) = -xprmat( 2,:,:,: ); % M(1,2)
        xprmat( 4,:,:,: ) = xprmat( 1,:,:,: );  % M(2,2)
        
        % Circular XPR
        xprmat( 1,:,:,: ) = xprmat( 1,:,:,: ) .* kappa;
        xprmat( 2,:,:,: ) = xprmat( 2,:,:,: ) .* kappa;
        kappa = -conj( kappa );
        xprmat( 3,:,:,: ) = xprmat( 3,:,:,: ) .* kappa;
        xprmat( 4,:,:,: ) = xprmat( 4,:,:,: ) .* kappa;
    end
    
    % Calculate the correct XPRmat for the ground reflection
    if use_ground_reflection
        scale = 1./max( abs(R_par), abs(R_per) );
        xprmat(1,2,:,:) = permute( R_par.*scale,[3,4,1,2] );
        xprmat(4,2,:,:) = permute( R_per.*scale,[3,4,1,2] );
    end
    
    % Generate random initial phases
    pin = zeros( n_mobiles,n_paths,n_freq );
    if h_builder.simpar(1,1).use_random_initial_phase
        for iF = 1 : n_freq  % Phases are independent for each frequency in multi-frequency simulations
            
            % Generate spatially correlated random variables for the NLOS phases
            if isempty( h_builder.pin_sos )
                randC = rand( n_mobiles,n_nlos_paths );
            else
                randC = val( h_builder.pin_sos(:,:,iF), rx_pos, tx_pos_SOS ).';   % Uniform
                randC = reshape( randC, n_mobiles, n_nlos_clusters, h_builder.scenpar.NumSubPaths );
                randC = reshape( permute( randC, [1,3,2] ) , n_mobiles,n_nlos_paths );
            end
            
            % Set phases
            pin(:,n_los_clusters+1:end,iF) = 2*pi*randC-pi;
        end
    end
    
    % Save output variables to builder
    if isempty( h_builder.NumClusters )
        h_builder.NumClusters = n_clusters;
    end
    if isempty( h_builder.NumSubPaths )
        h_builder.NumSubPaths = NumSubPaths;
    end
    if isempty( h_builder.pow )
        h_builder.pow = pow;
    end
    if isempty( h_builder.taus )
        h_builder.taus = taus;
    end
    if isempty( h_builder.AoD )
        h_builder.AoD = AoD;
    end
    if isempty( h_builder.AoA )
        h_builder.AoA = AoA;
    end
    if isempty( h_builder.EoD )
        h_builder.EoD = EoD;
    end
    if isempty( h_builder.EoA )
        h_builder.EoA = EoA;
    end
    if isempty( h_builder.xprmat )
        h_builder.xprmat = xprmat;
    end
    if isempty( h_builder.pin )
        h_builder.pin = pin;
    end
    
end
end
