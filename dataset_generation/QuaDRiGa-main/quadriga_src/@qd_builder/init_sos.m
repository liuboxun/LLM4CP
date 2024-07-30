function init_sos( h_builder, usage )
%INIT_SOS Initializes all SOS random number generators
%
% Calling object:
%   Object array
%
% Description:
%   This method initializes all random number generators of the "qd_builder" object array. The
%   results are stored in the properties "sos", "gr_sos", "path_sos", "xpr_sos", "pin_sos", and
%   "clst_dl_sos". Once initialized, the "qd_builder" object will generate identical parameters and
%   channels. Note that this does not work for 3GPP baseline simulations.
%
% Input:
%   usage
%   Controls the behavior of the method: If set to 0, all existing SOS generators will be discarded
%   and the method exits. By default (1), new SOS generators will be created, existing ones will be
%   replaced. It set to 2, existing SOS generators will be reused and missing ones will be created.
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

% Parse input variables.
if ~exist( 'usage','var' ) || isempty( usage )
    usage = 1;
end

if numel(h_builder) > 1
    
    % Recursive call for all objects in the array
    sic = size( h_builder );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        init_sos( h_builder( i1,i2,i3,i4 ), usage );
    end
    
else
    
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Remove existing parameters
    if usage == 0 || usage == 1
        h_builder.sos = [];
        h_builder.path_sos = [];
        h_builder.xpr_sos = [];
        h_builder.pin_sos = [];
        h_builder.clst_dl_sos = [];
        h_builder.gr_sos = [];
        h_builder.absTOA_sos = [];
        h_builder.subpath_coupling = [];
    end
    
    % Dual Mobility indicator
    if h_builder.dual_mobility == -1 
        h_builder.check_dual_mobility( false );
    end
    dual_mobility = h_builder.dual_mobility;
    
    % Check if we need to do anything else
    if usage == 0
        return
    end
    
    % Set the number of clusters
    L = h_builder.scenpar.NumClusters;
    
    % Number of paths per NLOS cluster
    M = h_builder.scenpar.NumSubPaths;
    
    % Spatial consistency decorrelation distance in [m]
    SC_lambda = h_builder.scenpar.SC_lambda;
    if h_builder.scenpar.absTOA_mu > -30
        absTOA_lambda = h_builder.scenpar.absTOA_lambda;
    else
        absTOA_lambda = 0;
    end
    
    % ACF
    acf = h_builder.simpar(1,1).autocorrelation_function;
    if strcmp( acf, 'Disable' )     % Disable spatial consistency for SSF
        acf = 'Comb300';
        SC_lambda = 0;
        absTOA_lambda = 0;
    end
    
    use_ground_reflection = logical( h_builder.scenpar.GR_enabled );
    if use_ground_reflection && L == 1
        error('You need at least 2 clusters to enable the ground reflection option.');
    end
    
    % Number of NLOS clusters
    Ln = L - 1 - double( use_ground_reflection );
    
    % Number of frequencies
    F = numel( h_builder.simpar(1,1).center_frequency );
    
    % Initialize SOS generators for the LSF model
    lambda = h_builder.lsp_vals(:,3,1);
    if isempty( h_builder.sos )
        lsf_sos = qd_sos([]);
        
        lsf_sos(1,1) = qd_sos( acf, 'Normal', lambda(1) );  % DS
        lsf_sos(1,2) = qd_sos( acf, 'Normal', lambda(2) );  % KF
        lsf_sos(1,3) = qd_sos( acf, 'Normal', lambda(3) );  % SF
        lsf_sos(1,4) = qd_sos( acf, 'Normal', lambda(4) );  % ASD
        lsf_sos(1,5) = qd_sos( acf, 'Normal', lambda(5) );  % ASA
        lsf_sos(1,6) = qd_sos( acf, 'Normal', lambda(6) );  % ESD
        lsf_sos(1,7) = qd_sos( acf, 'Normal', lambda(7) );  % ESA
        lsf_sos(1,8) = qd_sos( acf, 'Normal', lambda(8) );  % XPR
        
        if dual_mobility && h_builder.no_rx_positions > 2
            remove_bias( lsf_sos, h_builder.rx_positions, h_builder.tx_position );
        elseif h_builder.no_rx_positions > 2
            remove_bias( lsf_sos, h_builder.rx_positions );
        end
        
        lsf_sos(1,1).sos_phase(:,2) = lsf_sos(1,1).sos_phase(:,1);  % Delays (are identical if Tx and Rx positions are swapped)
        lsf_sos(1,2).sos_phase(:,2) = lsf_sos(1,2).sos_phase(:,1);  % K-Factor (is identical if Tx and Rx positions are swapped)
        lsf_sos(1,3).sos_phase(:,2) = lsf_sos(1,3).sos_phase(:,1);  % Shadow-Fading (is identical if Tx and Rx positions are swapped)
        lsf_sos(1,8).sos_phase(:,2) = lsf_sos(1,8).sos_phase(:,1);  % XPR (is identical if Tx and Rx positions are swapped)
        
        lsf_sos(1,4).sos_phase(:,1) = lsf_sos(1,5).sos_phase(:,2);  % Azimuth angles 
        lsf_sos(1,5).sos_phase(:,1) = lsf_sos(1,4).sos_phase(:,2);  % (departure become arrival angles if positions are swapped)
        
        lsf_sos(1,6).sos_phase(:,1) = lsf_sos(1,7).sos_phase(:,2);  % Elevation angles 
        lsf_sos(1,7).sos_phase(:,1) = lsf_sos(1,6).sos_phase(:,2);  % (departure become arrival angles if positions are swapped)
        
        h_builder.sos = lsf_sos;
    end
    
    % SOS generator for spatially correlated absolute time-of-arrival
    if ~h_builder.simpar(1,1).use_3GPP_baseline && absTOA_lambda > 0 && isempty( h_builder.absTOA_sos )
        
        % Delay offsets (are identical if Tx and Rx positions are swapped)
        absTOA_sos = qd_sos( acf, 'Normal', absTOA_lambda );
        absTOA_sos.sos_phase(:,2) = absTOA_sos(1,1).sos_phase(:,1);
        
        if dual_mobility && h_builder.no_rx_positions > 2
            remove_bias( absTOA_sos, h_builder.rx_positions, h_builder.tx_position );
        elseif h_builder.no_rx_positions > 2
            remove_bias( absTOA_sos, h_builder.rx_positions );
        end
        
        h_builder.absTOA_sos = absTOA_sos;
    end
    
    % Initialize the SOS generator for the ground reflection coefficient
    if ~h_builder.simpar(1,1).use_3GPP_baseline && SC_lambda > 0 && isempty( h_builder.gr_sos )
        h_builder.gr_sos = qd_sos( acf, 'Uniform', SC_lambda );
        h_builder.gr_sos.sos_phase(:,2) = h_builder.gr_sos.sos_phase(:,1);
    end
    
    % Initialize SOS generators for the generation of multipath components
    if ~h_builder.simpar(1,1).use_3GPP_baseline && SC_lambda > 0 && isempty( h_builder.path_sos )
        path_sos = qd_sos([]);
        for i_cluster = 1 : Ln
            % Delays (are identical if Tx and Rx positions are swapped)
            path_sos(i_cluster,1) = qd_sos( acf, 'Uniform', SC_lambda ); % Delays
            path_sos(i_cluster,1).sos_phase(:,2) = path_sos(i_cluster,1).sos_phase(:,1);
            
            % Azimuth angles (departure become arrival angles if positions are swapped)
            path_sos(i_cluster,2) = qd_sos( acf, 'Uniform', SC_lambda ); % ASD
            path_sos(i_cluster,3) = qd_sos( acf, 'Uniform', SC_lambda ); % ASA
            path_sos(i_cluster,2).sos_phase(:,1) = path_sos(i_cluster,3).sos_phase(:,2);
            path_sos(i_cluster,3).sos_phase(:,1) = path_sos(i_cluster,2).sos_phase(:,2);
            
            % Elevation angles (departure become arrival angles if positions are swapped)
            path_sos(i_cluster,4) = qd_sos( acf, 'Uniform', SC_lambda ); % ASD
            path_sos(i_cluster,5) = qd_sos( acf, 'Uniform', SC_lambda ); % ASA
            path_sos(i_cluster,4).sos_phase(:,1) = path_sos(i_cluster,5).sos_phase(:,2);
            path_sos(i_cluster,5).sos_phase(:,1) = path_sos(i_cluster,4).sos_phase(:,2);
        end
        h_builder.path_sos = path_sos;
    end
    
    % Initialize SOS generators for the XPR
    if ~h_builder.simpar(1,1).use_3GPP_baseline && SC_lambda > 0 && isempty( h_builder.xpr_sos )
        xpr_sos = qd_sos([]);
        for i_freq = 1 : F
            for i_cluster = 1 : Ln
                for i_sub = 1 : M+1
                    xpr_sos(i_cluster,i_sub,i_freq) = qd_sos( acf, 'Normal', SC_lambda );
                    if i_freq == 1 % Fix for Octave
                        xpr_sos(i_cluster,i_sub).sos_phase(:,2) = xpr_sos(i_cluster,i_sub).sos_phase(:,1);
                    else
                        xpr_sos(i_cluster,i_sub,i_freq).sos_phase(:,2) = xpr_sos(i_cluster,i_sub,i_freq).sos_phase(:,1);
                    end
                end
            end
        end
        h_builder.xpr_sos = xpr_sos;
    end
    
    % Initialize SOS generators for initial phases
    if ~h_builder.simpar(1,1).use_3GPP_baseline && SC_lambda > 0 && isempty( h_builder.pin_sos )
        pin_sos = qd_sos([]);
        for i_freq = 1 : F
            for i_cluster = 1 : Ln
                for i_sub = 1 : M
                    pin_sos(i_cluster,i_sub,i_freq) = qd_sos( acf, 'Uniform', SC_lambda );
                    if i_freq == 1 % Fix for Octave
                        pin_sos(i_cluster,i_sub).sos_phase(:,2) = pin_sos(i_cluster,i_sub).sos_phase(:,1);
                    else
                        pin_sos(i_cluster,i_sub,i_freq).sos_phase(:,2) = pin_sos(i_cluster,i_sub,i_freq).sos_phase(:,1);
                    end
                end
            end
        end
        h_builder.pin_sos = pin_sos;
    end
    
    % Initialize SOS generators for mmMAGIC intra-cluster delay offsets
    if ~h_builder.simpar(1,1).use_3GPP_baseline && SC_lambda > 0 && isempty( h_builder.clst_dl_sos )
        clst_dl_sos = qd_sos([]);
        for i_cluster = 1 : Ln
            for i_sub = 1 : M
                clst_dl_sos(i_cluster,i_sub) = qd_sos( acf, 'Uniform', SC_lambda );
                clst_dl_sos(i_cluster,i_sub).sos_phase(:,2) = clst_dl_sos(i_cluster,i_sub).sos_phase(:,1);
            end
        end
        h_builder.clst_dl_sos = clst_dl_sos;
    end
    
    % Initialize subpath coupling
    NumSubPaths = Ln * M + L - Ln;
    if isempty( h_builder.subpath_coupling )
        if ~isempty( h_builder.NumSubPaths )
            NumSubPaths = sum( h_builder.NumSubPaths );
        end
        if h_builder.simpar(1,1).use_3GPP_baseline
            h_builder.subpath_coupling = rand( 4 , NumSubPaths , h_builder.no_rx_positions );
        else
            h_builder.subpath_coupling = rand( 4 , NumSubPaths , F );
        end
    elseif size( h_builder.subpath_coupling,2 ) > NumSubPaths
        h_builder.subpath_coupling = h_builder.subpath_coupling( :,1:NumSubPaths,: );
    end
end
end
