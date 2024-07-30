classdef qd_builder < handle
%QD_BUILDER Class for generating the channel coefficients and the model parameters
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

properties
    name = '';                              % Name of the 'qd_builder' object
    simpar = qd_simulation_parameters;    	% Object of class 'qd_simulation_parameters'
end

properties(Dependent,SetAccess=protected)
    % Number of frequencies associated to the 'qd_builder' object (read-only property). 
    %   Note that 'qd_builder.get_channels' can only process a single frequency. Use
    %   'qd_builder.split_multi_freq'  to split the builder into multiple builders - one for each
    %   frequency - after generating parameters.  
    no_freq
    
    % Number of receiver positions associated to this 'qd_builder' object (read-only property).
    %   Note that each segment in longer tracks is considered a new Rx position.
    no_rx_positions
end

properties
    % Indicates if the builder is a dual-mobility builder
    dual_mobility = -1;
    
    % The 3D transmitter (TX) positions in metric Cartesian coordinates.
    %   Dimensions: [ 1,1 ] or [ 3, no_rx_positions ] 
    tx_position = [];

    %  Handles of 'qd_track' objects for each TX.
    %   This property is only required for dual-mobility simulations. For single-mobility, it can be
    %   omitted. However, it is possible to change the TX antenna orientation by setting 
    %   'tx_track.orientation' even for single-mobility scenarios.
    %   Dimensions: [ ] or [ 1,1 ] or [ no_freq, no_rx_positions ] 
    tx_track = [];
    
    % Transmit array antenna definitions. 
    %   Handles of qd_arrayant objects for each TX-RX Link.
    %   Dimensions: [ 1,1 ] or [ no_freq, no_rx_positions ]
    tx_array = []; 
    
    % The 3D receiver (RX) positions in metric Cartesian coordinates. 
    %   This variable is obtained from the properties 'qd_track.initial_position' or 
    %   'qd_layout.rx_position'. Dimensions: [ 3, no_rx_positions ]
    rx_positions = [];
    
    % Handles of 'qd_track' objects for each RX. 
    %   Dimensions: [ ] or [ no_freq, no_rx_positions ]
    rx_track = [];
    
    % Receive array antenna definitions. 
    %   Handles of 'qd_arrayant ' objects for each TX-RX-link. 
    %   Dimensions: [ no_freq, no_rx_positions ]
    rx_array = [];
end
    
properties(Dependent)
    scenario                    % The name of the scenario (text string)
    scenpar                     % The parameter table
end

properties
    plpar = [];                 % Parameters for the path loss (scenario-dependent)
end    

properties(Dependent,SetAccess=protected)
    lsp_vals                    % The distribution values of the LSPs
    lsp_xcorr_chk               % Indicator if cross-correlation matrix is positive definite
end

properties(Dependent)
    lsp_xcorr                   % The cross-correlation matrix for the LSPs
end
    
properties
    sos = [];                  	% The large-scale parameter SOS generators
    path_sos = [];              % The SOS generators for the generation of MPCs
    xpr_sos = [];               % The SOS generators for the generation of the linear NLOS polarization
    pin_sos = [];               % The SOS generators for the generation of initial phases
    clst_dl_sos = [];           % The SOS generators for the generation per-cluster delay offsets
    gr_sos = [];                % The SOS generator for the ground reflection coefficient
    absTOA_sos = [];            % The SOS generator for the absolute time-of-arrival offset
    
    % A list of random number used to determine for the mutual coupling of subpaths at the Tx and Rx.
    %   The dimensions correspond to the angle (AoD, AoA, EoD, EoA), the path index and the number
    %   of frequencies.
    subpath_coupling = [];
    
    ds = [];                    % The RMS delay spread in [s] for each receiver position with dimensions [n_freq, n_rx]
    kf = [];                    % The Rician K-Factor [linear scale] for each receiver position with dimensions [n_freq, n_rx]
    sf = [];                    % The shadow fading [linear scale] for each receiver position with dimensions [n_freq, n_rx]
    asD = [];                   % The azimuth spread of departure in [deg] for each receiver position with dimensions [n_freq, n_rx]
    asA = [];                   % The azimuth spread of arrival in [deg] for each receiver position with dimensions [n_freq, n_rx]
    esD = [];                   % The elevation spread of departure in [deg] for each receiver position with dimensions [n_freq, n_rx]
    esA = [];                   % The elevation spread of arrival in [deg] for each receiver position with dimensions [n_freq, n_rx]
    xpr  = [];                  % The cross polarization ratio [linear scale] for each receiver position with dimensions [n_freq, n_rx]
    gr_epsilon_r = [];          % The relative permittivity for the ground reflection
    absTOA_offset = [];         % The absolute time-of-arrival offset in [s]
    
    NumClusters = [];           % The number of clusters, including LOS and ground-reflection (optional) paths.
    
    % The number of sub-paths for each cluster.
    %   Note that the LOS path is always present in QuaDRiGa and it can only have one subpath.
    %   Dimensions: [ 1, NumClusters ]
    NumSubPaths = [];
    
    % The delays for each cluster in [s] relative to the LOS delay 
    %   (i.e. the first delay is always zero. Dimensions: [ no_rx_positions, NumClusters ]. 
    %   If the scenario parameters define a value for 'PerClusterDS_gamma', i.e. a
    %   frequency-dependence for the cluster-DS, the dimensions are [ no_rx_positions, NumClusters, no_freq ] 
    taus = [];
    
    % The absolute path-gain for each cluster (linear values).
    %   This variable combines the normalized cluster-powers pow, the scenario-dependent total path
    %   gain given by plpar and the shadow fading sf. Changes made to gain will simultaneously
    %   adjust pow and sf. Dimensions: [ no_rx_positions, NumClusters, no_freq ]
    gain = [];
end
    
properties(Dependent)
    % The normalized cluster-powers (squared average amplitude) for each cluster. 
    %   The sum over all clusters is 1. Changes made to pow will simultaneously adjust the absolute
    %   path-gain. Dimensions: [ no_rx_positions, NumClusters, no_freq ]
    pow
end

properties
    AoD = [];                  	% The initial azimuth of departure angles for each path in [rad].
    AoA = [];                 	% The initial azimuth of arrival angles for each path in [rad].
    EoD = [];                 	% The initial elevation of departure angles for each path in [rad].
    EoA = [];                 	% The initial elevation of departure angles for each path in [rad].
    
    % The complex-valued polarization transfer matrix 
    %   describing the polarization change during scattering. The dimensions correspond to
    %   polarization matrix index  '[1 3 ; 2 4]' and the subpath number.  
    xprmat = [];
    
    pin = [];                 	% The initial phases in [rad] for each sub-path.
    
    fbs_pos = [];               % The positions of the first-bounce scatterers in 3D Cartesian coordinates.
    lbs_pos = [];               % The positions of the last-bounce scatterers in 3D Cartesian coordinates.
end

properties(Dependent,Hidden)
    scenpar_nocheck
end

% Data storage
properties(Access=private)
    Pscenario               = '';
    Pscenpar                = [];
    pow_wo_kf
    Ppow = [];
end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
end

methods
    
    % Constructor
    function h_builder = qd_builder( scenario, check_parfiles, scenpar )
        if exist( 'scenpar' , 'var' ) && ~isempty( scenpar )
            % If scenpar is given, we skip all checks, assuming that the values are correct
            h_builder.Pscenpar = scenpar;
            
        elseif exist( 'scenario' , 'var' ) && ~isempty( scenario )
            if exist( 'check_parfiles' , 'var' )
                if ~( all(size(check_parfiles) == [1 1]) ...
                        && (isnumeric(check_parfiles) || islogical(check_parfiles)) ...
                        && any( check_parfiles == [0 1] ) )
                    error('QuaDRiGa:qd_builder:WrongInput','??? "check_parfiles" must be 0 or 1')
                end
            else
                check_parfiles = true;
            end
            set_scenario_table( h_builder, scenario, check_parfiles );
        end
        
        % Initialize simpar with default settings
        h_builder.simpar = qd_simulation_parameters;
    end
    
    % Get functions
    function out = get.no_freq(h_builder)
        out = numel( h_builder.simpar(1,1).center_frequency );
    end
    function out = get.no_rx_positions(h_builder)
        out = size( h_builder.rx_positions,2 );
    end
    function out = get.scenario(h_builder)
        out = h_builder.Pscenario;
    end
    function out = get.scenpar(h_builder)
        out = h_builder.Pscenpar;
    end
    function out = get.scenpar_nocheck(h_builder)
        out = h_builder.Pscenpar;
    end
    function out = get.pow(h_builder)
        no_rx = h_builder.no_rx_positions;
        no_fr = h_builder.no_freq;
        no_cl = size(h_builder.gain,2);
        if h_builder.no_rx_positions == 0 || isempty( h_builder.gain )
            out = [];
        else
            PL = h_builder.get_pl;
            SF = h_builder.sf;
            if isempty( SF )
                out = h_builder.gain ./ repmat(reshape(10.^(-0.1*PL.'),no_rx,1,no_fr),[1,no_cl,1]);
            else
                out = h_builder.gain ./ repmat(reshape(10.^(-0.1*PL.').*SF.',no_rx,1,no_fr),[1,no_cl,1]);
            end
        end
    end
    function out = get.lsp_vals(h_builder)
        if isempty( h_builder.Pscenpar )
            out = [];
        else
            % Carrier frequency in GHz
            f_GHz = h_builder.simpar(1,1).center_frequency / 1e9;
            oF = ones( 1,numel( f_GHz ));
            o8 = ones( 8,1 );
            
            scp = h_builder.Pscenpar;
            
            % Reference frequency offset (omega)
            omega = [ scp.DS_omega; scp.KF_omega; scp.SF_omega; scp.AS_D_omega; ...
                scp.AS_A_omega; scp.ES_D_omega; scp.ES_A_omega; scp.XPR_omega ];
            
            % Reference values (mu) including the frequency scaling
            mu = [ scp.DS_mu; scp.KF_mu; 0; scp.AS_D_mu; scp.AS_A_mu;...
                scp.ES_D_mu; scp.ES_A_mu; scp.XPR_mu ];
            gammap = [ scp.DS_gamma;scp.KF_gamma;0; scp.AS_D_gamma; ...
                scp.AS_A_gamma ;scp.ES_D_gamma ;scp.ES_A_gamma; scp.XPR_gamma];
            mu = mu( :,oF ) + gammap(:,oF) .* log10( omega(:,oF) + f_GHz(o8,:) );
            
            % STD of the LSPs (sigma) including the frequency scaling
            sigma = [ scp.DS_sigma;scp.KF_sigma;scp.SF_sigma; scp.AS_D_sigma;...
                scp.AS_A_sigma;scp.ES_D_sigma;scp.ES_A_sigma; scp.XPR_sigma ];
            delta = [ scp.DS_delta;scp.KF_delta;scp.SF_delta; scp.AS_D_delta;...
                scp.AS_A_delta;scp.ES_D_delta;scp.ES_A_delta; scp.XPR_delta ];
            sigma = sigma( :,oF ) + delta(:,oF) .* log10( omega(:,oF) + f_GHz(o8,:) );
            sigma( sigma<0 ) = 0;
            
            % Decorr dist (lambda)
            lambda = [ scp.DS_lambda;scp.KF_lambda;scp.SF_lambda;...
                scp.AS_D_lambda;scp.AS_A_lambda;scp.ES_D_lambda;...
                scp.ES_A_lambda; scp.XPR_lambda ];
            lambda = lambda( :,oF );
            
            % Distance-dependence (epsilon) of the reference value
            epsilon = [ scp.DS_epsilon; scp.KF_epsilon; 0;...
                scp.AS_D_epsilon; scp.AS_A_epsilon; scp.ES_D_epsilon;...
                scp.ES_A_epsilon; scp.XPR_epsilon ];
            epsilon = epsilon( :,oF );
            
            % Height-dependence (zeta) of the reference value
            zeta = [ scp.DS_zeta; scp.KF_zeta; 0;...
                scp.AS_D_zeta; scp.AS_A_zeta; scp.ES_D_zeta;...
                scp.ES_A_zeta; scp.XPR_zeta ];
            zeta = zeta( :,oF );
            
            % Elevation-dependence (alpha) of the reference value
            alpha = [ scp.DS_alpha; scp.KF_alpha; 0;...
                scp.AS_D_alpha; scp.AS_A_alpha; scp.ES_D_alpha;...
                scp.ES_A_alpha; scp.XPR_alpha ];
            alpha = alpha( :,oF );
            
            % Distance-dependence (kappa) of the reference STD
            kappap = [ scp.DS_kappa; scp.KF_kappa; scp.SF_kappa;...
                scp.AS_D_kappa; scp.AS_A_kappa; scp.ES_D_kappa;...
                scp.ES_A_kappa; scp.XPR_kappa ];
            kappap = kappap( :,oF );
            
            % Height-dependence (tau) of the reference STD
            tau = [ scp.DS_tau; scp.KF_tau; scp.SF_tau;...
                scp.AS_D_tau; scp.AS_A_tau; scp.ES_D_tau;...
                scp.ES_A_tau; scp.XPR_tau ];
            tau = tau( :,oF );
            
            % Elevation-dependence (beta) of the reference STD
            beta = [ scp.DS_beta; scp.KF_beta; scp.SF_beta;...
                scp.AS_D_beta; scp.AS_A_beta; scp.ES_D_beta;...
                scp.ES_A_beta; scp.XPR_beta ];
            beta = beta( :,oF );
            
            % Assemble output
            out = [mu(:), sigma(:), lambda(:), epsilon(:), zeta(:), alpha(:), kappap(:), tau(:), beta(:)];
            out = permute( reshape( out, [],numel( f_GHz ),9 ),[1,3,2] );
        end
    end
    function out = get.lsp_xcorr(h_builder)
        if isempty( h_builder.Pscenpar )
            out = [];
        else
            value = h_builder.Pscenpar;
            a = value.ds_kf;          % delay spread vs k-factor
            b = value.ds_sf;          % delay spread vs shadowing std
            c = value.asD_ds;         % departure AS vs delay spread
            d = value.asA_ds;         % arrival AS vs delay spread
            e = value.esD_ds;         % departure ES vs delay spread
            f = value.esA_ds;         % arrival ES vs delay spread
            g = value.sf_kf;          % shadowing std vs k-factor
            h = value.asD_kf;         % departure AS vs k-factor
            k = value.asA_kf;         % arrival AS vs k-factor
            l = value.esD_kf;         % departure DS vs k-factor
            m = value.esA_kf;         % arrival DS vs k-factor
            n = value.asD_sf;         % departure AS vs shadowing std
            o = value.asA_sf;         % arrival AS vs shadowing std
            p = value.esD_sf;         % departure ES vs shadowing std
            q = value.esA_sf;         % arrival ES vs shadowing std
            r = value.asD_asA;        % departure AS vs arrival AS
            s = value.esD_asD;        % departure ES vs departure AS
            t = value.esA_asD;        % arrival ES vs departure AS
            u = value.esD_asA;        % departure ES vs arrival AS
            v = value.esA_asA;        % arrival ES vs arrival AS
            w = value.esD_esA;        % departure ES vs arrival ES
            x1 = value.xpr_ds;        % xpr vs delay spread
            x2 = value.xpr_kf;        % xpr vs k-factor
            x3 = value.xpr_sf;        % xpr vs shadowing
            x4 = value.xpr_asd;       % xpr vs departure AS
            x5 = value.xpr_asa;       % xpr vs arrival AS
            x6 = value.xpr_esd;       % xpr vs departure ES
            x7 = value.xpr_esa;       % xpr vs arrival ES

            out = [ 1  a  b  c  d  e  f x1;...
                a  1  g  h  k  l  m x2;...
                b  g  1  n  o  p  q x3;...
                c  h  n  1  r  s  t x4;...
                d  k  o  r  1  u  v x5;...
                e  l  p  s  u  1  w x6;...
                f  m  q  t  v  w  1 x7;...
                x1  x2  x3  x4  x5  x6  x7 1];
        end
    end
    function out = get.lsp_xcorr_chk(h_builder)
        out = false;
        if ~isempty( h_builder.Pscenpar )
            [~,p] = chol( h_builder.lsp_xcorr, 'lower');
            if p <= 0
                out = true;
            end
        end
    end
    
    % Set functions
    function set.scenario(h_builder,value)
        if ~( ischar(value) )
            error('QuaDRiGa:qd_builder:WrongInput','??? "scenario" must be a string.')
        end
        set_scenario_table( h_builder, value);
    end
    
    function set.scenpar(h_builder,value)
        if ~( isstruct(value) )
            error('QuaDRiGa:qd_builder:WrongInput','??? "scenpar" must be a structure.')
        end
        set_scenario_table( h_builder, value );
    end
    
    function set.scenpar_nocheck(h_builder,value)
        % Faster when we know that "scenpar" is correct
        h_builder.Pscenpar = value;
    end
    
    function set.pow(h_builder,value)
        if ~isempty( value )
            no_rx = h_builder.no_rx_positions;
            no_fr = h_builder.no_freq;
            no_cl = size(value,2);
            if size( value,1 ) == 1 && no_rx > 1
                value = repmat( value, [no_rx,1,1] );
            elseif size( value,1 ) ~= no_rx
                error('QuaDRiGa:qd_builder:WrongInput','Number of rows in "pow" must match number of RX in builder.')
            end
            if size( value,3 ) == 1 && no_fr > 1
                value = repmat( value, [1,1,no_fr] );
            elseif size( value,3 ) ~= no_fr
                error('QuaDRiGa:qd_builder:WrongInput',...
                    'Number of 3rd-dimension elements in "pow" must match number of frequencies in builder.')
            end
            value = value ./ repmat( sum(value,2), [1,no_cl,1]);     % Normalize to sum-power = 1
            PL = h_builder.get_pl;
            SF = h_builder.sf;
            if isempty( SF )
                value = value .* repmat(reshape(10.^(-0.1*PL.'),no_rx,1,no_fr),[1,no_cl,1]);
            else
                value = value .* repmat(reshape(10.^(-0.1*PL.').*SF.',no_rx,1,no_fr),[1,no_cl,1]);
            end
        end
        h_builder.gain = value;
    end
    
    function set.lsp_xcorr(h_builder,value)
        if any( size( value ) ~= 8 )
            error('QuaDRiGa:qd_builder:WrongInput','??? "lsp_xcorr" must be a 8x8 matrix.')
        end
        sp = h_builder.Pscenpar;
        sp.ds_kf = value( 1,2 ) ;
        sp.ds_sf = value( 1,3 ) ;
        sp.asD_ds = value( 1,4 ) ;
        sp.asA_ds = value( 1,5 ) ;
        sp.esD_ds = value( 1,6 ) ;
        sp.esA_ds = value( 1,7 ) ;
        sp.xpr_ds = value( 1,8 ) ;
        sp.sf_kf = value( 2,3 ) ;
        sp.asD_kf = value( 2,4 ) ;
        sp.asA_kf = value( 2,5 ) ;
        sp.esD_kf = value( 2,6 ) ;
        sp.esA_kf = value( 2,7 ) ;
        sp.xpr_kf = value( 2,8 ) ;
        sp.asD_sf = value( 3,4 ) ;
        sp.asA_sf = value( 3,5 ) ;
        sp.esD_sf = value( 3,6 ) ;
        sp.esA_sf = value( 3,7 ) ;
        sp.xpr_sf = value( 3,8 ) ;
        sp.asD_asA = value( 4,5 ) ;
        sp.esD_asD = value( 4,6 ) ;
        sp.esA_asD = value( 4,7 ) ;
        sp.xpr_asd = value( 4,8 ) ;
        sp.esD_asA = value( 5,6 ) ;
        sp.esA_asA = value( 5,7 ) ;
        sp.xpr_asa = value( 5,8 ) ;
        sp.esD_esA = value( 6,7 ) ;
        sp.xpr_esd = value( 6,8 ) ;
        sp.xpr_esa = value( 7,8 ) ;
        h_builder.Pscenpar = sp;
    end
end

methods(Static)
    [ scenarios , config_folder , file_names ] = supported_scenarios( parse_shortnames )
    varargout = call_private_fcn( functionName, varargin )
    h_builder = gen_cdl_model( cdl_model, center_frequency, mobile_speed, duration, ds, kf, asd, ...
        asa, esd, esa, cas_factor, sample_density )
end

% methods % Legacy parameter generation functions
%     function gen_lsf_parameters( h_builder , force )   % Legacy "gen_lsf_parameters"
%         if exist('force','var') && force(1)
%             gen_parameters( h_builder, 0 );  % Reset all
%         end
%         gen_parameters( h_builder, 1 );
%     end
%     function gen_ssf_parameters( h_builder )  % Legacy "gen_ssf_parameters"
%         gen_parameters( h_builder );
%     end
% end
end
