function [ ds, kf, sf, asD, asA, esD, esA, xpr ] = gen_lsf_parameters( h_builder, usage, show_warnings, tx_pos, rx_pos )
%GEN_LSF_PARAMETERS Generates the large-scale-fading parameters for all terminals and frequencies
%
% Calling object:
%   Object array
%
% Description:
%   This method generates correlated large scale parameters for each user position. Those
%   parameters are needed by the channel builder to calculate initial SSF parameters for each track
%   or segment which are then evolved into time varying channels. The LSF parameters require that
%   the SOS random generators are initialized first. Hence, you need to call 'qd_builder.init_sos'
%   before calling 'qd_builder.gen_lsf_parameters'. The output of this method is written to the
%   object properties.
%
% LSF parameters:
%   ds              The RMS delay spread in [s]
%   kf              The Ricean K-Factor [linear scale]
%   sf              The shadow fading [linear scale]
%   asD             The azimuth spread of departure in [deg]
%   asA             The azimuth spread of arrival in [deg]
%   esD             The elevation spread of departure in [deg]
%   esA             The elevation spread of arrival in [deg]
%   xpr             The cross polarization ratio [linear scale]
%   gr_epsilon_r    The relative permittivity for the ground reflection (optional)
%   absTOA_offset   The absolute time-of-arrival offset in [s] (optional)
%
% Input:
%   usage
%   Controls the behavior of the method: If set to 0, all existing LSF parameters will be discarded
%   and the method exits. By default (1), new LSF parameters will be created, existing ones will be
%   replaced. It set to 2, existing LSF parameters will be reused and missing ones will be created.
%
%   show_warnings
%   If set to true (default), 'qd_builder.gen_lsf_parameters' performs a set of tests to determine
%   if all provided input variables are correctly initialized. This can be disabled by setting
%   'show_warnings = 0'.
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
if ~exist( 'show_warnings','var' ) || isempty( show_warnings )
    show_warnings = true;
end

% Initialize output variables
ds = [];
kf = [];
sf = [];
asD = [];
asA = [];
esD = [];
esA = [];
xpr = [];

if numel(h_builder) > 1
    
    % Recursive call for all objects in the array
    sic = size( h_builder );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        gen_lsf_parameters( h_builder( i1,i2,i3,i4 ), usage, show_warnings );
    end
    
else
    
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Update usage if there is a input position
    if exist( 'tx_pos','var' )
        usage = 3;
    end
    
    % Remove existing LSF parameters
    if usage == 0 || usage == 1
        h_builder.ds = [];
        h_builder.kf = [];
        h_builder.sf = [];
        h_builder.asD = [];
        h_builder.asA = [];
        h_builder.esD = [];
        h_builder.esA = [];
        h_builder.xpr = [];
        h_builder.gr_epsilon_r = [];
        h_builder.absTOA_offset = [];
    end
    
    % Check if we need to do anything else
    if usage == 0 || h_builder.no_rx_positions == 0
        return
    end
    
    % Check if the required SOS generators are available
    if isempty( h_builder.sos )
        error('QuaDRiGa:qd_builder:gen_lsf_parameters','SOS random generators are not initialized.');
    end
    if ~h_builder.simpar(1,1).use_3GPP_baseline && ~strcmp( h_builder.simpar(1,1).autocorrelation_function,'Disable' ) && ...
            h_builder.scenpar.absTOA_mu > -30 && h_builder.scenpar.absTOA_lambda > 0 && isempty( h_builder.absTOA_sos )
        error('QuaDRiGa:qd_builder:gen_lsf_parameters','SOS random generator for absolute TOA is not initialized.');
    end
    if ~h_builder.lsp_xcorr_chk
        error('QuaDRiGa:qd_builder:gen_lsf_parameters',...
            ['LSP cross-correlation matrix of "',h_builder.name,'" is not positive definite.']);
    end
    
    % Get required variables
    n_clusters  = h_builder.scenpar.NumClusters;
    n_freq      = numel( h_builder.simpar(1,1).center_frequency );
    lsp_vals    = h_builder.lsp_vals;
    lsp_xcorr   = h_builder.lsp_xcorr;
    lsf_sos     = h_builder.sos;
    ES_D_mu_A   = h_builder.scenpar.ES_D_mu_A;
    ES_D_mu_min = h_builder.scenpar.ES_D_mu_min;
    use_3GPP_baseline = h_builder.simpar(1,1).use_3GPP_baseline;
    use_ground_reflection = logical( h_builder.scenpar.GR_enabled );
    
    if exist( 'rx_pos','var' )
        n_mobiles = size( rx_pos, 2 );
        dual_mobility = size( tx_pos,2 ) > 1 || h_builder.dual_mobility;
    else
        n_mobiles = h_builder.no_rx_positions;
        tx_pos    = h_builder.tx_position;
        rx_pos    = h_builder.rx_positions;
        dual_mobility = h_builder.dual_mobility;
    end
    
    % Test if we have at least 2 clusters in case of ground reflection
    if use_ground_reflection && n_clusters == 1
        error('QuaDRiGa:qd_builder:gen_parameters:gr_2_clusters',...
            'You need at least 2 clusters to enable the ground reflection option.');
    end
    
    % Test if all generators are set to uniform distribution
    if show_warnings
        for m = 1 : size( lsf_sos, 2)
            if ~strcmp( lsf_sos(1,m).distribution , 'Normal' )
                warning('QuaDRiGa:qd_builder:generate_lsf:Normal',...
                    'All random generators should be set to Normal distribution.');
                warning ('off','QuaDRiGa:qd_builder:generate_lsf:Normal');
            end
        end
        warning ('on','QuaDRiGa:qd_builder:generate_lsf:Normal');
        
        % Test if DS, SF, KF, XPR are reciprocal
        r = [1,2,3,8];
        for n = r
            tmp = cat(3,lsf_sos(:,n).sos_phase);
            tmp = tmp(:,1,:) - tmp(:,2,:);
            if any(abs(tmp(:))>1e-6)
                warning('QuaDRiGa:qd_builder:generate_lsf','DS, SF, KF and XPR must be reciprocal.');
            end
        end
        
        % Test if azimuth spreads are reciprocal
        tmp1 = cat(3,lsf_sos(:,4).sos_phase);
        tmp2 = cat(3,lsf_sos(:,5).sos_phase);
        tmp3 = tmp1(:,1,:) - tmp2(:,2,:);
        tmp4 = tmp2(:,1,:) - tmp1(:,2,:);
        if any( abs( [tmp3(:);tmp4(:)] ) > 1e-6 )
            warning('QuaDRiGa:qd_builder:generate_lsf','Azimuth spreads are not reciprocal.');
        end
        
        % Test if elevation spreads are reciprocal
        tmp1 = cat(3,lsf_sos(:,6).sos_phase);
        tmp2 = cat(3,lsf_sos(:,7).sos_phase);
        tmp3 = tmp1(:,1,:) - tmp2(:,2,:);
        tmp4 = tmp2(:,1,:) - tmp1(:,2,:);
        if any( abs( [tmp3(:);tmp4(:)] ) > 1e-6 )
            warning('QuaDRiGa:qd_builder:generate_lsf','Elevation spreads are not reciprocal.');
        end
    end
    
    if dual_mobility
        tx_pos_SOS = tx_pos;
    else
        tx_pos_SOS = [];
    end
    
    % Generate LSF parameters
    % Generate 3-D spatially correlated normal distributed random variables
    % Combine the 3-D variables to 6-D variables
    par = zeros(8,n_mobiles);
    for n = 1 : 8
        par(n,:) = val( lsf_sos(1,n), rx_pos, tx_pos_SOS );
    end
    
    % In a dual-mobility scenario, ASD and ASA are correlated when the Tx and Rx are close. This
    % correlation has to be included in the values.
    if ~isempty( tx_pos_SOS ) % Dual-mobility
        % The distance between Tx and Rx
        d = sqrt( sum(abs( rx_pos - tx_pos ).^2,1) );
        
        % The distance-dependent correlations for the four angles
        corr_ASD_ASA = double( [ lsf_sos(1,4).acfi( d ) ; lsf_sos(1,5).acfi( d ) ] );
        corr_ASD_ASA( corr_ASD_ASA > 1-1e-5 ) =  1-1e-5;
        corr_ESD_ESA = double( [ lsf_sos(1,6).acfi( d ) ; lsf_sos(1,7).acfi( d ) ] );
        corr_ESD_ESA( corr_ESD_ESA > 1-1e-5 ) =  1-1e-5;
        
        % Apply the correlation to each user
        for n = 1 : n_mobiles
            R_sqrt = sqrtm( [ 1 corr_ASD_ASA(1,n) ; corr_ASD_ASA(2,n) 1 ] );
            par(4:5,n)    = R_sqrt * par(4:5,n);
            
            R_sqrt = sqrtm( [ 1 corr_ESD_ESA(1,n) ; corr_ESD_ESA(2,n) 1 ] );
            par(6:7,n)    = R_sqrt * par(6:7,n);
        end
    end
    
    % Apply the inter-parameter correlation model
    R_sqrt = sqrtm( lsp_xcorr );
    par = R_sqrt * par;
    
    % Calculate the variables d2D, hBS and alpha from the TX and RX positions
    if size(tx_pos,2) == 1 && size(rx_pos,2) > 1
        d2D = sqrt( sum( ( repmat(tx_pos(1:2,:),1,n_mobiles) - rx_pos(1:2,:) ).^2,1 ) );
        hBS = repmat( tx_pos(3,:),1,n_mobiles );
    else
        d2D = sqrt( sum( ( tx_pos(1:2,:) - rx_pos(1:2,:) ).^2,1 ) );
        hBS = tx_pos(3,:);
    end
    hBS( hBS < 1e-4 ) = 1e-4;   % hBS cannot be 0 or smaller than 0
    d2D( d2D < 1e-4 ) = 1e-4;   % d2D cannot be 0 or smaller than 0
    alpha = atan( hBS ./ d2D );
    
    % Calculate mean values
    oN = ones( 1,n_mobiles );
    o8 = ones( 1,8 );
    oF = ones( 1,n_freq );
    mu = lsp_vals(:,oN,:) + lsp_vals(:,4*oN,:) .* log10( d2D(o8,:,oF) ) + ...
        lsp_vals(:,5*oN,:) .* log10( hBS(o8,:,oF) ) + lsp_vals(:,6*oN,:) .* log10( alpha(o8,:,oF) );
    
    % Apply distant-dependent ESD scaling with linear distance
    if ES_D_mu_A ~= 0
        mu(6,:,:) = mu(6,:,:) + ES_D_mu_A .* d2D(1,:,oF)/1000;
    end
    
    % Apply minimum value for the ESD
    mu( 6, mu(6,:) < ES_D_mu_min ) = ES_D_mu_min;
    
    % Calculate STD
    sigma = lsp_vals(:,2*oN,:) + lsp_vals(:,7*oN,:) .* log10( d2D(o8,:,oF) ) + ...
        lsp_vals(:,8*oN,:) .* log10( hBS(o8,:,oF) ) + lsp_vals(:,9*oN,:) .* log10( alpha(o8,:,oF) );
    
    % Apply mu and sigma from the parameter table for each frequency
    par = par( :,:,oF ) .* sigma + mu;
    
    % Transform to linear values
    par( [1,4:7],:,: ) = 10.^( par( [1,4:7],:,: ) );
    par( [2,3,8],:,: ) = 10.^( 0.1 * par( [2,3,8],:,: ) );
    
    % Write output LSP values
    if usage == 3
        ds   = permute( par(1,:,:) , [3,2,1] );
        kf   = permute( par(2,:,:) , [3,2,1] );
        sf   = permute( par(3,:,:) , [3,2,1] );
        if nargout > 3
            asD  = permute( par(4,:,:) , [3,2,1] );
            asA  = permute( par(5,:,:) , [3,2,1] );
            esD  = permute( par(6,:,:) , [3,2,1] );
            esA  = permute( par(7,:,:) , [3,2,1] );
            xpr  = permute( par(8,:,:) , [3,2,1] );
        end
        
    else
        if isempty( h_builder.ds )
            h_builder.ds = permute( par(1,:,:) , [3,2,1] );
        end
        if isempty( h_builder.kf )
            h_builder.kf = permute( par(2,:,:) , [3,2,1] );
        end
        if isempty( h_builder.sf )
            h_builder.sf = permute( par(3,:,:) , [3,2,1] );
        end
        if isempty( h_builder.asD )
            h_builder.asD = permute( par(4,:,:) , [3,2,1] );
        end
        if isempty( h_builder.asA )
            h_builder.asA = permute( par(5,:,:) , [3,2,1] );
        end
        if isempty( h_builder.esD )
            h_builder.esD = permute( par(6,:,:) , [3,2,1] );
        end
        if isempty( h_builder.esA )
            h_builder.esA = permute( par(7,:,:) , [3,2,1] );
        end
        if isempty( h_builder.xpr )
            h_builder.xpr = permute( par(8,:,:) , [3,2,1] );
        end
        
        % Ground Reflection Parameters
        if use_ground_reflection && h_builder.scenpar.GR_epsilon == 0
            
            % Generate spatially correlated random variables
            if isempty( h_builder.gr_sos )
                randC = rand( 1,n_mobiles );
            else
                randC = val( h_builder.gr_sos, rx_pos, tx_pos_SOS );  % Uniform distribution
            end
            
            % There are 3 ground types defined: dry, medium dry, wet
            % The complex-valued relative permittivity is frequency-dependent and given by:
            f_GHz = h_builder.simpar(1,1).center_frequency/1e9;
            g_dry = 3 + 1j * 0.003 * f_GHz.^1.34;
            g_med = 30.4 * f_GHz.^-0.47 + 1j * 0.18 * f_GHz.^1.05;
            g_wet = 31.3 * f_GHz.^-0.48 + 1j * 0.63 * f_GHz.^0.77;
            
            % The resulting permittivity is obtained by a linear interpolation of the ground type
            % depending on the random variable "randC". A value of 0 means "dry", 0.5 means "medium"
            % and 1 means "wet".
            gr_epsilon_r = zeros(n_freq,n_mobiles);
            
            i1 = randC <= 0.5;        % Dry to Medium wet ground
            i2 = randC >  0.5;        % Medium wet to wet ground
            
            % The weights for the linear interpolation
            w1 = 2 * randC(i1);
            w2 = 2 * (randC(i2)-0.5);
            
            % The relative permittivity of the ground at the different frequencies
            if any(i1)
                gr_epsilon_r( :,i1 ) = g_dry.'*(1-w1) + g_med.'*w1;
            end
            if any(i2)
                gr_epsilon_r( :,i2 ) = g_med.'*(1-w2) + g_wet.'*w2;
            end
            
        elseif use_ground_reflection
            gr_epsilon_r = ones( n_freq, n_mobiles ) * h_builder.scenpar.GR_epsilon;
        else
            gr_epsilon_r = [];
        end
        if isempty( h_builder.gr_epsilon_r )
            h_builder.gr_epsilon_r = gr_epsilon_r;
        end
        
        % Absolute time-of-arrival offset
        if h_builder.scenpar.absTOA_mu > -30 && isempty( h_builder.absTOA_offset )
            if isempty( h_builder.absTOA_sos )
                randC = randn( 1,n_mobiles );
            else
                randC = val( h_builder.absTOA_sos, rx_pos, tx_pos_SOS );
            end
            h_builder.absTOA_offset =...
                10.^( randC * h_builder.scenpar.absTOA_sigma + h_builder.scenpar.absTOA_mu );
        end
        
    end
    
end

end
