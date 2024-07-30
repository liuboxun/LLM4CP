function gen_lsf_from_ssf( h_builder, ignore_gr )
%GEN_LSF_FROM_SSF Calculates the LSF parameters from existing SSF parameters
%
% Calling object:
%   Object array
%
% Description:
%   This method uses pre-initialized small-scale-fading parameters (SSPs), i.e., taus, gain, AoD,
%   AoA, EoD, EoA, xprmat, and calculates the corresponding large-scale-fading parameters (LSPs),
%   i.e., ds, kf, sf, asD, asA, esD, esA, xpr, and gr_epsilon_r.  In general, the method
%   'qd_builder.gen_ssf_parameters' tries to map the given LSPs to SSPs as closely as possible.
%   However, there are limits, e.g., angular spreads cannot be larger than 180° . This method can
%   therefore be used to obtain the true LSPs that can be derived from the generated paths. The
%   output of this method is written to the object properties. Existing LSPs are overwritten.
%
% Input:
%   ignore_gr
%   Boolean variable that enables (default) or disables the calculation of the ground reflectivity
%   gr_epsilon_r. By default, 'gen_lsf_from_ssf' checks if the second path matches the ground
%   reflection delay and angles. If so, the method attempts to calculate the ground reflectivity
%   from the polarization transfer matrix. This can be disabled by setting 'ignore_gr = 0'. An
%   existing value of gr_epsilon_r will not be overwritten in this case.
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
if ~exist( 'ignore_gr','var' ) || isempty( ignore_gr )
    ignore_gr = false;
end

if numel(h_builder) > 1
    sic  = size( h_builder );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        gen_lsf_from_ssf( h_builder( i1,i2,i3,i4 ), ignore_gr );
    end
    
else
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    if ignore_gr % Save existing GR parameters
        gr_epsilon_r = h_builder.gr_epsilon_r;
    end
        
    % Remove existing LSF parameters
    h_builder.gen_lsf_parameters(0,0);
    
    % Get required variables
    n_freq = h_builder.no_freq;
    n_rx = h_builder.no_rx_positions;
    n_clusters = h_builder.NumClusters;
    n_paths = sum(h_builder.NumSubPaths);
    tx_pos = h_builder.tx_position;
    rx_pos = h_builder.rx_positions;
    
    if ~isempty( n_clusters ) && n_rx >= 1
        
        % Test size of delays
        if  size(h_builder.taus,1) ~= n_rx || size(h_builder.taus,2) ~= n_clusters || ...
                ~( size(h_builder.taus,3) == 1 || size(h_builder.taus,3) == n_freq )
            error('QuaDRiGa:qd_builder:get_lsp_from_ssp','SSF Parameters are not correctly initialized.');
        end
        
        % Test if all SSF parameters are given
        if size(h_builder.gain,1) ~= n_rx || size(h_builder.gain,2) ~= n_clusters || size(h_builder.gain,3) ~= n_freq || ...
                any( size( h_builder.AoD ) ~= [ n_rx,n_clusters ] ) || ...
                any( size( h_builder.AoA ) ~= [ n_rx,n_clusters ] ) || ...
                any( size( h_builder.EoD ) ~= [ n_rx,n_clusters ] ) || ...
                any( size( h_builder.EoA ) ~= [ n_rx,n_clusters ] ) || ...
                size(h_builder.xprmat,1) ~= 4 || size(h_builder.xprmat,2) ~= n_paths || ...
                size(h_builder.xprmat,3) ~= n_rx || size(h_builder.xprmat,4) ~= n_freq || ...
                any( size( tx_pos ) ~= [ 3,n_rx ] ) || ...
                any( size( rx_pos ) ~= [ 3,n_rx ] )
            error('QuaDRiGa:qd_builder:get_lsp_from_ssp','SSF Parameters are not correctly initialized.');
        end
        
        % Determine if the builder has a GR component
        if ( ignore_gr && ~isempty( gr_epsilon_r ) ) || h_builder.check_los == 2
            use_ground_reflection = true;
        else
            use_ground_reflection = false;
        end
        
        % Call of "h_builder.check_los" might add a zero-power LOS component - n_paths increases by 1
        n_paths = sum(h_builder.NumSubPaths);
        
        % Get the absolute, non-normalized path powers
        pow = h_builder.gain;
        
        % Initialize LSF parameters
        ds = zeros( n_freq, n_rx );
        kf = zeros( n_freq, n_rx );
        asD = zeros( n_rx, n_freq );
        asA = zeros( n_rx, n_freq );
        esD = zeros( n_rx, n_freq );
        esA = zeros( n_rx, n_freq );
        
        % Estimate LSPs
        sf = permute( sum(pow,2),[3,1,2] ) ./ 10.^(-0.1*h_builder.get_pl);
        for i_freq = 1 : n_freq
            if size( h_builder.taus, 3 ) == n_freq
                ds( i_freq,: ) = qf.calc_delay_spread( h_builder.taus(:,:,i_freq), pow(:,:,i_freq) ).';
            else
                ds( i_freq,: ) = qf.calc_delay_spread( h_builder.taus, pow(:,:,i_freq) ).';
            end
            [ asD(:,i_freq), esD(:,i_freq) ] = qf.calc_angular_spreads_sphere( h_builder.AoD, h_builder.EoD, pow(:,:,i_freq), false );
            [ asA(:,i_freq), esA(:,i_freq) ] = qf.calc_angular_spreads_sphere( h_builder.AoA, h_builder.EoA, pow(:,:,i_freq), false );
            if use_ground_reflection && n_clusters > 2
                kf( i_freq,: ) = sum( pow(:,1:2,i_freq),2) ./ sum( pow(:,3:end,i_freq),2);
            elseif ~use_ground_reflection && n_clusters > 1
                kf( i_freq,: ) = pow(:,1,i_freq) ./ sum( pow(:,2:end,i_freq) ,2);
            else
                kf( i_freq,: ) = Inf;
            end
        end
        asD = asD'*180/pi;
        esD = esD'*180/pi;
        asA = asA'*180/pi;
        esA = esA'*180/pi;
        
        % Estimate the XPR
        [ xprL, xprC ] = qf.calc_xpr( reshape( h_builder.xprmat, 4,n_paths,[] ) );
        xpr = xprC;
        xpr( ~isinf(xprL) ) = xprL( ~isinf(xprL) ) + xprC( ~isinf(xprL) );
        xpr = clst_avg( 0.5*(xpr), h_builder.NumSubPaths );
        if use_ground_reflection && n_clusters > 2
            xpr = mean(xpr(:,3:end),2);
        elseif ~use_ground_reflection && n_clusters > 1
            xpr = mean(xpr(:,2:end),2);
        else
            xpr = Inf( n_rx*n_freq, 1);
        end
        xpr = reshape( xpr, n_rx,n_freq ).';
        
        % Calculate the ground reflectivity
        if use_ground_reflection && ~ignore_gr
            d_2d = hypot( tx_pos(1,:) - rx_pos(1,:), tx_pos(2,:) - rx_pos(2,:) );
            theta_r = atan2( ( rx_pos(3,:) + tx_pos(3,:) ), d_2d );
            
            % The polarization reflection coefficient
            R_par = permute( h_builder.xprmat(1,2,:,:),[4,3,1,2]);
            R_per = permute( h_builder.xprmat(4,2,:,:),[4,3,1,2]);
            
            % The reflection coefficient
            R = reshape( 1./(1+h_builder.gain(:,1,:)./h_builder.gain(:,2,:)), n_rx, n_freq )';
            R(R<1e-14) = 1e-14;
            scale = sqrt( R./(0.5*abs(R_par).^2  + 0.5*abs(R_per).^2) );
            
            y =  R_per./scale;
            a = repmat( sin( theta_r ),    n_freq,1 );
            b = repmat( cos( theta_r ).^2, n_freq,1 );
            
            gr_epsilon_r = conj( ( a.^2.*( y-1 ).^2 + b.*( y+1 ).^2 ) ./ ( y+1 ).^2 );
            
        elseif ~ignore_gr
            gr_epsilon_r = [];
        end
        
        % Set LSPs in builder
        h_builder.ds = ds;
        h_builder.kf = kf;
        h_builder.sf = sf;
        h_builder.asD = asD;
        h_builder.asA = asA;
        h_builder.esD = esD;
        h_builder.esA = esA;
        h_builder.xpr = xpr;
        h_builder.gr_epsilon_r = gr_epsilon_r;
    end
    
end
end
