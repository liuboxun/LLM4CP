function gen_ssf_from_scatterers( h_builder, path_length_tolerance )
%GEN_SSF_FROM_SCATTERERS Calculates the SSF parameters from existing FBS and LBS positions
%
% Calling object:
%   Object array
%
% Description:
%   This method uses pre-initialized first-bounce (FBS) and last-bounce scatterer (LBS) positions
%   to calculate the corresponding small-scale-fading parameters (SSPs), i.e., taus, AoD, AoA, EoD,
%   EoA. In general, the method 'qd_builder.gen_fbs_lbs' tries to map the given SSPs to scatterer
%   positions by minimizing the distance between FBS and LBS. However, the resulting optimization
%   problem might have no solution. In this case, the mapping changes the departure angles. This
%   method can therefore be used to obtain the true SSPs that can be derived from the scatterer
%   positions. The output of this method is written to the object properties. Existing SSPs are
%   overwritten.
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

if ~exist('path_length_tolerance','var') || isempty( path_length_tolerance )
  path_length_tolerance = 1e-4;     % 0.1 mm
end

if numel(h_builder) > 1
    sic  = size( h_builder );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        gen_ssf_from_scatterers( h_builder( i1,i2,i3,i4 ) );
    end
    
else
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Dual Mobility indicator
    if h_builder.dual_mobility == -1
        h_builder.check_dual_mobility( false );
    end
    
    % Check if we need to do anything else
    if h_builder.no_rx_positions == 0 || isempty( h_builder.lbs_pos )
        return
    end
    
    % Get required variables
    n_rx = h_builder.no_rx_positions;
    n_freq = h_builder.no_freq;
    fbs_pos = h_builder.fbs_pos;
    lbs_pos = h_builder.lbs_pos;
    tx_pos = permute( h_builder.tx_position,[1,3,2]);
    rx_pos = permute( h_builder.rx_positions,[1,3,2]);
    n_paths = size(lbs_pos,2);
    
    if size(lbs_pos,1) ~= 3 || size(lbs_pos,3) ~= n_rx || size(lbs_pos,4) ~= n_freq
        error('QuaDRiGa:qd_builder:gen_ssf_from_scatterers',...
            '"lbs_pos" must have dimensions [ 3, n_paths, n_rx, n_freq ].');
    end
    if isempty( fbs_pos )
        fbs_pos = lbs_pos;
    elseif size(fbs_pos,1) ~= 3 || size(fbs_pos,2) ~= n_paths ||...
            size(fbs_pos,3) ~= n_rx || size(fbs_pos,4) ~= n_freq
        error('QuaDRiGa:qd_builder:gen_ssf_from_scatterers',...
            '"fbs_pos" must have dimensions [ 3, n_paths, n_rx, n_freq ].');
    end
    if size(tx_pos,1) ~= 3 || size(tx_pos,3) ~= n_rx
        error('QuaDRiGa:qd_builder:gen_ssf_from_scatterers',...
        '"tx_pos" must have dimensions [ 3, n_rx ].');
    end
    if size(rx_pos,1) ~= 3 || size(rx_pos,3) ~= n_rx
        error('QuaDRiGa:qd_builder:gen_ssf_from_scatterers',...
            '"rx_pos" must have dimensions [ 3, n_rx ].');
    end
    
    NumSubPaths = h_builder.NumSubPaths;
    if isempty( NumSubPaths )
        NumSubPaths = ones( 1,n_paths );
    elseif sum(NumSubPaths) ~= n_paths
        error('QuaDRiGa:qd_builder:gen_ssf_from_scatterers',...
            'sum( NumSubPaths ) must be equal to "n_paths".');
    end
    n_clusters = size( NumSubPaths, 2 );
    
    % 3D LOS distance
    d3d = permute(sqrt(sum( ( tx_pos - rx_pos ).^2,1)),[3,1,2]);
    
    % Set path lengths
    a = lbs_pos - repmat( rx_pos, [1,n_paths,1,n_freq] );
    b = fbs_pos - repmat( tx_pos, [1,n_paths,1,n_freq] );
    da = sqrt( sum(abs( a ).^2,1) );
    db = sqrt( sum(abs( b ).^2,1) );
    dc = sqrt( sum(abs( lbs_pos - fbs_pos ).^2,1) );
    dpath = permute( (da+db+dc) ,[3,2,4,1] ) - repmat( d3d, [1, n_paths, n_freq] );
    
    % Calculate angles
    AoA = permute( atan2( a(2,:,:,:),a(1,:,:,:) ) , [3,2,4,1] );
    AoD = permute( atan2( b(2,:,:,:),b(1,:,:,:) ) , [3,2,4,1] );
    EoA = permute( atan2( a(3,:,:,:),sqrt(a(1,:,:,:).^2+a(2,:,:,:).^2) ) , [3,2,4,1] );
    EoD = permute( atan2( b(3,:,:,:),sqrt(b(1,:,:,:).^2+b(2,:,:,:).^2) ) , [3,2,4,1] );
    
    % Average paths within clusters
    if any( NumSubPaths ~= 1 )
        dpath = clst_avg( dpath, NumSubPaths );
        AoA = angle( clst_avg( exp(1j*AoA), NumSubPaths ) );
        AoD = angle( clst_avg( exp(1j*AoD), NumSubPaths ) );
        EoA = clst_avg( EoA, NumSubPaths );
        EoD = clst_avg( EoD, NumSubPaths );
    end
    
    % Average over frequencies
    if n_freq > 1
        values_differ = false;
        for iF = 2 : n_freq
            values_differ = any(any( abs( dpath(:,:,1) - dpath(:,:,iF) ) > path_length_tolerance ));
        end
        if ~values_differ
            dpath = sum( dpath,3)/n_freq;
        end
        AoA = angle(sum(exp(1j*AoA),3)/n_freq);
        AoD = angle(sum(exp(1j*AoD),3)/n_freq);
        EoA = sum( EoA,3)/n_freq;
        EoD = sum( EoD,3)/n_freq;
    end
    
    % Save output variables to builder
    h_builder.NumClusters = n_clusters;
    h_builder.NumSubPaths = NumSubPaths;
    h_builder.taus = dpath / qd_simulation_parameters.speed_of_light;
    h_builder.AoD = AoD;
    h_builder.AoA = AoA;
    h_builder.EoD = EoD;
    h_builder.EoA = EoA;
    
end
end
