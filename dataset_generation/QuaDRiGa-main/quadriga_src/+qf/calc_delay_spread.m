function [ ds, mean_delay ] = calc_delay_spread( taus, pow, threshold, granularity )
%CALC_DELAY_SPREAD Calculates the delay spread in [s]
%
% Description:
%   This function calculates the RMS delay spread from a given set of delays and powers. It is used
%   by the "qd_builder" to map the path powers to delays.
%
% Input:
%   taus
%   A vector of deays [s]. Dimensions: [ n_taus x n_path ]
%
%   pow
%   A vector of path powers in [W]. Dimensions: [ n_taus x n_path ]
%
%   threshold
%   An additional threshold in [dB] (scalar value) for the path powers relative to the strongest
%   path. For example: with a threshold of 20 dB, paths having less than 1/100 of the power of the
%   strongest path are not included in the calculation of the DS. Default: Inf dB (all paths are
%   used)
%
%   granularity
%   This parameter in units of [s] (scalar value) can be used to group paths in delay domain. For
%   example: At a system bandwidth of 20 MHz the time resolution is roughly limited to 50 ns.
%   However, several paths might be estimated from different antenna elements at slightly different
%   delays (e.g. one at 110 ns and another at 113 ns). Since the delay difference (3 ns) is below
%   the time resolution (50 ns), they might belong to the same path. Thus, setting the granularity
%   to 50 ns will sum up the powers of the two paths and treat them as one. Default: 0 s (all paths
%   are treated as independent)
%
% Output:
%   ds
%   The RMS delay spread for each delay vector. Dimensions: [ n_taus x 1 ]
%
%   mean_delay
%   The mean delay in [s]. Dimensions: [ n_taus x 1 ]
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

if ~exist('threshold','var') || isempty( threshold )
    threshold = Inf;
end

if ~exist('granularity','var') || isempty( granularity )
    granularity = 0;
end

N = size( taus,1 );
oN = ones(1,N);
if size( pow,1 ) < N
    pow = pow( oN, : );
end

if granularity > 0
    min_delay = floor( min(taus(:))/granularity ) * granularity ;
    max_delay =  ceil( max(taus(:))/granularity ) * granularity ;
    delay_win = min_delay : granularity : max_delay;
    
    % calculate the PDP
    pdp = zeros( N , numel( delay_win ) );
    ind = round( (taus - min_delay)./granularity ) + 1;
    for n = 1 : N
        pdp(n,:) = accumarray( ind(n,:)', pow(n,:)', [numel( delay_win ),1] )';
    end
    [ ds, mean_delay ] = qf.calc_delay_spread( delay_win, pdp, threshold );
    
    
else
    oP = ones(1,size(pow,2));

    % Apply threshold
    if ~isinf( threshold )
        max_pow = max(pow,[],2);
        min_pow = max_pow./10.^(0.1*threshold);
        pow( pow < min_pow(:,oP) ) = 0;
    end
    
    % Normalize powers
    pt = sum( pow,2 );
    pow = pow./pt( :,oP );
            
    mean_delay = sum( pow.*taus,2 );
    
    tmp = taus - mean_delay( :,oP );
    
    ds = sqrt( sum(pow.*(tmp.^2),2) - sum( pow.*tmp,2).^2 );
end

end
