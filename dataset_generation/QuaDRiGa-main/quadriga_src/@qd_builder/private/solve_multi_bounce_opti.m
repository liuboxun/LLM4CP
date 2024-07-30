function [ norm_a, norm_b, norm_c, valid ] =...
    solve_multi_bounce_opti( ahat, bhat, r, dist, d_min  )
%SOLVE_MULTI_BOUNCE_OPTI Solves the multi-bounce optimization problem
%
%  This function calculates the lengths |a|, |b| and |c| such that:
%       r = bhat*|b| + c - ahat*|a|
%       dist = |a| + |b| + |c|
%       |a| > d_min
%       |b| > d_min
%
%  There are 2 objectives:
%       1. |c| should be minimized
%       2. If |c| is minimal only for "|a| = d_min" or "|b| = d_min", then
%          the problem is relaxed. |c| can double compared to 1. and at
%          best |a| and |b| should be equal.
%
%   Inputs:
%       ahat    Unit length vector pointing from the Rx to the LBS
%               Size [ 3 x N x M ]
%
%       bhat    Unit length vector pointing from the Tx to the FBS
%               Size [ 3 x N x M ]
%
%       r       Vector pointing from Tx to Rx
%               Size [ 3 x N x M ] or [ 3 x N ] or [ 3 x 1 ]
%
%       dist    Path length dist = |a| + |b| + |c|
%               Size [ 1 x N x M ] or [ N x M ] or [ 1 x N ] or [ 1 ]
%
%       d_min   Minimum distance to the nearest scatterer. Size [ 1 ]
%
%   Outputs
%       norm_a  Length of a
%               [ 1 x N x M ]
%
%       norm_b  Length of b
%               [ 1 x N x M ]
%
%       norm_c  Length of c
%               [ 1 x N x M ]
%
%       valid   Indicates the the optimization problem has a solution.
%               [ 1 x N x M ]
%
%       c       Vector from the FBS to the LBS
%               [ 3 x N x M ]
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

N = size( ahat, 2 );
M = size( ahat, 3 );
NM = N*M;
oNM = ones(1,NM);

if M > 1
    ahat = reshape( ahat , 3 , NM );
    bhat = reshape( bhat , 3 , NM );
end

if numel( r ) == 3
    r = r(:,oNM);
elseif size(r,2) == N && size(r,3) == M
    r = reshape( r , 3 , NM );
elseif size(r,2) == N && size(r,3) == 1
    r = reshape( r(:,:,ones(1,M)) , 3 , NM );
end

if numel( dist ) == 1
    dist = dist( 1,oNM );
elseif size(dist,1) == 1 && size(dist,2) == N && size(dist,3) == M
    dist = reshape( dist , 1 , NM );
elseif size(dist,1) == N && size(dist,2) == M
    dist = permute( dist , [3,1,2] );
    dist = reshape( dist , 1 , NM );
elseif size(dist,1) == 1 && size(dist,2) == N && size(dist,3) == 1 && M > 1
    dist = reshape( dist(:,:,ones(1,M)) , 1 , NM );
end

% Calculate the maximum distance between the RX and the LBS
dist_los = sqrt(sum(abs(r).^2,1));
d_max = min( 0.5*(dist+dist_los), 4000 );

% Initial step to determine if the optimization problem has a solution.
norm_a = oNM * d_min;
[ ~, norm_c, norm_b ] = solve_cos_theorem( -bhat , r + ahat.*d_min , dist-d_min );

step_size  = max( 0.05*d_max, 10 );
d_max = d_max + step_size;

norm_a_new = norm_a;
norm_b_new = norm_b;
norm_c_new = norm_c;

upd = norm_b > 0 & d_max > d_min;

% Initial Optimization
% Iteratively find the optimal solution where |c| is minimized.

stp_cnt = 0;
while any( upd ) && stp_cnt < 400
    norm_a_new(upd) = norm_a(upd) + step_size(upd);
    
    % Case |a| < d_min
    ind = norm_a_new < d_min & upd;
    step_size( ind )  = 0.37 * abs( step_size( ind ) );
    norm_a_new( ind ) = d_min;
    
    % Case |a| > d_max
    ind = norm_a_new > d_max & upd;
    if any( ind )
        norm_a_new( ind ) = d_max( ind ); 
        step_size( ind ) = -0.9 * abs( step_size( ind ) );
    end
    
    [ ~, norm_c_new(upd) , norm_b_new(upd) ] = solve_cos_theorem( -bhat(:,upd) ,...
        r(:,upd) + ahat(:,upd).*norm_a_new([1,1,1],upd) , dist(upd)-norm_a_new(upd) );
    
    % Case |b| < d_min
    ind = norm_b_new < d_min & upd;
    if any( ind )
        step_size( ind ) = 0.9 * step_size( ind );
    end
    
    % Check if the length |c| has decreased
    ind = norm_c_new < norm_c & ~ind & upd;
    
    % Reduce step size and change direction
    step_size( ~ind ) = -0.37 * step_size( ~ind );
    
    % Use new values in next iteration step if |c| got shorter
    norm_a( ind ) = norm_a_new( ind );
    norm_b( ind ) = norm_b_new( ind );
    norm_c( ind ) = norm_c_new( ind );
    
    upd = upd & abs( step_size ) > 0.1;
    stp_cnt = stp_cnt + 1;
end

% Relaxation
% The optimal solution might be on one of the end points. This is not
% desirable since the scattering clusters are too close to the antennas.
% Hence, we relax the problem with the following constraints:
%
%   1. The length |c| obtained at the end point is allowed to double.
%   2. The lengths |a| and |b| ideally have the same length.

upd = norm_b > 0 & d_max > d_min & ( norm_b < 1.5*d_min | norm_a < 1.5*d_min );

step_size(:) = 0;
ind = upd & norm_b < 1.5*d_min;
step_size( ind ) = -max( 0.005*dist(ind), 1 );
ind = upd & norm_a < 1.5*d_min;
step_size( ind ) =  max( 0.005*dist(ind), 1 );

c_limit = norm_c;
c_limit( upd ) = 2*c_limit( upd );

length_diff = abs( norm_a - norm_b );
length_diff_new = length_diff;

stp_cnt = 0;
while any( upd ) && stp_cnt < 200
    norm_a_new(upd) = norm_a(upd) + step_size(upd);
    
    % Case |a| < d_min
    ind = norm_a_new < d_min & upd;
    step_size( ind )  = 0.37 * abs( step_size( ind ) );
    norm_a_new( ind ) = d_min;
    
    % Case |a| > d_max
    ind = norm_a_new > d_max & upd;
    if any( ind )
        norm_a_new( ind ) = d_max( ind );
        step_size( ind ) = 0;
    end
    
    [ ~, norm_c_new(upd) , norm_b_new(upd) ] = solve_cos_theorem( -bhat(:,upd) ,...
        r(:,upd) + ahat(:,upd).*norm_a_new([1,1,1],upd) , dist(upd)-norm_a_new(upd) );
    
    % Case |b| < d_min
    ind = norm_b_new < d_min & upd;
    if any( ind )
        step_size( ind ) = 0.9 * step_size( ind );
    end

    length_diff_new( upd ) = abs( norm_a_new(upd) - norm_b_new(upd) );
    
    % We can still improve if:
    ind = norm_c_new < c_limit & length_diff_new < length_diff & ~ind;
    
    % Reduce step size and change direction if we approach the limit
    step_size( ~ind & upd ) = -0.37 * step_size( ~ind & upd );
    
    % Use new values in next iteration step if |c| got shorter
    length_diff( ind ) = length_diff_new( ind );
    norm_a( ind ) = norm_a_new( ind );
    norm_b( ind ) = norm_b_new( ind );
    norm_c( ind ) = norm_c_new( ind );
    
    upd = upd & abs( step_size ) > 0.1;
    stp_cnt = stp_cnt + 1;
end

valid = abs( norm_a + norm_b + norm_c - dist) < 1e-6 &...
    norm_b >= d_min &...
    norm_a >= d_min;

if M > 1
    norm_a = reshape( norm_a , 1,N,M );
    norm_b = reshape( norm_b , 1,N,M );
    norm_c = reshape( norm_c , 1,N,M );
    valid  = reshape( valid  , 1,N,M );
end

% For debugging
if 0
    figure(1)
    plot(norm_a(:))
    hold on
    plot(norm_b(:),'r')
    plot(norm_c(:),'k')
    plot(norm_a(:) + norm_b(:) + norm_c(:),'-m')
    plot(dist,'--g','Linewidth',2)
    hold off
    legend('a','b','c','sum','dist','Location','NorthWest')
    
    
    norm_a_global = 1:1:200;
    norm_b_global = zeros( numel( norm_a_global ),N*L );
    norm_c_global = zeros( numel( norm_a_global ),N*L );
    for n = 1:numel( norm_a_global )
        [ ~, norm_c_global(n,:), norm_b_global(n,:) ] =...
            solve_cos_theorem( -bhat , r + ahat.*norm_a_global(n) , dist-norm_a_global(n) );
    end
    
    for n = 1:N*L
        tmp = norm_a_global + norm_b_global(:,n)' + norm_c_global(:,n)' - dist(n);
        ii = tmp < 1 & norm_b_global(:,n)' >= -5;
        
        figure(2)
        plot(norm_a_global(ii) , norm_b_global(ii,n) , 'r' )
        hold on
        plot(norm_a_global(ii) , norm_c_global(ii,n) , 'k' )
        plot(norm_a_global(ii) , norm_a_global(ii) , 'b' )
        
        hold off
        title(n)
        pause
    end
    
    
end

end
