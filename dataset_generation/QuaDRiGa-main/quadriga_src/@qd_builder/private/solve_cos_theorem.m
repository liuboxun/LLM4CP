function [ b, norm_b, norm_a ]  = solve_cos_theorem( ahat , r , dist  )
%SOLVE_COS_THEOREM Calculates the vector b by solving the cosine theorem
%
%  This function calculates the vector b such that:
%       b = r + ahat*|a|
%       dist = |a| + |b|
%
%   Inputs:
%       ahat    Unit length vector pointing from the Rx to the LBS
%               Size [ 3 x N x M ]
%
%       r       Vector pointing from Tx to Rx
%               Size [ 3 x N x M ] or [ 3 x N ] or [ 3 x 1 ]
%
%       dist    Path length dist = |a| + |b|
%               Size [ 1 x N x M ] or [ 1 x N ] or [ 1 ]
%
%   Outputs
%       b       Solution of "b = r + ahat*|a|"
%               [ 3 x N x M ]
%
%       norm_b  Length of b
%               [ 1 x N x M ]
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

N = size( ahat , 2 );
M = size( ahat , 3 );

oN = ones(1,N);
oM = ones(1,M);

if numel( r ) == 3
    r = r(:,oN,oM);
elseif M > 1 && size(r,2) == N && size(r,3) == 1
    r = r(:,:,oM);
end
norm_r_sq = sum( r.^2 );

if numel( dist ) == 1
    dist = dist(1,oN,oM );
elseif M>1 && size(dist,2) == N && size(dist,3) == 1
    dist = r(1,:,oM);
elseif size( dist,1 ) == N && size( dist,2 ) == M
    dist = permute( dist , [3,1,2] );
end

rT_ahat = sum( ahat .* r );
tmp     = ( dist + rT_ahat );
indZ    = abs( tmp ) < 1e-12;
tmp( indZ ) = 1;
norm_a  = 0.5 * ( dist.^2 - norm_r_sq ) ./ tmp;
norm_a( indZ ) = 0;

b = r + norm_a([1,1,1],:,:).*ahat;
norm_b = sqrt( sum( b.^2 ) );

end
