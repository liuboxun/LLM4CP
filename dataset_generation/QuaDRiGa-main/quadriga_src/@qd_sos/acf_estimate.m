function [ Re, De, Re_dual ] = acf_estimate( h_sos, no_pos, D )
%ACF_ESTIMATE Estimates the 3D and 6D ACF from randomly generated positions
%
% Calling object:
%   Single object
%
% Description:
%   This method creates random positions  within a cube of 0.4*max(D) m edge length. Then,
%   spatially correlated Normal distributed random numbers are generated for each position using
%   qd_sos.val. The distance between each pair of positions is calculated and pairs with similar
%   distance are grouped, i.e. positions with a distance between 0 and 2 meters of each other
%   belonged to group 1, positions with a distance between 2 and 4 meters belonged to group 2, and
%   so on. The Pearson correlation coefficient is calculated for the samples within each group.
%
%   For the 6D estimation, half the positions are used for the transmitter and half are used for
%   the receiver. Spatially correlated values are generated using the dual-mobility extension. The
%   grouping is done for the receiver positions under the constraint that the transmitter cannot
%   move more than the receiver. For example, if group 3 contains all pairs where the receivers
%   have a distance between 4 and 6 meters, then the transmitters have a distance between 0 and 6
%   meters.
%
% Input:
%   no_pos
%   Number of positions for the ACF estimation.
%
%   D
%   Distance vector containing the center distances in [m] of the groups.
%
% Output:
%   Re
%   Estimated 3D ACF
%
%   De
%   Distance vector containing the actual center distances in [m] of the groups calculated from the
%   random positions.
%
%   Re_dual
%   Estimated 6D ACF for the dual-mobility extension.
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

if numel( h_sos ) > 1 
   error('QuaDRiGa:qd_sos:acf_estimate','acf_estimate not definded for object arrays.');
else
    h_sos = h_sos(1,1); % workaround for octave
end

max_dist = max(D*0.4);

% Random positions in cubic space segment
pt = zeros(3,no_pos,'single');
for n = 1 : 2
    if n == 2
        pr = pt;
    end
    pt =(2*rand(3,no_pos,'single')-1)*max_dist;
end

% Calculate random values distances in 3D and 6D space
p3 = [ pt , pr ];
IP = p3' * p3;
d3 = sqrt(bsxfun(@plus, diag(IP), diag(IP)') - 2 * IP);
d3 = triu( d3 );
v3 = h_sos.val( p3 );

if nargout > 2
    p61 = pt;
    p62 = pr;
    
    IP = p61' * p61;
    d61 = sqrt(bsxfun(@plus, diag(IP), diag(IP)') - 2 * IP);
    d61 = triu( d61 );
    
    IP = p62' * p62;
    d62 = sqrt(bsxfun(@plus, diag(IP), diag(IP)') - 2 * IP);
    d62 = triu( d62 );
    
    p6 = [ pt ; pr ];
    IP = p6' * p6;
    d6 = sqrt(bsxfun(@plus, diag(IP), diag(IP)') - 2 * IP);
    d6 = triu( d6 );
    v6 = h_sos.val( pt,pr );
end

% Calculate correlation
Re = NaN( size(D) );
De = Re;
Re_dual = Re;
De_dual = Re;

for n = 1 : numel(D)
    
    if n == 1
        Dmin = 1e-5;
    else
        Dmin = 0.5*( D(n-1) + D(n) );
    end
    if n == numel(D)
        Dmax = D(n) + mean( diff(D) )/2;
    else
        Dmax = 0.5*( D(n) + D(n+1) );
    end
    
    ii = d3 > Dmin & d3 <= Dmax;
    [ i1,i2 ] = qf.qind2sub( size(ii), find( ii ) );
    if numel( i1 ) > 5
        Re(n) = qf.xcorrcoeff( v3(i1) , v3(i2));
        De(n) = mean(d3(ii));
    end

    if nargout > 2
        ij = d61 > Dmin & d61 <= Dmax;
        ik = d62 <= Dmax;
        ii = ij & ik;
        
        [ i1,i2 ] = qf.qind2sub( size(ii), find( ii ) );
        if numel( i1 ) > 5
            Re_dual(n) = qf.xcorrcoeff( v6(i1) , v6(i2));
            De_dual(n) = mean(d6(ii));
        end
    end
    
end

end
