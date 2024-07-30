function [ s, h_sos ] = rand( dist_decorr, ca, cb, acf_type )
%RAND Generates spatially correlated uniformly distributed random numbers
%
% Calling object:
%   None (static method)
%
% Input:
%   dist_decorr
%   Vector of decorrelation distances  [1 x M] or [ M x 1 ]
%
%   ca
%   Coordinates for the first mobile device in [m] given as [3 x N] matrix. The rows correspond to
%   the x,y and z coordinate.
%
%   cb
%   Coordinates for the corresponding second mobile device in [m] given as [3 x N] matrix. The rows
%   correspond to the x,y and z coordinate. This variable must either be empty or have the same
%   size as "ca".
%
%   acf_type
%   String describing the shape of the autocorrelation function and the number of sinusoids,
%   Default: 'Comb300'
%
% Output:
%   s
%   Random spatially correlated numbers [ M x N ]
%
%   h_sos
%   A handle to the used qd_sos object
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


s = zeros( numel( dist_decorr ), size(ca,2) );
h_sos = qd_sos([]);

if ~exist( 'cb','var' ) || isempty( cb )
   cb = []; 
end

if ~exist( 'acf_type','var' ) || isempty( acf_type )
   acf_type = 'Comb300'; 
end

for n = 1 : numel( dist_decorr )
    if dist_decorr(n) == 0
        dist_decorr(n) = 0.1;
    end
    h_sos(1,n) = qd_sos( acf_type, 'Uniform', dist_decorr(n) );
    s(n,:) = h_sos(1,n).val( ca, cb );
end

end
