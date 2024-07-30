function val_subpath = clst_expand( val_cluster, num_subpath )
%CLST_EXPAND Copies the cluster values to subpath values
% 
% Input:
%   val_cluster     Data for each cluster [ N x L x M ]
%   num_subpath     Number of subpaths per cluster [ 1 x L ]
%
%   M = max muber of subpaths
%   Only the first values are used. 
%   E.g.: M = 20, num_subpath(l) = 3 --> use first 3 values from third dimension.
%
% Output:
%   val_subpath     Data for each subpath [ N x  sum(num_subpath) ]
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

[N,L,M] = size( val_cluster );

expand_only = true;
if M == max( num_subpath )
    expand_only = false;
end

if isa( val_cluster,'logical' ) % Fix for older MATLAB versions
    val_subpath = false( N,sum(num_subpath) );
else
    val_subpath = zeros( N,sum(num_subpath), class( val_cluster ) );
end

ls = 1;
for l = 1:L
    le = ls + num_subpath(l) - 1;
    if expand_only
        val_subpath( :,ls:le ) = val_cluster( : , l*ones(1,num_subpath(l)) );
    else
        val_subpath( :,ls:le ) = val_cluster( : , l , 1:num_subpath(l) );
    end
    ls = le + 1;
end

end
