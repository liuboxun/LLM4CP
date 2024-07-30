function val_clst = clst_extract( val_full, num_subpath, ind_cluster )
%CLST_EXTRACT Extracts the subpath values 
% 
% Input:
%   val_full        Data for each cluster [ N x  sum(num_subpath) x O ]
%   num_subpath     Number of subpaths per cluster [ 1 x L ]
%   ind_cluster     Cluster index [ 1 ]
%
% Output:
%   val_clst        Data for each subpath [ N x  M x O ]
%
% M is the number od subpaths for the selected cluster
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

st = sum( num_subpath( 1:ind_cluster-1 ) )+1;
en = st + num_subpath(ind_cluster)-1;
val_clst = val_full( :,st:en,:,:,: );

end
