function [ ds, p, d ] = merging_avg_ds( cf, dl )
%CALC_DS Calculates the DS
%
% Input:
%   cf          Coefficients                    [ R x T x L x S ]
%   dl          Delays                          [ R x T x L x S ]
%
% Output: 
%   ds          Average DS                      [ 1 x 1 ]
%   p           Average normalized path-power   [ L x 1 ]
%   d           Average normalized path-delay   [ L x 1 ]
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

L = size( cf,3 );

p = permute( abs(cf).^2 , [3,1,2,4] );
p = mean( reshape( p, L, [] ) , 2 );
pn = p./sum(p,1);

d = permute( dl , [3,1,2,4] );
d = mean( reshape( d, L, [] ) , 2 );

ds = sqrt( sum(pn.*d.^2,1) - sum(pn.*d,1).^2 );

end
