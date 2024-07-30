function [ Ri, Di ] = acf_2d( h_sos )
%ACF_2D Interpolates the ACF to a 2D version
%
%   This method calculates a 2D version of the given ACF (qd_sos.acf). The distance ranges from -2 Dmax to 2 Dmax, where
%   Dmax is the maximum value in qd_sos.dist. Values outside the specified range are set to 0. 
%
% Output:
%   Ri      2D ACF
%   Di      Vector of sample points (in x and y direction) for the 2D ACF in [m]
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
   error('QuaDRiGa:qd_sos:acf_2d','acf_2d not definded for object arrays.');
else
    h_sos = h_sos(1,1); % workaround for octave
end

R   = h_sos.acf;
D   = h_sos.dist;

Di = [ D(1:end-1) , D(end)+D ];
Di = [ -Di(end:-1:2) , Di ];

nDi = numel(Di);
oDi = ones(1,nDi,'uint8');

Dc  = Di( oDi,: );
Dc  = abs( Dc + 1j*Dc.' );
ind = Dc(:) <= max( D );

Ri  = zeros( nDi,nDi,'single' );
Ri(ind) = qf.interp( D, 0, R, Dc(ind) );

end
