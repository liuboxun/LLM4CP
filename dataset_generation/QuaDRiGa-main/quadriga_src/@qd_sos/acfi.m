function val = acfi( h_sos, dist )
%ACFI Linear interpolation of the ACF
%
% Calling object:
%   Single object
%
% Input:
%   dist
%   Vector containing the distances in [m] for which the ACF should be interpolated.
%
% Output:
%   val
%   Interpolated ACF at the given distances
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
   error('QuaDRiGa:qd_sos:acfi','acfi not definded for object arrays.');
else
    h_sos = h_sos(1,1); % workaround for octave
end

s   = size( dist );
D   = reshape( dist, 1,[] );
D( D>h_sos.dist(end) ) = h_sos.dist(end);
val = qf.interp( h_sos.dist, 0, h_sos.acf, D );
val = reshape( val,s );

end
