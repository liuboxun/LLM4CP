function save( h_sos, filename )
%SAVE Saves the coefficients to a mat-file
%
%  Sinusoid coefficients can be stored in a mat-file. In this way, it is possible to precompute the sinusoid
%  coefficients and save some significant time when initializing the method. It is possible to adjust the decorrelation
%  distance of a precomputed function without needing to perform the calculations again.
%
% Input:
%   filename 	Path or filename to the coefficient file. 
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
   error('QuaDRiGa:qd_sos:save','save not definded for object arrays.');
else
    h_sos = h_sos(1,1); % workaround for octave
end

if exist( 'filename', 'var' )
   h_sos.name = filename;
end

fr   = h_sos.sos_freq;
acf  = h_sos.acf;
dist = h_sos.dist;

save( h_sos.name,'fr','acf','dist' );

end
