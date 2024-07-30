function h_sos = load( filename )
%LOAD Loads coefficients from mat file
%
% Sinusoid coefficients can be stored in a mat-file by calling qd\_sos.save. This (static) method
% loads them again. In this way, it is possible to precompute the sinusoid coefficients and save
% some significant time when initializing the method. It is possible to adjust the decorrelation
% distance of a precomputed function without needing to perform the calculations again.
%
% Input:
%   filename 	Path or filename to the coefficient file. 
%
% Output:
%   h_sos   A qd_sos object.
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

load(filename);

h_sos                   = qd_sos([]);
h_sos.name              = filename;
h_sos.Pdist_decorr      = single( dist( find( acf < (exp(-1) + 1e-6) ,1 ) ) );
h_sos.dist              = single( dist );
h_sos.acf               = single( acf );
h_sos.sos_freq          = single( fr );
h_sos.sos_amp           = sqrt( single( 2 / size(fr,1) ) );

h_sos.init;

end

