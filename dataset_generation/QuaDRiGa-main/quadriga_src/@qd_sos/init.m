function init( h_sos, use_same )
%INIT Initializes the random phases
%
% Calling object:
%   Object array
%
% Input:
%   use_same
%   If set to 1 (default), identical phases are used for the dual-mobility option, assuming that
%   devices are on the same radio-map. If set to 0, different values are used.
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


if ~exist( 'use_same','var' ) 
    use_same = true;
end

if numel(h_sos) > 1
    
    sic = size( h_sos );
    for n = 1 : prod( sic )
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
        init( h_sos(i1,i2,i3,i4), use_same );
    end
    
else
    % Fix for Ocatave
    h_sos = h_sos(1,1);
    
    if use_same
        N = 1;
    else
        N = 2;
    end

    for n = 1 : N
        
        % Generate random variables
        r = rand(1,h_sos.no_coefficients);
        
        % Map to Interval [-pi , pi[
        r = 2*pi*(r.'-0.5);
       
        % Set SOS phases
        if use_same
            h_sos.sos_phase = repmat( single(r),1,2);
        else
            h_sos.sos_phase(:,n) = single( r );
        end
        
    end
end

end
