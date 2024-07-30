function single( h_arrayant )
%SINGLE Set all properties to single precision to increase computation performance
%
% QuaDRiGa Copyright (C) 2011-2020
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

if numel(h_arrayant) > 1
    sic = size( h_arrayant );
    prc = false( sic ); % Processed elements
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            single( h_arrayant(i1,i2,i3,i4) ); % Set to single presision
            prc( i1,i2,i3,i4 ) = true;
            
            m = qf.eqo( obj(i1,i2,i3,i4), obj ); % Determine equal handles
            m(i1,i2,i3,i4) = false; % Remove own handle
            
            if any( m(:) )
                prc( m ) = true;
            end
        end
    end
else
    % Set all variables to single presision
    h_arrayant.center_frequency = single( h_arrayant.center_frequency );
    h_arrayant.elevation_grid = single( h_arrayant.elevation_grid );
    h_arrayant.azimuth_grid = single( h_arrayant.azimuth_grid );
    h_arrayant.Pelement_position = single( h_arrayant.Pelement_position );
    h_arrayant.PFa = single( h_arrayant.PFa );
    h_arrayant.PFb = single( h_arrayant.PFb );
    h_arrayant.Pcoupling = single( h_arrayant.Pcoupling );
    h_arrayant.Pphase_diff = single( h_arrayant.Pphase_diff );
end

end
