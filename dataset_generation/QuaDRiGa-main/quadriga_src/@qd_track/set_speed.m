function set_speed( h_track , speed , no_chk )
%SET_SPEED Sets a constant speed in [m/s] for the entire track. 
%
% Calling object:
%   Object array
%
% Description:
%   This function fills the 'track.movement_profile' field with a constant speed value. This helps
%   to reduce computational overhead since it is possible to reduce the computation time by
%   interpolating the channel coefficients.
%
% Input:
%   speed
%   The terminal speed in [m/s]
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

if nargin < 3 || ~logical( no_chk )
    if exist( 'speed','var' )
        if ~isempty( speed )
            if ~( all(size(speed) == [1 1]) && isnumeric(speed) && isreal(speed) && speed > 0 )
                error('??? Invalid sampling interval. The value must be real and > 0.')
            end
        end
    else
        speed = 1;
    end
end

if numel(h_track) > 1
    
    sic = size( h_track );
    prc = false( sic );
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            set_speed( h_track(i1,i2,i3,i4), speed, false );
            prc( qf.eqo( h_track(i1,i2,i3,i4), h_track ) ) = true;
        end
    end
    
else
    h_track = h_track(1,1);
    if isempty( speed )
        h_track.movement_profile = [];
    else
        len = get_length( h_track );
        if len == 0
            h_track.movement_profile = [ 0 , 1/speed ; 1 h_track.no_snapshots ];
        else
            h_track.movement_profile = [ 0 , len/speed ; 0 len ];
        end
    end
end

end

