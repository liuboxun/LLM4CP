function h = checksum( h_track )
%CHECKSUM Calculates a checksum of all tracks to identify changes in the simulation settings
%
% Calling object:
%   Object array
%
% Output:
%   h
%   The checksum (uint64 number).
%
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

if numel(h_track) > 1
    
    h = uint64( 0 );
    sic = size( h_track );
    for n = 1 : prod( sic )
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
        h = h + checksum( h_track(i1,i2,i3,i4) );
        if h > 2147483647; h = mod(h,2147483647); end
    end
    
else
    h_track = h_track(1,1);
    
    h = sum( h_track.initial_position(:) ) + sum( h_track.positions(:) ) + sum( h_track.orientation(:) );
    h = h + sum(h_track.movement_profile(:));
    h = uint32( mod( abs(h)*100, 2147483647 ) );
    
    h = h + sum( cast( h_track.name ,'uint32' ) );
    if h > 2147483647; h = mod(h,2147483647); end
    
    h = h + uint32( mod( sum(h_track.segment_index), 2147483647 ) );
    if h > 2147483647; h = mod(h,2147483647); end
    
    x = cat(2,h_track.scenario{:});
    if ~isempty( x )
        h = h + sum( cast( x,'uint32' ) );
        if h > 2147483647; h = mod(h,2147483647); end
    end
    
    h = uint64( h );
end

end
