function [ len, dist ] = get_length( h_track )
%GET_LENGTH Calculates the length of the track in [m]
%
% Calling object:
%   Object array
%
% Output:
%   len
%   Length of a track in [m]
%
%   dist
%   Distance of each position (snapshot) from the start of the track in [m]
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
if numel(h_track) > 1
    
    % Do for each element in array
    sic     = size( h_track );
    len     = zeros( sic );
    dist    = cell( sic );
    
    for n=1:numel(h_track)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
        [ len(i1,i2,i3,i4), dist{i1,i2,i3,i4} ] = get_length( h_track(i1,i2,i3,i4) );
    end
    
else
    h_track = h_track(1,1);
    p = h_track.positions;
    if size(p,2) > 1
        dist = sqrt( sum( abs( p(:,2:end) - p(:,1:end-1) ).^2 ,1 ) );
        dist = [0, cumsum(dist) ];
        len = dist(end);
    else
        dist = 0;
        len = 0;
    end
end
