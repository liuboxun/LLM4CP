function [ ip, ip1 ] = init_path_indices( h_channel, ic1, seg_ind, seg_has_gr )
%INIT_PATH_INDICES Initializes the path indices
%
% Input:
%   h_channel
%   An [ 1 x N ] array of channel objects obtained from "qd_builder".
%
%   ic1
%   Index of the current segment [ 1 x 1 ]
%
%   seg_ind
%   A [ N x 1 ] uint16 array indicating which channel object links to which track.
%
%   seg_has_gr
%   A [ N x 1 ] logical array indicating if the segment has a ground reflection component.
%
% Output:
%   ip
%   Path indices in the output channel
%
%   ip1
%   Matching path indices in the current channel
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

% Find the track in the channel objects
current_trk = seg_ind( ic1 );            % Current  track index
trk_ind = seg_ind == current_trk;        % Indices that blong to the current track

ip1 = 1 : h_channel( 1,ic1 ).no_path;

if any( seg_has_gr( trk_ind ) ) && ~seg_has_gr( ic1 )
    % The track has a ground reflection, but not in the first segment. The second
    % path is therefor a NLOS path. In this case, this path gets moved to the
    % third positions and the second is reserved for the GR.
    ip = [ 1 , 3 : h_channel( 1,ic1 ).no_path+1 ];
else
    ip = ip1;
end

end

