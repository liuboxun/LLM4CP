function ic2 = find_overlapping_segment( h_channel, ic1, seg_ind )
%FIND_OVERLAPPING_SEGMENT Finds the overlapping segment of a channel
%
% Input:
%   h_channel
%   An [ 1 x N ] array of channel objects obtained from "qd_builder".
%
%   ic1
%   Index of the current segment [ 1 x 1 ]
%
%   seg_ind
%   The index-list of the segments [ N x 1 ]
%
% Output:
%   ic2
%   Index of the segment that overlaps with the current one  [ 1 x 1 ]
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

% Parses the channel names if "seg_ind" is not given
if ~exist( 'seg_ind','var' ) || isempty( seg_ind )
    [ ~, seg_ind, order ] = parse_channel_names( h_channel );
    h_channel = h_channel( 1,order );           % Order channels to match the names
end

current_trk = seg_ind( ic1 );           % Current  track index
trk_ind = seg_ind == current_trk;       % Indices that belong to the current track
n_seg = sum( trk_ind );                 % Number of segments belonging to the current track
i_seg = sum( trk_ind( 1:ic1 ) );        % Current segment

% Is the track closed?
i_first_seg = find( trk_ind,1 );
if h_channel( 1,i_first_seg ).initial_position > 1
    closed = true;
else
    closed = false;
end

% This determines the the start and end-segments for closed tracks
if i_seg < n_seg && h_channel(1,ic1).no_snap > 1
    ic2 = find( trk_ind(ic1+1:end) , 1 ) + ic1;
elseif closed
    ic2 = i_first_seg;
else
    ic2 = [];
end

end
