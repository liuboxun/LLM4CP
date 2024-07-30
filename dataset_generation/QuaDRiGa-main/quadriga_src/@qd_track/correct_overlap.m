function correct_overlap( h_track, overlap )
%CORRECT_OVERLAP Corrects positions of the segment start to account for the overlap.
%
% Calling object:
%   Object array
%
% Description:
%   After the channel coefficients are calculated, adjacent segments can be merged into a time-
%   continuous output. The merger assumes that the merging interval happens at the end of one
%   segment, before a new segments starts. In reality, however, the scenario change happens in
%   the middle of the overlapping part (and not at the end of it). This function corrects the
%   position of the segment start to account for that.
%
% Input:
%   overlap
%   The length of the overlapping part relative to the segment length. It can have values in
%   between 0 (no overlap) and 1 (ramp along the entire segment). The default value is 0.5. You
%   need to make sure that the same value is used when calling "qd_channel.merge".
%
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

% Parse input arguments
check = true;
if nargin < 2
    overlap = 0.5;
    check = false;
end

if check
    if ~( isnumeric(overlap) && all(size(overlap) == [1 1]) && isreal(overlap) ...
            && overlap<=1 && overlap>=0 )
        error('??? Overlap must be scalar, and in between 0 and 1')
    end
end

if numel(h_track) > 1
    
    sic = size( h_track );
    prc = false( sic );
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            correct_overlap( h_track(i1,i2,i3,i4), overlap );
            prc( qf.eqo( h_track(i1,i2,i3,i4), h_track ) ) = true;
        end
    end
    
else
    h_track = h_track(1,1);
    
    % Only do if there are more than on segment
    if h_track.no_segments > 1
        
        % Get the distance relative to the start point for each point on the track
        [~,dist] = get_length( h_track );
       
        % Get a list of the segment indices
        seg_ind = [ h_track.segment_index , h_track.no_snapshots ];

        for n = 1 : h_track.no_segments-1
            
            % Length of the segment in [m]
            l_seg_c = dist( seg_ind(n+1) ) - dist( seg_ind(n) );    % Current segment
            l_seg_n = dist( seg_ind(n+2) ) - dist( seg_ind(n+1) );  % Next segment
            l_over  =  l_seg_c * overlap;                           % Overlapping part  
            
            % When the segment end-point is shifted, the following segment get shorter. Hoever, the
            % next segment is allowed to shorten by maximal 33% of it original length. Here, we
            % calculate the distance that the current segment end point is shifted.
            % The factor of 0.67 is numerically optimized to minimize the distance between the
            % center of the mergin interval and the old segment center.
            l_shift = l_seg_n - max( l_seg_n * 0.67 , l_seg_n - 0.67*l_over );
            
            % The new length of the current segment
            l_seg_cn = l_seg_c + l_shift;
            
            % The end point of the extended current segment
            l_seg_cn_end = dist( seg_ind(n) ) +  l_seg_cn;
            
            % Find the next point on the track
            ii = find( dist > l_seg_cn_end , 1 );
            
            if dist(ii) - l_seg_cn_end < 0.1
                % There is a point on the track that is closer that 10 cm to the desired segment end
                % point or there are parameters assiciated with the track. Store that point and be done!
                seg_ind(n+1) = ii;
                
            else
                % There is no point close anough, and we need to add a new snapsot to the track.
                % Calculate the position of the new point
                PT = h_track.positions;
                P  = PT( :,ii-1 );          % Start point of LineSegment
                U  = PT( :,ii)-P;           % Direction
                r = ( l_seg_cn_end - dist(ii-1) ) ./ sqrt(sum(U.^2));
                PG =  P + r*U;
                
                or = h_track.orientation;                   % Copy the orientations
                h_track.positions = [ PT(:,1:ii-1), PG, PT(:,ii:end) ];
                h_track.orientation = [ or(:,1:ii-1), or(:,ii-1:end) ];
                
                seg_ind(n+1) = ii;
                seg_ind(n+2:end) = seg_ind(n+2:end)+1;
                [~,dist] = get_length( h_track );
            end
        end
        
        % Assign new segment index
        h_track.segment_index = seg_ind(1:end-1);
    end
end

end
