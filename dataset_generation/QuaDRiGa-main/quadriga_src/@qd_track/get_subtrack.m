function subtracks = get_subtrack( h_track, i_segment, i_tx )
%GET_SUBTRACK Splits the track in subtracks for each segment
%
% Calling object:
%   Single object
%
% Description:
%   After defining segments along the track, one needs the subtrack that corresponds only to one
%   segment to perform the channel calculation. This new track can consist of two segments. The
%   first segment contains the positions from the previous segment, the second from the current.
%   This is needed to generate overlapping channel segments for the merging process. This function
%   returns the subtracks for the given segment indices. When no input argument is provided, all
%   subtracks are returned.
%
% Input:
%   i_segment
%   A list of indices indicating which subtracks should be returned. By default, all subtracks are
%   returned.
%
% Input:
%   i_tx
%   A list of indices indicating which transmitter should be returned. Usually, each transmitter in
%   a layout gets assigned a scenario. The scenario-IDs are stored in "qd_track.scenario", where
%   the number of rows corresponds to the number of transmitters in a layout. By default, all
%   transmitters are returned.
%
% Output:
%   subtracks
%   A vector of qd_track objects corresponding to the number of segments.
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

if numel( h_track ) > 1
    error('QuaDRiGa:qd_track:get_subtrack','get_subtrack not definded for object arrays.');
else
    h_track = h_track(1,1); % workaround for octave
end

% Number of segments
if ~exist('i_segment','var') || isempty( i_segment )
    i_segment = 1 : h_track.no_segments;
end

% Try to get an estimate of the number of transmitters in the layout
no_tx = size( h_track.scenario,1 );
if ~exist('i_tx','var') || isempty( i_tx )
    i_tx = 1 : no_tx;
elseif no_tx == 1 && any( i_tx > 1 )
    i_tx(:) = 1;
elseif no_tx > 1 && any( i_tx > no_tx )
    error('QuaDRiGa:qd_track:get_subtrack','Requested Tx indes not found in track.');
end

% Make a copy of the parameters
par = h_track.par;
names = {'o2i_loss','o2i_d3din'};

for seg = 1 : numel(i_segment)
    segment = i_segment(seg);
    
    % Select the part of the track that corresponds to the given segment
    if h_track.no_segments == 1
        % The current track has only one segment
        if h_track.closed == 1
            ind = 1 : h_track.no_snapshots-1;
        else
            ind = 1 : h_track.no_snapshots;
        end
        seg_ind = 1;
        scen_ind = 1;
        
    elseif h_track.no_segments == h_track.no_snapshots
        % Each snapshot has its own segment
        ind = h_track.segment_index( segment );
        seg_ind = 1;
        scen_ind = segment;
        
    elseif segment == 1
        % The current track has more than one segment, and the returned
        % segment is the first one.
        if h_track.closed == 1
            ind = [ h_track.segment_index( h_track.no_segments ) : h_track.no_snapshots-1 ,...
                1:h_track.segment_index( segment+1 )-1 ];
            seg_ind = [ 1 , h_track.no_snapshots-h_track.segment_index( h_track.no_segments )+1 ];
            scen_ind = [ h_track.no_segments , 1 ];
        else
            ind = 1 : h_track.segment_index( segment+1 )-1;
            seg_ind = 1;
            scen_ind = 1;
        end
        
    elseif segment == h_track.no_segments
        % The current track has more than one segment, and the returned
        % segment is the last one.
        if h_track.closed == 1
            ind = h_track.segment_index( segment-1 ) : h_track.no_snapshots-1;
        else
            ind = h_track.segment_index( segment-1 ) : h_track.no_snapshots;
        end
        seg_ind = [ 1 , h_track.segment_index( segment ) - h_track.segment_index( segment-1 ) + 1 ];
        scen_ind = [ segment-1 , segment ];
        
    else
        % The current track has more than one segment, and the returned
        % segment neither the first, nor the last one.
        ind = h_track.segment_index( segment-1 ) : h_track.segment_index( segment+1 )-1;
        seg_ind = [ 1 , h_track.segment_index( segment ) - h_track.segment_index( segment-1 ) + 1 ];
        scen_ind = [ segment-1 , segment ];
        
    end
    
    % Create new track with the corresponding data
    tr = qd_track([]);
    if h_track.no_segments > 1
        tr.name                 = [h_track.name,'_seg',num2str(segment,'%04u') ];
    else
        tr.name                 = h_track.name;
    end
    tr.positions                = h_track.positions( :,ind );
    tr.segment_index            = seg_ind;
    if tr.no_segments == 2
        sp = tr.positions( :, seg_ind(2) );
        tr.scenario = h_track.scenario(i_tx,scen_ind([2,2]));
    else
        sp = tr.positions( :, 1 );
        tr.scenario = h_track.scenario(i_tx,scen_ind);
    end
    
    if ~isempty( h_track.orientation )
        tr.orientation = h_track.orientation(:,ind);
    end
    
    if ~isempty(par)
        % Init struct
        out_par = struct('o2i_loss',[],'o2i_d3din', []);
        
        % Copy the data
        for n = 1 : numel( fieldnames( out_par ) )
            try
                val = par.(names{n});
                if ~isempty( val )
                    if size( val,1 ) == no_tx
                        i_tx_par = i_tx;
                    elseif size( val,1 ) == 1
                        i_tx_par = 1;
                    else
                        error('QuaDRiGa:qd_track:get_subtrack',...
                            ['"',h_track.name,'": Number of values in the parameter struct does not match the number of transmitters.']);
                    end
                    out_par.( names{n} ) = val(i_tx_par,scen_ind,:);
                end
            end
        end
        
        % Save to subtrack
        tr.par = out_par;
    end
    
    % Set the initial position
    tr.initial_position = h_track.initial_position + sp;
    for n=1:3
        tr.positions(n,:) = tr.positions(n,:) - sp(n);
    end
    
    % Append to subtracks-list
    if numel( i_segment ) == 1
        subtracks = tr;
    else
        subtracks(seg) = tr;
    end
end
end
