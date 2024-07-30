function [ trk_names, seg_ind, order, trk_has_gr, seg_has_gr, tx_ind, rx_ind, tx_names, rx_names ] ...
    = parse_channel_names( h_channel )
%PARSE_CHANNEL_NAMES Parses the channel names
%
% The channel names can have different formats:
%   "Scen_Tx_Rx_Seg", "Scen_Tx_Rx" or "Tx_Rx"
%
% This function parses all the names and extracts the unique track names and the segments
% that belong to a track. The channel objects  are reordered in alphabetical order to match
% the order in "Tx_Rx_Seg".
%   
% Input:
%   h_channel
%   An [ 1 x N ] array of channel objects obtained from "qd_builder".
%
% Output:
%   trk_names
%   A { T x 1 } cell array containing the unique track names. "T" is the number of tracks.
%   Track names are ordered alphabetically.
%
%   seg_ind
%   A [ N x 1 ] uint32 array indicating which channel object links to which track.
%
%   order
%   A [ N x 1 ] index list containing correct order of the channel objects.
%
%   trk_has_gr
%   A [ T x 1 ] logical array indicating if the track has a ground reflection component.
%
%   seg_has_gr
%   A [ N x 1 ] logical array indicating if the segment has a ground reflection component.
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

% Get the number of channels
n_channel = numel(h_channel);
sic = size( h_channel );
if numel( sic ) ~= 2 && sic(1)~= 1
    error('"h_channel" must be a 1xN vector of qd_channel objects.');
end

trk_names  = cell( n_channel, 1 );
seg_names  = cell( n_channel, 1 );
tx_names   = cell( n_channel, 1 );
rx_names   = cell( n_channel, 1 );
seg_has_gr = false( n_channel, 1 );
single_seg = false( n_channel, 1 );

% Parse the name stings from the channel objects. 
for i_channel = 1 : n_channel                   % Do for each channel
    name = h_channel(1,i_channel).name;         % read channel name
    tmp = regexp( name , '_' );                 % split "Scen_Tx_Rx_Seg"

    if isempty( tmp )                           % No underscore in name
        if isempty( name )
            tx = 'Tx';                          % default tx name
            rx = 'Rx';                          % default rx name
        else
            tx = [name,'-Tx'];                  % default tx name
            rx = [name,'-Rx'];                  % default rx name
        end
        seg = 'seg0001';                        % set segment number to 1
        single_seg( i_channel ) = true;         % Indicate the track has only one segment
    elseif numel( tmp ) == 1                    % We have already "Tx_Rx"
        tx = name(1:tmp(1)-1);                  % store tx name
        rx = name(tmp(1)+1:end);                % store rx name
        seg = 'seg0001';                        % set segment number to 1
        single_seg( i_channel ) = true;         % Indicate the track has only one segment
    else
        tx = name(tmp(1)+1:tmp(2)-1);           % store tx name
        if numel(tmp)==2                        % if there is only one segment ...
            rx = name(tmp(2)+1:end);            % store rx name
            seg = 'seg0001';                    % set segment number to 1
            single_seg( i_channel ) = true;     % Indicate the track has only one segment
        else
            rx = name(tmp(2)+1:tmp(3)-1);       % store rx name
            seg = name(tmp(3)+1:end);           % store segment name
        end
    end
    
    trk_names{i_channel} = [tx,'_',rx];        % create a unique track id ...
    seg_names{i_channel} = [tx,'_',rx,'_',seg];   % ... and a segment id
    tx_names{i_channel}  = tx; 
    rx_names{i_channel}  = rx; 
    
    % Check if the channel object has a ground reflection component
    % This is stored in the par-struct
    chan_par = h_channel(1,i_channel).par;
    if ~isempty( chan_par ) && isfield( chan_par , 'has_ground_reflection' ) && chan_par.has_ground_reflection == 1
        seg_has_gr( i_channel ) = true;
    end
end

% Sort everything
[ ~, order ] = sort( seg_names );                   
trk_names    = trk_names( order );
seg_names    = seg_names( order );
seg_has_gr   = seg_has_gr( order );
single_seg   = single_seg( order );
tx_names     = tx_names( order );
rx_names     = rx_names( order );

% Indicator if the track has a GR component
trk_has_gr = false(1,1);

% Get the unique track names
% The following code implements "unique(id_trk)" but does not sort the
% values in id_trk.
pt = 1;
while pt < numel( trk_names )
    if single_seg( pt )
        ii = [];
    else
        ii = find( strcmp( trk_names(pt) , trk_names(pt+1:end) ) ) + pt;
    end
    trk_has_gr( pt,1 ) = any( seg_has_gr( [pt;ii] ) );
    if ~isempty( ii )
        ind = setdiff( 1:numel(trk_names), ii );
        trk_names = trk_names( ind );
        single_seg = single_seg( ind );
    end
    pt = pt + 1;
end

% Get the unique tx names
pt = 1;
while pt < numel( tx_names )
    ii = find( strcmp( tx_names(pt) , tx_names(pt+1:end) ) ) + pt;
    if ~isempty( ii )
        ind = setdiff( 1:numel(tx_names), ii );
        tx_names = tx_names( ind );
    end
    pt = pt + 1;
end

% Get the unique rx names
pt = 1;
while pt < numel( rx_names )
    ii = find( strcmp( rx_names(pt) , rx_names(pt+1:end) ) ) + pt;
    if ~isempty( ii )
        ind = setdiff( 1:numel(rx_names), ii );
        rx_names = rx_names( ind );
    end
    pt = pt + 1;
end

% Remove the "_seg" element from "seg_names" so that the names match the track names
seg_tx = seg_names;
seg_rx = seg_names;
for i_channel = 1 : n_channel
    ind = regexp( seg_names{i_channel} , '_' );
    seg_tx{i_channel} = seg_names{i_channel}(1:ind(1)-1);
    seg_rx{i_channel} = seg_names{i_channel}(ind(1)+1:ind(2)-1);
    seg_names{i_channel} = seg_names{i_channel}(1:ind(2)-1);
end

% Get the indices
seg_ind = zeros( n_channel,1,'uint32');
tx_ind  = zeros( n_channel,1,'uint32');
rx_ind  = zeros( n_channel,1,'uint32');

for i_trk = 1 : numel( trk_names )
    ind = regexp( trk_names{i_trk} , '_' );
    itx = find( strcmp( trk_names{i_trk}(1:ind(1)-1) , tx_names ) );
    irx = find( strcmp( trk_names{i_trk}(ind(1)+1:end) , rx_names ) );
    
    if all( single_seg )
        ind = i_trk;
    else
        ind = strcmp( trk_names{i_trk} , seg_names );
    end
    seg_ind(ind) = uint32( i_trk );
    tx_ind(ind) = uint32( itx );
    rx_ind(ind) = uint32( irx );
end
    
end
