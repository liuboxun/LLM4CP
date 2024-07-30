function h_builder = init_builder( h_layout, check_parfiles, split_tx  )
%INIT_BUILDER Creates 'qd_builder' objects based on layout specification
%
% Calling object:
%   Single object
%
% Description:
%   This function processes the data in the 'qd_layout' object. First, all tracks in the layout are
%   split into subtracks. Each subtrack corresponds to one segment. Then, then scenario names are
%   parsed. A 'qd_builder' object is created for each scenario and for each transmitter. For
%   example, if there are two BS, each having urban LOS and NLOS users, then 4 'qd_builder' objects
%   will be created (BS1-LOS, BS2-NLOS, BS2-LOS, and BS2-NLOS). The segments are then assigned to
%   the 'qd_builder' objects.
%
% Input:
%   check_parfiles
%   Enables (1, default) or disables (0) the parsing of shortnames and the validity-check for the
%   config-files. This is useful, if you know that the parameters in the files are valid. In this
%   case, this saves some execution time.
%
%   split_tx
%   If set to true (1), each Tx gets assigned to a new builder object. Hence, all LSPs and SSF
%   parameters will be independently generated for each Tx. If set to false (0), Txs belonging to
%   the same scenario will be combined into one builder, enabling spatial consistency for the Txs.
%   The default value is 1, if all Txs are static, and 0, if at least one Tx is mobile (dual-
%   mobility feature).
%
% Output:
%   h_builder
%   A matrix of 'qd_builder' objects. Rows correspond to the scenarios, columns correspond to the
%   transmitters.
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

if numel( h_layout ) > 1
    error('QuaDRiGa:qd_layout:init_builder','init_builder not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

% Parse Input variables
if exist( 'check_parfiles' , 'var' )
    if ~( all(size(check_parfiles) == [1 1]) ...
            && (isnumeric(check_parfiles) || islogical(check_parfiles)) ...
            && any( check_parfiles == [0 1] ) )
        error('QuaDRiGa:qd_layout:init_builder','??? "check_parfiles" must be 0 or 1')
    end
else
    check_parfiles = true;
end

if ~exist( 'split_tx' , 'var' ) || isempty( split_tx )
    % If all Txs are static, we create one builder per Tx. For dual-mobility scenarios, we merge all
    % transmitters into one builder per scenario.
    split_tx = ~h_layout.dual_mobility;
end

% Parse the positions
[ ~, ~, ind, rx_track, tx_track ] = parse_positions( h_layout );
no_seg = size( rx_track,1 );
no_tx  = size( rx_track,2 );
if no_tx > 1
    ind    = ind( :,:,ones(1,no_tx) );
end
for t = 1 : no_tx                                               % Add Tx id to index list
    ind(3,:,t) = t;
end

% Combine all Tx positions into one object if "split_tx" is false
if ~split_tx
    no_seg   = no_seg*no_tx;
    no_tx    = 1;
    ind      = reshape( ind, 3,[] );
    rx_track = qf.reshapeo( rx_track, [no_seg,1] );
    tx_track = qf.reshapeo( tx_track, [no_seg,1] );
end

no_scen = 0;                                                    % Scenario counter
scen_list = {};                                                 % Empty scenario list
tx_array_bld = {};
rx_array_bld = {};
tx_track_bld = {};
rx_track_bld = {};

for t = 1 : no_tx
    rx_track_bld{1,t} = {};
    for s = 1 : no_seg
        
        % We only process links where there is an entry in the pairing matrix
        if any( h_layout.pairing(1,:) == ind(3,s,t) & h_layout.pairing(2,:) == ind(1,s,t) )
            scenario = rx_track(s,t).scenario{1,end};        	% Read scenario from Rx track
            [ ~,si ] = ismember( scenario , scen_list );     	% Get the index in the builder list
            if si == 0                                       	% Create new scenario
                no_scen = no_scen + 1;
                scen_list{1,no_scen} = scenario;               	% Update scenario list
                si = no_scen;                                 	% Get scenario index
                tx_array_bld{si,t} = {};
                rx_array_bld{si,t} = {};
                tx_track_bld{si,t} = {};
                rx_track_bld{si,t} = {};
            end
            if isempty( rx_track_bld{si,t} )
                tx_array_bld{si,t} = h_layout.tx_array(:,t);
                rx_array_bld{si,t} = h_layout.rx_array(:,ind(1,s,t));
                tx_track_bld{si,t} = tx_track(s,t);
                rx_track_bld{si,t} = rx_track(s,t);
            else
                tx_array_bld{si,t}(:,end+1) = h_layout.tx_array(:,t);
                rx_array_bld{si,t}(:,end+1) = h_layout.rx_array(:,ind(1,s,t));
                tx_track_bld{si,t}(1,end+1) = tx_track(s,t);
                rx_track_bld{si,t}(1,end+1) = rx_track(s,t);
            end
        end
    end
end

h_builder = qd_builder;                                         % Initialize builder object
for si = 1 : no_scen
    
    % Create new builder for the first Tx
    h_builder(si,1) = qd_builder( scen_list{1,si}, check_parfiles );
    h_builder(si,1).name = regexprep(scen_list{1,si},'_','-');
    h_builder(si,1).simpar = h_layout.simpar;
    
    % Create builder objects for additional  Txs
    for t = 2 : no_tx
        h_builder(si,t)         = qd_builder( scen_list{1,si}, false, h_builder(si,1).scenpar );
        h_builder(si,t).name    = h_builder(si,1).name;
        h_builder(si,t).simpar  = h_layout.simpar;
        h_builder(si,t).plpar   = h_builder(si,1).plpar;
    end
    
    % Assign data to the builder
    for t = 1 : no_tx
        if ~isempty( rx_track_bld{si,t} )
            h_builder(si,t).tx_array = tx_array_bld{si,t};
            h_builder(si,t).rx_array = rx_array_bld{si,t};
            h_builder(si,t).tx_track = tx_track_bld{si,t};
            h_builder(si,t).rx_track = rx_track_bld{si,t};
        end
        if split_tx
            h_builder(si,t).name = [h_builder(si,t).name,'_',h_layout.tx_track(1,t).name ];
        else
            h_builder(si,t).name = [h_builder(si,t).name,'_*' ];
        end
    end
end

% Check if all builder objects are dual-mobility (or not) and format the data correctly
check_dual_mobility( h_builder );

% Fix for octave
if numel( h_builder ) == 1
    h_builder = h_builder(1,1);
end

end
