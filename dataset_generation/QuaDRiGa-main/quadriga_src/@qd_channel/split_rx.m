function chan_out = split_rx( h_channel, varargin )
%SPLIT_RX Splits channel arrays based on receive antenna indices.
%
% Calling object:
%   Object array
%
% Description:
%   This function can be used to split large receive array antennas into smaller arrays. Example: A
%   channel array has channels from three base stations to three mobile terminals (MTs). However,
%   the MT antennas are merged into one array. To split the channels, the following command can be
%   used:
%        cs = c.split_rx( {1:2,3:4,5,6} );  
%
%   Notes:
%   * The method parses the name-string of the channel objects qd_channel.name in order to
%     determine the Tx-Rx relationship. There are two allowed formats: (a) "tx_rx" and (b)
%     "scenario_tx_rx"
%   * The order of the inputs must match the transmitters in alphabetical order, i.e. the first
%     input corresponds to "Rx01", the second to "Rx02"  and so on. This is independent of the
%     order in "qd_layout.rx_name", which  might have a different order.
%   * If only one cell is given as input, but there are several RXs in the channel array, the
%     same splitting is applied to each one of them.
%   * Outputs are sorted alphabetically according to "tx_rx" (scenario names are ignored)
%   * If the input array is shaped as [ Rx, Tx ], the output will be shaped as [ Rx * SPLT, Tx ]
%
%
% Input:
%   varargin
%   A list of cell-arrays containing the receive antenna indices.
%
% Output:
%   chan_out
%   An array of handles to the split 'qd_channel' objects
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

splt = varargin;

no_c = numel( h_channel );                                  % The number of channels
if size( h_channel,2 ) ~= no_c
    c_in = qf.reshapeo( h_channel, [ 1, no_c ] );           % Reorder channel handles
else
    c_in = h_channel;                                       % Duplicate handles
end

% Parse channel names
[ ~, ~, order, ~,~, tx_ind, rx_ind, tx_names, rx_names ] = parse_channel_names( c_in );
c_in = c_in( 1,order );                                     % Sort input channels

tx_ind_unique = unique( tx_ind );                           % BS indices
rx_ind_unique = unique( rx_ind );                           % MT indices

no_bs = numel( tx_ind_unique );                             % Number of BS
no_rx = numel( rx_ind_unique );                             % Number of MT
no_freq = size( h_channel,3 );                              % Number of frequencies

% Test if input array was sorted
input_is_sorted = false;
if size( h_channel,1 ) == no_rx && size( h_channel,2 )*size( h_channel,3 ) == no_bs
    input_is_sorted = true;
end

% Parse input data
if numel( splt ) ~= no_rx
    if  numel( splt ) == 1
        if ~iscell( splt{1} )
            error('??? Inputs must be a cell array.');
        else
            % Copy the data
            for n = 2:no_rx
                splt{n} = splt{1};
            end
        end
    else
        error('??? Number of inputs does not match the number of RXs in the channel array.');
    end
end

for n = 1 : numel( splt )
    for m = 1 : numel( splt{n} )
        if size( splt{n}{m},1 ) ~= 1
            splt{n}{m} = splt{n}{m}(:)';
        end
        if any( rem( splt{n}{m} , 1 ) ~= 0 )
            error('??? Inputs must be vectors of integer numbers.');
        end
    end
end

chan_out = qd_channel([]);
splt_cnt = 1;
bs_cnt   = 1;
for i_bs = 1 : no_bs
    
    if input_is_sorted
        splt_cnt = 1;
        bs_cnt   = i_bs;
    end
    
    for i_mt = 1 : no_rx
        no_splt = numel( splt{i_mt} );
        
        % Search the correct channel
        i_c = tx_ind == tx_ind_unique( i_bs ) & rx_ind == rx_ind_unique( i_mt );
        if any( i_c )
            i_c = find( i_c, 1);
            
            for i_splt = 1 : no_splt
                rx_splt = splt{i_mt}{i_splt};
                
                chan_out(splt_cnt,bs_cnt) = qd_channel( [] );        % Create empty object
                copy( chan_out(splt_cnt,bs_cnt), c_in( 1,i_c ) );    % Copy data
                
                cf = chan_out(splt_cnt,bs_cnt).coeff( rx_splt,:,:,: );
                ipath = find( ~all(all(all(cf == 0,1),2),4) );
                
                chan_out(splt_cnt,bs_cnt).coeff = cf(:,:,ipath,:);
                if c_in( 1,i_c ).individual_delays
                    chan_out(splt_cnt,bs_cnt).delay = chan_out(splt_cnt,bs_cnt).delay( rx_splt,:,ipath,: );
                end
                
                % Set new channel name including the sector ID
                name = [ tx_names{i_bs},'_',rx_names{i_mt},'s',sprintf('%d', i_splt)  ];
                chan_out(splt_cnt,bs_cnt).name = name;
                
                splt_cnt = splt_cnt + 1;
            end
        end
    end
end
if input_is_sorted
    if no_freq > 1
        chan_out = qf.reshapeo( chan_out , [no_rx*no_splt , no_bs/no_freq , no_freq] );
    end
else
    % Sort channel by name
    n_channel = numel(chan_out);
    names = {};
    for i_channel = 1:n_channel
        names{i_channel} = chan_out(1,i_channel).name;
    end
    [~,ind] = sort(names);
    chan_out = chan_out(1,ind);
end
end
