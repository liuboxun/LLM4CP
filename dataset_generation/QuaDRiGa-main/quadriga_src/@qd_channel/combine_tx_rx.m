function h_channel_comb = combine_tx_rx( h_channel )
%COMBINE_TX_RX Combines channels from a qd_channel array into a single object
%
% Calling object:
%   Object array
%
% Description:
%   This method operates on an array of qd_channel objects containing channels from several BSs to
%   several MTs. It then merges these channels into a single qd_channel object. For example, if you
%   use QuaDRiGa to generate channels from two single-antenna BSs to one single-antenna MT, you get
%   an two-element array of qd_channel objects. The first object contains the SISO channel from BS1
%   to the MT, the second object contains the SISO channel from BS2 to the MT. However, for some
%   evaluations, it would be preferable to have a MISO channel where the two BSs operate as a
%   distributed-antenna system. This is done by this method. If you call "combine_tx_rx" on the
%   qd_channel array, you get a MISO channel object with two transmit antennas and one receive
%   antenna as output. Note that for mobile receivers, all individual channel objects must have the
%   same number of snapshots.
%
% Output:
%   h_channel_comb
%   Single qd_channel object containing the channel coefficients of the combined channel.
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

no_c = numel( h_channel );                                  % The number of channels
if size( h_channel,2 ) ~= no_c
    c_in = qf.reshapeo( h_channel, [ 1, no_c ] );           % Reorder channel handles
else
    c_in = h_channel;                                       % Duplicate handles
end

% Test if all channels have the same number of snapshots
no_snap = c_in(1,1).no_snap;
center_frequency = c_in(1,1).center_frequency;
for ic = 2 : no_c
    if c_in(1,ic).no_snap ~= no_snap
        error('QuaDRiGa:qd_channel:combine_tx_rx','All channel objects must have the same number of snapshots.')
    end
    if c_in(1,ic).center_frequency ~= center_frequency
        error('QuaDRiGa:qd_channel:combine_tx_rx','All channel objects must have the same center frequency.')
    end
end

% Make sure that all channels have individual delays enabled
for ic = 1 : no_c
    if ~c_in( 1,ic ).individual_delays
        c_in( 1,ic ) = copy( c_in( 1,ic ) );
        c_in( 1,ic ).individual_delays = true;
    end
end

% Parse channel names
[ ~, ~, order, ~,~, tx_ind, rx_ind, tx_names, rx_names ] = parse_channel_names( c_in );
c_in = c_in( 1,order );                                     % Sort input channels

% Extract the number of MBs and MTs
no_tx = max( tx_ind );
no_rx = max( rx_ind );

no_rx_ant = zeros( 1,no_rx );
no_tx_ant = zeros( 1,no_tx );
max_no_path = 0;
min_SD = inf;
tx_rx_names_ambiguous = false;
for r = 1 : no_rx
    for t = 1 : no_tx
        ii = find( tx_ind == t & rx_ind == r );
        if numel( ii ) > 1
            tx_rx_names_ambiguous = true;
            ii = ii(1);
        end
        if ~isempty( ii )
            if c_in(1,ii).no_path > max_no_path                   % Determine the maximum number of paths
                max_no_path = c_in(1,ii).no_path;
            end
            if no_tx_ant(t) == 0                                        % Determine the number of BS antennas
                no_tx_ant(t) = c_in(1,ii).no_txant;
            elseif no_tx_ant(t) ~= c_in(1,ii).no_txant
                error('QuaDRiGa:qd_channel:combine_tx_rx','Inconsistent number of Tx antennas.')
            end
            if no_rx_ant(r) == 0                                           % Determine the number of MT antennas
                no_rx_ant(r) = c_in(1,ii).no_rxant;
                
                % Calculate the sample density
                pos = c_in(1,ii).rx_position;
                if size( c_in(1,ii).tx_position,2 ) == size( pos,2 )
                    pos = pos - c_in(1,ii).tx_position;
                end
                d_pos = diff( pos,[],2 );
                d_dist = sqrt(sum(d_pos.^2,1));
                mobile_speed_max = max( d_dist );
                SD = qd_simulation_parameters.speed_of_light / (2*mobile_speed_max*center_frequency);
                if min_SD > SD
                    min_SD = SD;
                    rx_position = pos;
                end
                
            elseif no_rx_ant(r) ~= c_in(1,ii).no_rxant
                error('QuaDRiGa:qd_channel:combine_tx_rx','Inconsistent number of Rx antennas.')
            end
        end
    end
end
if tx_rx_names_ambiguous
    warning('QuaDRiGa:qd_channel:combine_tx_rx:names_ambiguous',...
        'TX and RX names are not unique. Channels with ambiguous names are not stored.');
end

% Combine coefficients and delays
coeff = zeros( sum( no_rx_ant ), sum( no_tx_ant ), max_no_path, no_snap );
delay = coeff;
ir = 1;
for r = 1 : no_rx
    it = 1;
    for t = 1 : no_tx
        ii = find( tx_ind == t & rx_ind == r, 1 );
        if ~isempty( ii )
            coeff( ir:ir+no_rx_ant(r)-1, it:it+no_tx_ant(t)-1, 1:c_in(1,ii).no_path, : ) = c_in(1,ii).coeff;
            delay( ir:ir+no_rx_ant(r)-1, it:it+no_tx_ant(t)-1, 1:c_in(1,ii).no_path, : ) = c_in(1,ii).delay;
        end
        it = it+no_tx_ant(t);
    end
    ir = ir+no_rx_ant(r);
end

% Write output data
h_channel_comb = qd_channel( coeff, delay );
h_channel_comb.center_frequency = c_in(1,1).center_frequency;
h_channel_comb.version = c_in(1,1).version;

% Write port description
str = 'Input ports:\r\n';
ip = 1;
for n = 1 : no_tx
    for m = 1 : no_tx_ant(n)
       str = [str,'   ',num2str(ip,'%02d'),' : ',tx_names{n}];
       if no_tx_ant(n) > 1
           str = [str,' - Ant. ',num2str(m)];
       end
       str = [str,'\r\n'];
       ip = ip+1;
    end
end
str = [str,'\r\nOutput ports:\r\n'];
ip = 1;
for n = 1 : no_rx
    for m = 1 : no_tx_ant(n)
       str = [str,'   ',num2str(ip,'%02d'),' : ',rx_names{n}];
       if no_rx_ant(n) > 1
           str = [str,' - Ant. ',num2str(m)];
       end
       str = [str,'\r\n'];
       ip = ip+1;
    end
end
h_channel_comb.par = struct('port_dsc',str);

% We need to store the Rx positions of the fastest moving user (this is needed for exporting the
% channels to the Propsim emulator)
if size(rx_position,2) ~= no_snap
    error('QuaDRiGa:qd_channel:combine_tx_rx','Number of Rx positions does not match number of snapshots.')
end
h_channel_comb.rx_position = rx_position;

end
