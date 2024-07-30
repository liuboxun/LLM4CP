function [ pairs, power ] = set_pairing( h_layout, method, threshold, tx_power, overlap, check_parfiles )
%SET_PAIRING Determines links for which channel coefficient are generated.
%
% Calling object:
%   Single object
%
% Description:
%   This function can be used to automatically determine the links for which channel coefficients
%   should be generated. For example, in a large network there are multiple base stations and
%   mobile terminals. The base stations, however, only serve a small area. It the terminal is far
%   away from this area, it will receive only noise from this particular BS. In this case, the
%   channel coefficients will have very little power and do not need to be calculated. Disabling
%   those links can reduce the computation time and the storage requirements for the channel
%   coefficients significantly. There are several methods to do this which can be selected by the
%   input variable 'method'.
%
% Methods:
%   'all'
%   Enables the simulation of all links. Links where the Tx and the Rx are at the same position are
%   are deactivated. 
%
%   'power'
%   Calculates the expected received power taking into account the path loss, the antenna patterns,
%   the LOS polarization, and the receiver orientation. If the power of a link is below the
%   'threshold', it gets deactivated.
%
% Input:
%   method
%   Link selection method. Supported are: 'all', or 'power' (see above)
%
%   threshold
%   If the Rx-power is below the threshold in [dBm], the link gets deactivated
%
%   tx_power
%   A vector of tx-powers in [dBm] for each transmitter in the layout. This power is applied to
%   each transmit antenna in the tx-array antenna. By default (if 'tx_power' is not given), 0 dBm
%   are assumed
%
%   overlap
%   The length of the overlapping part relative to the segment length. It can have values in
%   between 0 (no overlap) and 1 (ramp along the entire segment). The default value is 0.5. You
%   need to make sure that the same value is used when calling "qd_channel.merge". (only used for
%   'sf')
%
%   check_parfiles
%   Disables (0) or enables (1, default) the parsing of shortnames and the validity-check for the
%   config-files. This is useful, if you know that the parameters in the files are valid. In this
%   case, this saves execution time.
%
% Output:
%   pairs
%   An index-list of links for which channel are created. The first row corresponds to the Tx and
%   the second row to the Rx. An identical copy gets assigned to 'qd_layout.pairing'.
%
%   power
%   A matrix containing the estimated receive powers for each link in [dBm]. Rows correspond to the
%   receiving terminal, columns correspond to the transmitter station. For MIMO links, the power of
%   the strongest MIMO sublink is reported.
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
   error('QuaDRiGa:qd_layout:set_scenario','set_scenario not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

if exist( 'method','var' ) && ~isempty( method )
    
    supported_types = { 'all', 'power' };
    
    if ~( ischar(method) && any( strcmpi(method,supported_types)) )
        str = '??? "method" is not supported. Use: ';
        no = numel(supported_types);
        for n = 1:no
            str = [str,supported_types{n}];
            if n<no
                str = [str,', '];
            end
        end
        error(str);
    end
    
else
    method = 'all';
end

% Initialize output
power = [];

% Perform common tasks for profiles "power" and "sf"
switch method
    case { 'power','sf' }
        
        % Check if threshold is given
        if ~exist( 'threshold','var' )
            error('??? "threshold" is undefined.')
        elseif all( size(threshold) == [1 1]) && isnumeric(threshold) && isreal(threshold)
            % OK
        else
            error('??? "threshold" has wrong format.')
        end
        
        % Check if tx-power is given
        if ~exist( 'tx_power','var' ) || isempty( tx_power )
            tx_power = zeros(1,h_layout.no_tx);
        elseif any( size(tx_power) ~= [1 h_layout.no_tx] )
            error(['??? Number of columns in "tx_power" must match the number',...
                ' of transmitters in the layout.'])
        elseif isnumeric( tx_power ) && isreal( tx_power )
            % OK
        else
            error('??? "tx_power" has wrong format.')
        end
        
        % Check if overlap is given
        if ~exist( 'overlap' , 'var' ) || isempty( overlap )
            overlap = 0.5;
        end
        
        % Check if check_parfiles is given
        if ~exist( 'check_parfiles' , 'var' ) || isempty( check_parfiles )
            check_parfiles = true;
        end
        
        % Test for multi-frequency simulations
        if numel( h_layout.simpar(1,1).center_frequency ) > 1
            error('??? Multi-frequency simulations are not allowed when pairing by "power".')
        end
        
        % Test if tracks have segments
        if any( cat(1,h_layout.rx_track.no_segments) ~= 1 )
            error('??? Tracks must not have segments when pairing by "power".')
        end
        
        % Reset the pairing matrix
        h_layout.set_pairing('all');
        
end


% Implement the code for "power" and "sf"
switch method
    case 'all'
        h_builder = [];
        
    case 'power'
        % Get the parameter sets, but do not create the maps
        h_builder = h_layout.init_builder( check_parfiles );
end


switch method
    case 'power'
        % Initialize power-matrix
        power = zeros( h_layout.no_rx, h_layout.no_tx );
        
        rx_name = h_layout.rx_name;
        tx_name = h_layout.tx_name;
              
        h_channel = get_los_channels( h_builder );
        sic = size( h_builder );
       
        for i_builder = 1 : numel( h_builder )
            [ i1,i2 ] = qf.qind2sub( sic, i_builder );
                      
            % Get the powers and maximize over all Tx and Rx antennas
            P = abs( h_channel(i1,i2).coeff ).^2;
            P = reshape( P, h_channel(i1,i2).no_rxant*h_channel(i1,i2).no_txant , h_channel(i1,i2).no_snap );
            P = max(P,[],1);
            
            % Parse tx index
            tmp = regexp( h_channel(i1,i2).name , '_' );
            tx_name_local = h_channel(i1,i2).name( tmp+1:end );
            tx_ind = strcmp( tx_name , tx_name_local );
            
            % Write power values to the "power" matrix
            for i_mt = 1 : h_channel(i1,i2).no_snap
                
                % Parse rx index
                rx_name_local = h_builder(i1,i2).rx_track(1,i_mt).name;
                rx_ind = strcmp( rx_name , rx_name_local );
                
                power( rx_ind, tx_ind ) = P( i_mt );
            end
        end
        power = 10*log10( power );
        power = power + tx_power( ones(1,h_layout.no_rx),: );
end

% Assemble the link matrix
switch method
    case 'all'
        pairs = zeros( 2, h_layout.no_tx*h_layout.no_rx );
        o_rx  = ones( 1,h_layout.no_rx);
        i_end = 0;
        rx_position = h_layout.rx_position;
        tx_position = h_layout.tx_position;
        for i_tx = 1 : h_layout.no_tx
            dist = sqrt(sum((rx_position - tx_position(:,i_tx*o_rx)).^2));
            ind_mt = find( dist > 1e-4 );
            ind_pairs = ( 1 : numel(ind_mt) ) + i_end ;
            
            pairs( 1, ind_pairs ) = i_tx;
            pairs( 2, ind_pairs ) = ind_mt;
            
            i_end = numel(ind_mt)+i_end;
        end
        pairs = pairs(:,1:i_end);
        
    case 'power'
        pairs = zeros( 2, h_layout.no_tx*h_layout.no_rx );
        i_end = 0;
        for i_tx = 1 : h_layout.no_tx
            ind_mt = find( power(:,i_tx) > threshold );
            ind_pairs = ( 1 : numel(ind_mt) ) + i_end ;
            
            pairs( 1, ind_pairs ) = i_tx;
            pairs( 2, ind_pairs ) = ind_mt;
            
            i_end = numel(ind_mt)+i_end;
        end
        pairs = pairs(:,1:i_end);
end

h_layout.Ppairing = pairs;

end
