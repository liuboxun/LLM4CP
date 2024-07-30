function swap_tx_rx( h_channel, swap_coeff, swap_pos, swap_name, swap_par )
%SWAP_TX_RX Swaps the TX and RX
%
% Calling object:
%   Object array
%
% Description:
%   This method exchanges the transmitter and and the receiver, e.g. for transforming an uplink
%   channel into a downlink channel. This includes transposing coefficient and delay matrix,
%   exchanging the names of the TX and RX in the channel name string, exchanging the positions and
%   changing the variable names in the "par" structure (e.g. to change AoA to AoD and vice-versa).
%
% Input:
%   swap_coeff
%   If set to true (default), the channel coefficients and delay matrix are transposed.
%
%   swap_pos
%   If set to true (default), the tx and rx position are exchanged.
%
%   swap_name
%   If set to true (default), the channel name string is updated.
%
%   swap_par
%   If set to true (default), the variable names in "par" are exchanged.
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

if ~exist('swap_coeff','var') || isempty( swap_coeff )
    swap_coeff = true;
end

if ~exist('swap_pos','var') || isempty( swap_pos )
    swap_pos = true;
end

if ~exist('swap_name','var') || isempty( swap_name )
    swap_name = true;
end

if ~exist('swap_par','var') || isempty( swap_par )
    swap_par = true;
end

if numel(h_channel) > 1
    
    sic = size( h_channel );
    for i_cb = 1 : numel(h_channel)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        swap_tx_rx( h_channel( i1,i2,i3,i4 ), swap_coeff, swap_pos, swap_name, swap_par );
    end
    
else
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_channel = h_channel(1,1);
    
    % Swap delays
    if h_channel.individual_delays && swap_coeff
        h_channel.Pdelay = permute( h_channel.Pdelay, [2,1,3,4] );
    end
    
    % Swap coeffs
    if swap_coeff
        h_channel.Pcoeff = permute( h_channel.Pcoeff, [2,1,3,4] );
    end    
    
    % Swap tx and Rx positions
    if swap_pos
        tmp = h_channel.Ptx_position;
        h_channel.Ptx_position = h_channel.Prx_position;
        h_channel.Prx_position = tmp;
    end
    
    % Swap name
    if swap_name
        name = h_channel.name;
        tmp = regexp( name , '_' );                 % split "Scen_Tx_Rx_Seg"
        if numel( tmp ) == 1                        % "Tx_Rx"
            tx = name(1:tmp(1)-1);
            rx = name(tmp(1)+1:end);
            name = [rx,'_',tx];
        elseif numel( tmp ) == 2                    % "Scen_Tx_Rx"
            scen = name(1:tmp(1)-1);
            tx = name(tmp(1)+1:tmp(2)-1);
            rx = name(tmp(2)+1:end);
            name = [scen,'_',rx,'_',tx];
        elseif numel( tmp ) == 3                    % "Scen_Tx_Rx_Seg"
            scen = name(1:tmp(1)-1);
            tx = name(tmp(1)+1:tmp(2)-1);
            rx = name(tmp(2)+1:tmp(3)-1);
            seg = name(tmp(3)+1:end);
            name = [scen,'_',rx,'_',tx,'_',seg];
        end
        h_channel.name = name;
    end

    if swap_par
        tokens = {'asD','asA';'asA','asD';...
            'esD','esA';'esA','esD';...
            'AoA','AoD';'AoD','AoA';...
            'EoA','EoD';'EoD','EoA';...
            'fbs_pos','lbs_pos';'lbs_pos','fbs_pos';...
            'Jones_A','Jones_D';'Jones_D','Jones_A';...
            'mse_A','mse_D';'mse_D','mse_A';...
            'resolved_A','resolved_D';'resolved_D','resolved_A';...
            'pow_A','pow_D';'pow_D','pow_A';...
            'xpr_A','xpr_D';'xpr_D','xpr_A'};
        par = struct;
        if ~isempty( h_channel.par )
            fields = fieldnames( h_channel.par );
            for iF = 1 : numel( fields )
                has_token = 0;
                for iT = 1 : size( tokens,1 )
                    if ~isempty( strfind( fields{iF}, tokens{iT,1} ) )
                        has_token = iT;
                    end
                end
                if has_token
                    tmp = regexprep( fields{iF}, tokens{has_token,1}, tokens{has_token,2} );
                    par.(tmp) = h_channel.par.(fields{iF});
                else
                    par.(fields{iF}) = h_channel.par.(fields{iF});
                end
            end
        end
        h_channel.par = par;
    end
end

end
