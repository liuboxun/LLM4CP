function mat_save( h_channel, filename, version )
%MAT_SAVE Save qd_channel object array to a mat-file
%
% Calling object:
%   Object array
%
% Input:
%   filename
%   Name of file, specified as a character vector or string scalar
%
%   version
%   MAT-file version. Default is "-v6"
%
%
% QuaDRiGa Copyright (C) 2011-2021
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

if ~exist('filename','var') || isempty( filename )
    error('QuaDRiGa:qd_channel:mat_save','??? Filename is missing')
end

if ~exist('version','var') || isempty( version )
    version = '-v6';
end

% Empty struct for the channel data
chn = struct;

% Dimensions of the channel object
no_existing_channels = 0;
new_order = reshape( 1:numel( h_channel ) , size(h_channel ) );
channel_dims = size( new_order );
channel_dims = [channel_dims ones(1,4-numel(channel_dims))];
chn.ChannelDims = int64( channel_dims );

% Save the order of the channels
order = reshape( new_order , 1 , [] );
order = [order,zeros(1, 65535-numel(order) )];
chn.Order = uint16( order );

% Save the channel objects
sic = size( h_channel );
for n = 1 : prod( sic )
    [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
    channel_no = no_existing_channels + n;
    
    % Name
    var_name = ['channel_',num2str(channel_no,'%05.0f'),'_Name'];
    chn.(var_name) = h_channel(i1,i2,i3,i4).name;
    
    % Version
    var_name = ['channel_',num2str(channel_no,'%05.0f'),'_Version'];
    chn.(var_name) = h_channel(i1,i2,i3,i4).version;
    
    % Center Frequency
    if ~isempty(  h_channel(i1,i2,i3,i4).center_frequency )
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_CenterFrequency'];
        chn.(var_name) = single( h_channel(i1,i2,i3,i4).center_frequency );
    end
    
    % Channel Coefficients
    if ~isempty(  h_channel(i1,i2,i3,i4).coeff )
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_Coeffs'];
        chn.(var_name) = single( h_channel(i1,i2,i3,i4).coeff );
       
        % Initial Position
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_Initial_position'];
        chn.(var_name) = uint32( h_channel(i1,i2,i3,i4).initial_position );
    end
    
    % Delays
    if ~isempty(  h_channel(i1,i2,i3,i4).delay )
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_Delays'];
        chn.(var_name) = single( h_channel(i1,i2,i3,i4).delay );
        
        % Individual Delays
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_IndividualDelays'];
        chn.(var_name) = logical( h_channel(i1,i2,i3,i4).individual_delays );
    end
    
    % Write the rx-position only if it is specified in the channel object
    if ~isempty( h_channel(i1,i2,i3,i4).rx_position )
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_rx_position'];
        chn.(var_name) = single( h_channel(i1,i2,i3,i4).rx_position );
    end
    
    % Write the tx-position only if it is specified in the channel object
    if ~isempty( h_channel(i1,i2,i3,i4).tx_position )
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_tx_position'];
        chn.(var_name) = single( h_channel(i1,i2,i3,i4).tx_position );
    end
    
    % Write the par-struct
    if ~isempty( h_channel(i1,i2,i3,i4).par )
        names = fieldnames( h_channel(i1,i2,i3,i4).par );
        
        % Save list of variables in PAR
        var_name = ['channel_',num2str(channel_no,'%05.0f'),'_PARnames'];
        chn.(var_name) = names;
        
        for i_par = 1 : numel( names )
            var_name = ['channel_',num2str(channel_no,'%05.0f'),'_par_',names{i_par}];
            if isstruct( h_channel(i1,i2,i3,i4).par.(names{i_par}))  % Struct
                warning('QuaDRiGa:qd_channel:mat_save',...
                    'Cannot write nested structure in "par" to HDF5 file.')
                
            elseif iscell( h_channel(i1,i2,i3,i4).par.(names{i_par}))  % Struct
                warning('QuaDRiGa:qd_channel:mat_save',...
                    'Cannot write cell arrays in "par" to HDF5 file.')
                
            elseif ischar( h_channel(i1,i2,i3,i4).par.(names{i_par})) || ...        % Strings
                    isinteger( h_channel(i1,i2,i3,i4).par.(names{i_par})) || ...    % Integer Numbers
                    islogical( h_channel(i1,i2,i3,i4).par.(names{i_par}))           % Logical numbers
                chn.(var_name) = h_channel(i1,i2,i3,i4).par.(names{i_par});
                
            else % Numeric data types
                chn.(var_name) = single( h_channel(i1,i2,i3,i4).par.(names{i_par}) );
            end
        end
    end
end

save(filename,'-struct','chn',version);

end
