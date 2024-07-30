classdef qd_channel < handle
%CHANNEL Class for handling channel coefficients
%
% DESCRIPTION
% Objects of this class are the output of the channel model. They are
% created by the 'channel_builder'. By default, channel
% coefficients are provided in time domain, as a list of delays and
% complex-valued amplitudes. However, this class also implements certain
% methods to post-process the channel data. Those include:     
%
%   - Transformation into frequency domain
%   - Interpolation in time domain (to change the terminal speed and
%     sampling rate) 
%   - Combining channel traces into longer segments (including birth and
%     death of clusters) 
%
% Optional PLUGIN: quadriga_channel_export
% In addition to the open-source version of QuaDRiGa, the "channel export
% plugin" provides import- and export functions for commonly used channel
% formats. This plugin adds additional methods to the "channel":
%
%    hdf5_load - Load data from stored HDF5 file
%    hdf5_save - Save data to HDF5 file
%    mport_meas_data - Convert band-limited frequency-domain data into QuaDRiGa channels
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
    
properties
    % Name of the 'channel' object.
    %
    %   This string is a unique identifier of the channel object. The
    %   'channel_builder' creates one channel object for each MT,
    %   each Tx and each segment. They are further grouped by scenarios
    %   (propagation environments). The string consists of four parts
    %   separated by an underscore '_'. Those are:
    %
    %       - The scenario name from 'track.scenario'
    %       - The transmitter name from 'layout.tx_name'
    %       - The receiver name from 'layout.rx_name'
    %       - The segment number
    %
    %   After 'channel.merge' has been called, the name string consists of:
    %
    %       - The transmitter name from 'layout.tx_name'
    %       - The receiver name from 'layout.rx_name'
    %
    name = 'New_channel';
    
    % Version number of the QuaDRiGa release that was used to create
    % the 'channel' object.
    version = qd_simulation_parameters.version;
    
    % Center frequency in [Hz]
    center_frequency = 0;
end

properties(Dependent)   
    % The complex-valued channel coefficients for each path.
    %   The indices of the 4-D tensor are:
    %   [ Rx-Antenna , Tx-Antenna , Path , Snapshot ]
    coeff
    
    % The delays for each path.
    %
    %   There are two different options. If the delays are identical on
    %   the MIMO links, i.e. 'individual_delays = 0', then 'delay' is a
    %   2-D matrix with dimensions [ Path , Snapshot ].
    %
    %   If the delays are different on the MIMO links, then 'delay' is
    %   a 4-D tensor with dimensions
    %   [ Rx-Antenna , Tx-Antenna , Path , Snapshot ].
    delay
    
    % A flexible data structure that containing additional data
    %   This structure allows you to store additional data with the channel
    %   object. Separate variables can be identified by fieldnames 
    %   (e.g. "par.sf" or "par.snr"). Each field must contain numeric data
    %   (real or complex numbers) or strings. Nested structures are not
    %   allowed (e.g. "par.sf.val1").  
    par
    
    % The snapshot number for which the initial LSPs have been generated.
    %   Normally, this is the first snapshot. However, if the user
    %   trajectory consists of more than one segment, then
    %   'initial_position' points to the snapshot number where the
    %   current segment starts. For example: If 'initial_position' is
    %   100, then snapshots 1-99 are overlapping with the previous segment.
    initial_position
    
    % Position of each Tx in global Cartesian coordinates using units
    % of [m].
    tx_position
    
    % The receiver position global Cartesian coordinates using units of
    % [m] for each snapshot.
    rx_position
end
    
properties(Dependent,SetAccess=protected)    
    no_rxant                    % Number of receive elements (read only)
    no_txant                    % Number of transmit elements (read only)
    no_path                     % Number of paths (read only)
    no_snap                     % Number of snapshots (read only)
end

% Informative parameters
properties(Dependent)
    % Indicates if the path delays are identical on each MIMO link
    % (0) or if each link has different delays (1).
    individual_delays
end

% Data storage
properties(Access=private)
    Pcoeff              = [];
    Pdelay              = [];
    Ppar                = [];
    Pinitial_position   = [];
    Ptx_position        = [];
    Prx_position        = [];
end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
end

methods
    % Constructor
    function h_channel = qd_channel( Ccoeff , Cdelay , Cinitial_position )
        if exist( 'Ccoeff','var' ) && ~isempty(Ccoeff)
            [ no_rxant, no_txant, no_path, no_snap] = size(Ccoeff);
            h_channel.Pcoeff = Ccoeff;
            if exist( 'Cdelay', 'var' ) && ~isempty(Cdelay)
                if ( numel(size(Cdelay)) == 2 && size(Cdelay,1) == no_path && size(Cdelay,2) == no_snap ) || ...
                        ( numel(size(Cdelay)) == 2 && size(Cdelay,1) == no_rxant && size(Cdelay,2) == no_txant ) || ...
                        ( numel(size(Cdelay)) == 3 && size(Cdelay,1) == no_rxant && size(Cdelay,2) == no_txant && size(Cdelay,3) == no_path ) || ...
                        ( numel(size(Cdelay)) == 4 && size(Cdelay,1) == no_rxant && size(Cdelay,2) == no_txant && ...
                        size(Cdelay,3) == no_path && size(Cdelay,4) == no_snap )
                    h_channel.Pdelay = Cdelay;
                else
                    error('QuaDRiGa:qd_channel:WrongInput','??? "Cdelay" does not match size of "Ccoeff".')
                end
            else
                h_channel.Pdelay = zeros( no_path,no_snap );
            end
            if exist( 'Cinitial_position', 'var' ) && ~isempty(Cinitial_position)
                h_channel.Pinitial_position = Cinitial_position;
            else
                h_channel.Pinitial_position = 1;
            end
        end
    end
    
    % Get functions
    function out = get.coeff(h_channel)
        out = h_channel(1,1).Pcoeff;
    end
    function out = get.delay(h_channel)
        out = h_channel(1,1).Pdelay;
    end
    function out = get.par(h_channel)
        out = h_channel(1,1).Ppar;
    end
    function out = get.initial_position(h_channel)
        out = h_channel(1,1).Pinitial_position;
    end
    function out = get.tx_position(h_channel)
        out = h_channel(1,1).Ptx_position;
    end
    function out = get.rx_position(h_channel)
        out = h_channel(1,1).Prx_position;
    end
    function out = get.no_rxant(h_channel)
        out = size( h_channel(1,1).Pcoeff,1);
    end
    function out = get.individual_delays(h_channel)
        if numel( size( h_channel(1,1).Pdelay) ) == 4 || numel( size( h_channel(1,1).Pdelay) ) == 3 || ...
                ( numel( size( h_channel(1,1).Pcoeff) ) == 2 && all(size( h_channel(1,1).Pcoeff) == size( h_channel(1,1).Pdelay)) )
            out = true;
        else
            out = false;
        end
    end
    function out = get.no_txant(h_channel)
        out = size( h_channel(1,1).Pcoeff,2);
    end
    function out = get.no_path(h_channel)
        s = size(h_channel.Pcoeff);
        if numel(s) < 3
            if ~any(s)
                out = 0;
            else
                out = 1;
            end
        else
            out = s(3);
        end
    end
    function out = get.no_snap(h_channel)
        s = size(h_channel(1,1).Pcoeff);
        if numel(s) < 4
            if ~any(s)
                out = 0;
            else
                out = 1;
            end
        else
            out = s(4);
        end
    end
    
    function v = get_version(h_channel)
        %GET_VERSION returns the version number
        min_ver = Inf;
        for n = 1 : size(h_channel,1)
            for m = 1 : size(h_channel,2)
                tmp = regexp( h_channel(n,m).version,'(?<a>[0-9]+).(?<b>[0-9]+).(?<c>[0-9]+)-(?<d>[0-9]+)' ,'tokens');
                tmp = str2double(tmp{1});
                tmp_ver = tmp(1)*1e6 + tmp(2)*1e3 + tmp(3);
                if tmp_ver < min_ver
                    v = tmp;
                    min_ver = tmp_ver;
                end
            end
        end
    end
    
    % Set functions
    function set.name(h_channel,value)
        if ~( ischar(value) )
            error('??? "name" must be a string.')
        end
        h_channel(1,1).name = value;
    end
    
    function set.version(h_channel,value)
        if ~( ischar(value) )
            error('??? "version" must be a string.')
        elseif isempty( regexp(value,'[0-9]+.[0-9]+.[0-9]+-[0-9]+', 'once') )
            error('??? "version" must be a version-string.')
        end
        h_channel(1,1).version = value;
    end
    
    function set.individual_delays(h_channel,value)
        if ~( all(size(value) == [1 1]) && isreal(value) )
            error('??? "individual_delays" must be numeric and scalar')
        end
        value = logical( value );
        if h_channel(1,1).individual_delays && ~value && ~isempty(h_channel(1,1).Pdelay)
            
            if h_channel(1,1).no_txant == 1 && h_channel(1,1).no_rxant == 1
                h_channel(1,1).Pdelay = reshape( h_channel(1,1).Pdelay(1,1,:,:) , h_channel(1,1).no_path , h_channel(1,1).no_snap );
            else
                % Calculate power-weighted mean
                P = abs(h_channel(1,1).Pcoeff).^2;
                P = reshape( P, h_channel(1,1).no_rxant*h_channel(1,1).no_txant , h_channel(1,1).no_path , h_channel(1,1).no_snap );
                tmp = sum( P , 1 );
                P = P ./ tmp( ones(1,h_channel(1,1).no_rxant*h_channel(1,1).no_txant),:,: );
                
                D = reshape( h_channel(1,1).Pdelay, h_channel(1,1).no_rxant*h_channel(1,1).no_txant , h_channel(1,1).no_path , h_channel(1,1).no_snap );
                D = sum( D.*P,1 );
                D( tmp == 0 ) = 0;
                D = permute( D , [2,3,1] );
                
                h_channel(1,1).Pdelay = D;
            end
            
        elseif ~h_channel(1,1).individual_delays && value && ~isempty(h_channel(1,1).Pdelay)
            % Use the same delay for all antenna elements
            tmp = reshape( h_channel(1,1).Pdelay,1,1,h_channel(1,1).no_path,h_channel(1,1).no_snap );
            h_channel(1,1).Pdelay = tmp( ones(1,h_channel(1,1).no_rxant) , ones(1,h_channel(1,1).no_txant) , : , :  );
        end
    end
    
    function set.par(h_channel,value)
        if ~isempty( value )
            if ~isstruct( value )
                error('??? "par" must be a structure')
            else
                names = fieldnames( value );
                for i_names = 1:numel( names )
                    if isstruct( value.( names{i_names} ) )
                        error('??? "par" cannot contain nested structures');
                    elseif iscell( value.( names{i_names} ) )
                        error('??? "par" cannot contain cell arrays');
                    end
                end
                h_channel(1,1).Ppar = value;
            end
        else
            h_channel(1,1).Ppar = [];
        end
    end
    
    function set.initial_position(h_channel,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                && isreal(value) && mod(value,1)==0 && value >= 0 )
            error('??? "initial_position" must be integer and >= 0')
        elseif value > h_channel(1,1).no_snap
            error('??? "initial_position" must not exceed the number of snapshots')
        end
        h_channel(1,1).Pinitial_position = value;
    end
    
    function set.rx_position(h_channel,value)
        if isempty( value)
            h_channel(1,1).Prx_position = [];
        else
            if ~( isnumeric(value) && isreal(value) && size(value,2) > 0 )
                error('QuaDRiGa:Channel:wrongInputValue','??? "rx_position" must consist of real numbers')
            elseif ~all( size(value,1) == 3 )
                error('QuaDRiGa:Channel:wrongInputValue','??? "rx_position" must have three rows.')
            end
            h_channel(1,1).Prx_position = value;
        end
    end
    
    function set.tx_position(h_channel,value)
        if isempty( value)
            h_channel(1,1).Ptx_position = [];
        else
            if ~( isnumeric(value) && isreal(value) )
                error('QuaDRiGa:Channel:wrongInputValue','??? "tx_position" must consist of real numbers')
            elseif ~all( size(value,1) == 3 )
                error('QuaDRiGa:Channel:wrongInputValue','??? "tx_position" must have 3 rows')
            end
            h_channel(1,1).Ptx_position = value;
        end
    end
    
    function set.coeff(h_channel,value)
        h_channel.Pcoeff = value;
    end
    function set.delay(h_channel,value)
        h_channel.Pdelay = value;
    end
end

methods(Static)
    [ h_channel, dims ] = hdf5_load( varargin )
    [ h_channel, dims ] = mat_load( filename, varargin )
    [ h_channel, snr, pdp ] = import_meas_data( Y, B, L_max, usage, noise_limit, delay_limit, pilot_grid, verbose, show_pdp )
    h_channel = fr2cir( freq_response, bandwidth, no_path, min_pow, pilot_grid, delay_range, use_gpu, no_chunk, verbose  )
    varargout = call_private_fcn( functionName, varargin )
    h_channel = process_data( Y, h_arrayant, no_path, no_subpath, search_range, tx_pos, rx_pos, bandwidth, pilot_grid, no_chunk, verbose, plot_title, plot_filename, plot_ylim, plot_PDPgain )
    h_channel = process_data2( Y, h_arrayant, no_path, no_subpath, search_range, tx_pos, rx_pos, bandwidth, pilot_grid, no_chunk, verbose, plot_title, plot_filename, plot_ylim, plot_PDPgain )
end

end
