classdef qd_layout < handle
%QD_LAYOUT Network layout definition class
%
% DESCRIPTION
% Objects of this class define the network layout of a simulation run. Each
% network layout has one or more transmitters and one or more receivers. Each
% transmitter and each receiver need to be equipped with an array antenna which
% is defined by the qd_arrayant class. In general, we assume that the transmitter is at
% a fixed position and the receiver is mobile. Thus, each receivers movement is
% described by a track.
%
% EXAMPLE
%
%    a = qd_arrayant('dipole');               % Generate dipole array antenna
%
%    l = qd_layout;                           % Create new layout
%    l.simpar.center_frequency = 2.1e9;       % Set simulation parameters
%    l.simpar.sample_density = 8;             % Set sample density
%
%    l.no_tx = 2;                             % We want two Tx
%    l.tx_position = [-50 50 ; 0 0 ; 30 30];  % Tx are at 30m height and 100m apart
%    l.tx_array = a;                          % All Tx have a dipole antenna
%
%    l.no_rx = 10;                            % 10 Receivers
%    l.randomize_rx_positions( 300,1,2 );     % Rx radius: 300m, height: 1-2m
%    l.rx_track.set_scenario({'C2l','C2n'});  % Assign scenarios to the Rx
%    l.rx_array = a;                          % All Rx have a dipole antenna
%
%    l.set_pairing;                           % Evaluate all links
%    c = l.get_channels;                      % Generate input for channel_builder
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
    name = 'Layout';          	% Name of the layout
    simpar = qd_simulation_parameters;  	% Handle of a 'simulation_parameters' object
    update_rate = [];           % Channel update rate in seconds
    ReferenceCoord = [];        % Reference coordinates for KML read/write
end

properties(Dependent)
    no_tx                       % Number of transmitters (or base stations)
    no_rx                     	% Number of receivers (or mobile terminals)
    tx_name                    	% Identifier of each Tx, must be unique
    tx_position               	% Position of each Tx in global Cartesian coordinates using units of [m]
    tx_array                  	% Handles of 'qd_arrayant' objects for each Tx
    tx_track                    % Handles of track objects for each Tx
    rx_name                   	% Identifier of each Tx, must be unique
    rx_position               	% Initial position of each Rx (relative to track start)
    rx_array                  	% Handles of qd_arrayant objects for each Rx
    rx_track                    % Handles of track objects for each Rx
    
    % An index-list of links for which channel are created. The first
    % row corresponds to the Tx and the second row to the Rx.
    pairing
    
    % Indicator if the layout contains moving transmitters
    dual_mobility
end

properties(Dependent,Hidden)
    track                       % Handles of track objects for each Rx (legacy option)
end

properties(Dependent,SetAccess=protected)
    % Number of links for which channel coefficients are created (read only)
    no_links
end

properties(Access=private)
    Ptx_array       = qd_arrayant('omni');
    Ptx_track       = qd_track([]);
    Prx_array       = qd_arrayant('omni');
    Prx_track       = qd_track([]);
    Ppairing        = [1;1];
    Pdual_mobility  = false;
end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
    h_qd_builder_init = [];
    use_channel_interpolation = [];
    builder_index = [];
    track_checksum = [];
end

methods
    % Constructor
    function h_layout = qd_layout( simpar )
        
        % At some times, MATLAB seems to use old values from the memory
        % when constructing a new layout. We prevent this by assigning
        % the default settings here.
        h_layout.name = 'Layout';
        h_layout.Prx_track = qd_track([]);
        h_layout.Prx_track.name = 'Rx0001';
        h_layout.Ptx_track = qd_track([]);        
        h_layout.Ptx_track.initial_position = [0;0;25];
        h_layout.Ptx_track.name = 'Tx0001';
        h_layout.Ptx_array = qd_arrayant('omni');
        h_layout.Prx_array = qd_arrayant('omni');
        
        if nargin >= 1
            h_layout.simpar = simpar;
        else
            h_layout.simpar = qd_simulation_parameters;
        end
    end
    
    % Get functions
    function out = get.no_tx(h_layout)
        out = numel( h_layout.Ptx_track );
    end
    function out = get.no_rx(h_layout)
        out = numel( h_layout.Prx_track );
    end
    function out = get.tx_name(h_layout)
        out = cat( 2 , {h_layout.Ptx_track.name} );
    end
    function out = get.tx_position(h_layout)
        out = cat( 2, h_layout.Ptx_track.initial_position );
    end
    function out = get.tx_array(h_layout)
        out = h_layout.Ptx_array;
    end
    function out = get.tx_track(h_layout)
        out = h_layout.Ptx_track;
    end
    function out = get.rx_name(h_layout)
        out = cat( 2 , {h_layout.Prx_track.name} );
    end
    function out = get.rx_position(h_layout)
        out = cat( 2, h_layout.Prx_track.initial_position );
    end
    function out = get.rx_array(h_layout)
        out = h_layout.Prx_array;
    end
    function out = get.rx_track(h_layout)
        out = h_layout.Prx_track;
    end
    function out = get.track(h_layout) % Legacy option
        out = h_layout.Prx_track;
    end
    function out = get.pairing(h_layout)
        out = h_layout.Ppairing;
    end
    function out = get.no_links(h_layout)
        out = size( h_layout.Ppairing , 2 );
    end
    function out = get.dual_mobility(h_layout)
        out = any( cat(1,h_layout.tx_track(1,:).no_snapshots) > 1 ) | h_layout.Pdual_mobility;
    end
    
    % Set functions
    function set.name(h_layout,value)
        if ~( ischar(value) )
            error('QuaDRiGa:qd_layout:WrongInput','??? "name" must be a string.')
        end
        h_layout.name = value;
    end
    
    function set.simpar(h_layout,value)
        if ~( isa(value, 'qd_simulation_parameters') )
            error('QuaDRiGa:qd_layout:WrongInput','??? "simpar" must be object of the class "qd_simulation_parameters".')
        elseif ~all( size(value) == [1,1]  )
            error('QuaDRiGa:qd_layout:WrongInput','??? "simpar" must be scalar.')
        end
        h_layout.simpar = value;
    end
    
    function set.no_tx(h_layout,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                && isreal(value) && mod(value,1)==0 && value > 0 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "no_tx" must be integer and > 0')
        end
        if h_layout.no_tx > value
            h_layout.Ptx_array = h_layout.Ptx_array(:,1:value);
            h_layout.Ptx_track = h_layout.Ptx_track(1,1:value);
        elseif h_layout.no_tx < value
            Ntx_array = h_layout.Ptx_array;
            Ntx_track = h_layout.Ptx_track;
            for n = h_layout.no_tx+1 : value
                Ntx_array(:,n) = Ntx_array(:,n-1);
                trk = qd_track([]);
                trk.name = ['Tx',sprintf('%04.0f',n)];
                trk.initial_position = [0;0;25];
                Ntx_track(1,n) = trk;
            end
            h_layout.Ptx_array = Ntx_array;
            h_layout.Ptx_track = Ntx_track;
        end
        h_layout.set_pairing('all');
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.no_rx(h_layout,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                && isreal(value) && mod(value,1)==0 && value > 0 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "no_rx" must be integer and > 0')
        end
        if h_layout.no_rx > value
            h_layout.Prx_array = h_layout.Prx_array(:,1:value);
            h_layout.Prx_track = h_layout.Prx_track(1,1:value);
        elseif h_layout.no_rx < value
            for n = h_layout.no_rx+1 : value
                h_layout.Prx_array(:,n) = h_layout.Prx_array(:,n-1);
                trk = qd_track([]);
                trk.name = ['Rx',sprintf('%04.0f',n)];
                h_layout.Prx_track(1,n) = trk;
            end
        end
        h_layout.set_pairing('all');
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.tx_name(h_layout,value)
        if ~( iscell(value) )
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_name" must be a cell array.')
        elseif ~any( size(value) == 1 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_name" must be a vector on strings.')
        end
        if size(value,1)~=1
            value = value';
        end
        no_txx = h_layout.no_tx;
        if size( value , 2 ) ~= no_txx
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_name" must match the number of Tx.')
        end
        for n = 1:no_txx
            if ~ischar( value{n} )
                error('QuaDRiGa:qd_layout:WrongInput','??? Each "tx_name" must be a string.')
            end
        end
        if numel( unique( value ) ) < numel(value)
            error('QuaDRiGa:qd_layout:WrongInput','??? Each "tx_name" must be unique.')
        end
        for n = 1:size(value,2)
            trk = h_layout.Ptx_track(1,n); % Workaround for Octave 4.0
            trk.name = value{1,n};
            if no_txx == 1
                h_layout.Ptx_track = trk;
            else
                h_layout.Ptx_track(1,n) = trk;
            end
        end
    end
    
    function set.tx_position(h_layout,value)
        if ~( isnumeric(value) && isreal(value) )
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_position" must consist of real numbers')
        elseif ~all( size(value,1) == 3 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_position" must have 3 rows')
        end
        no_pos = size(value,2);
        if no_pos ~= h_layout.no_tx
            h_layout.no_tx = no_pos;
        end
        for n = 1:no_pos
            trk = h_layout.Ptx_track(1,n); % Workaround for Octave 4.0
            trk.initial_position = double( value(:,n) );
            if no_pos == 1
                h_layout.Ptx_track = trk;
            else
                h_layout.Ptx_track(1,n) = trk;
            end
        end
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.tx_array(h_layout,value)
        values = size(value,2);
        if ~( isa(value, 'qd_arrayant') )
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_array" must be objects of the class qd_arrayant')
        elseif ~( values == h_layout.no_tx || values == 1 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "tx_array" must match "no_tx". Try to set "no_tx" first.')
        end
        if values == 1 && h_layout.no_tx > 1
            value( 1,2:h_layout.no_tx ) = value(1,1);
        end
        h_layout.Ptx_array = value;
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.tx_track(h_layout,value)
        if ~( isa(value, 'qd_track') )
            error('QuaDRiGa:qd_layout:WrongInput','??? "track" must be objects of the class qd_track')
        end
        if numel(value) ~= h_layout.no_tx
            h_layout.no_tx = numel(value);
        end
        nm = {value.name};
        if numel( unique(nm) ) < numel(value)
            error('QuaDRiGa:qd_layout:WrongInput','??? Track name must be unique.')
        end
        h_layout.Ptx_track = value;
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.rx_name(h_layout,value)
        if ~( iscell(value) )
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_name" must be a cell array.')
        elseif ~any( size(value) == 1 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_name" must be a vector on strings.')
        end
        if size(value,1)~=1
            value = value';
        end
        no_rxx = h_layout.no_rx;
        if size( value , 2 ) ~= no_rxx
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_name" must match the number of Rx.')
        end
        for n = 1 : no_rxx
            if ~ischar( value{n} )
                error('QuaDRiGa:qd_layout:WrongInput','??? Each "rx_name" must be a string.')
            end
        end
        if numel( unique( value ) ) < numel(value)
            error('??? Each "rx_name" must be unique.')
        end
        for n = 1:size(value,2)
            trk = h_layout.Prx_track(1,n); % Workaround for Octave 4.0
            trk.name = value{1,n};
            if no_rxx == 1
                h_layout.Prx_track = trk;
            else
                h_layout.Prx_track(1,n) = trk;
            end
        end
    end
    
    function set.rx_position(h_layout,value)
        if ~( isnumeric(value) && isreal(value) )
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_position" must consist of real numbers')
        elseif ~all( size(value,1) == 3 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_position" must have 3 rows')
        end
        scen = h_layout.rx_track(1,1).scenario(:,1);
        no_pos = size(value,2);
        if no_pos ~= h_layout.no_rx
            h_layout.no_rx = no_pos;
        end
        for n = 1:no_pos
            trk = h_layout.Prx_track(1,n); % Workaround for Octave 4.0
            trk.initial_position = double( value(:,n) );
            if isempty(trk.scenario{1})
                trk.scenario = scen;
            end
            if no_pos == 1
                h_layout.Prx_track = trk;
            else
                h_layout.Prx_track(1,n) = trk;
            end
        end
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.rx_array(h_layout,value)
        values = size(value,2);
        if ~( isa(value, 'qd_arrayant') )
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_array" must be objects of the class qd_arrayant')
        elseif ~( values == h_layout.no_rx || values == 1 )
            error('QuaDRiGa:qd_layout:WrongInput','??? "rx_array" must match "no_rx". Try to set "no_rx" first.')
        end
        if values == 1 && h_layout.no_rx > 1
            value( 1,2:h_layout.no_rx ) = value(1,1);
        end
        h_layout.Prx_array = value;
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.rx_track(h_layout,value)
        if ~( isa(value, 'qd_track') )
            error('QuaDRiGa:qd_layout:WrongInput','??? "track" must be objects of the class track')
        end
        if numel(value) ~= h_layout.no_rx
            h_layout.no_rx = numel(value);
        end
        nm = {value.name};
        if numel( unique(nm) ) < numel(value)
            error('QuaDRiGa:qd_layout:WrongInput','??? Track name must be unique.')
        end
        h_layout.Prx_track = value;
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function set.track(h_layout,value)
        h_layout.rx_track = value;
    end
    
    function set.pairing(h_layout,value)
        value_list = reshape( value,1,[] );
        if ~( isnumeric(value) &&...
                isreal(value) &&...
                all( mod(value_list,1)==0 ) &&...
                size(value,1) == 2 &&...
                all( value_list > 0 ) )
            error('QuaDRiGa:qd_layout:WrongInput','??? "pairing" must be a positive integer matrix with two rows')
        elseif any( value(1,:)>h_layout.no_tx )
            error('QuaDRiGa:qd_layout:WrongInput','??? "pairing" refers to non-existing Tx')
        elseif any( value(2,:)>h_layout.no_rx )
            error('QuaDRiGa:qd_layout:WrongInput','??? "pairing" refers to non-existing Rx')
        end
        value_new = unique( value(1,:) + size(value,2) * value(2,:) );
        if numel( value_new ) < size(value,2)
            value = [ rem( value_new, size(value,2) ) ; fix(value_new./size(value,2)) ];
            warning('QuaDRiGa:qd_layout:WrongInput','removed multiple entries from "pairing".');
        end
        h_layout.Ppairing = value;
    end
    
    function set.dual_mobility(h_layout,value)
        h_layout.Pdual_mobility = logical(value(1));
        if ~isempty( h_layout.h_qd_builder_init )
            warning('QuaDRiGa:qd_layout:BuilderReset','Reset of pre-initialized "qd_builder" objects.');
            h_layout.h_qd_builder_init = [];
            h_layout.builder_index = [];
        end
    end
    
    function names_are_unique = has_unique_track_names( h_layout )
        % HAS_UNIQUE_TRACK_NAMES Checks if all names are unique
        names_are_unique = true;
        id = h_layout.rx_name;
        for n = 2 : numel( id )
            if names_are_unique
                names_are_unique = ~any( strcmp( id{n-1}, id(n:end) ) );
            end
        end
        id = h_layout.tx_name;
        for n = 2 : numel( id )
            if names_are_unique
                names_are_unique = ~any( strcmp( id{n-1}, id(n:end) ) );
            end
        end
        if nargout == 0 && ~names_are_unique
            error('QuaDRiGa:qd_layout:has_unique_track_names','All Rx tracks and all Tx tracks must have unique names.');
        end
    end
end
methods(Static)
    h_layout = generate( layout_type, no_sites, isd, h_array, no_sectors, sec_orientation )
    [ h_layout, ReferenceCoord ] = kml2layout( fn, split_seg, ForceRefCoord )
    varargout = call_private_fcn( functionName, varargin )
end

end
