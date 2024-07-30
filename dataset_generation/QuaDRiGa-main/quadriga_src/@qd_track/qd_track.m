classdef qd_track < handle
%QD_TRACK QuaDRiGa movement model class
%
% DESCRIPTION
% One feature of the channel model is the continuous evolution of wireless channels when the
% terminal moves through the environment. A track describes the movement of a mobile terminal. It is
% composed of an ordered list of positions. During the simulation, one snapshot is generated for
% each position on the track.
%
% Along the track, wireless reception conditions may change, e.g. when moving from an area with LOS
% to a shaded area. This behavior is described by segments. A segment is a subset of positions that
% have similar reception conditions. Each segment is classified by a segment index (i.e. the center
% position of the segment) and a scenario. 
%
% EXAMPLE
%
%    t = qd_track('circular',10*pi,0);
%    t.initial_position = [5;0;0];
%    t.segment_index = [1,40,90];
%    t.scenario = { 'C2l' , 'C2n' , 'C2l' };
%    t.interpolate_positions( 100 );
%    t.compute_directions;
%
% The first line creates a new trackas a circle with a circumference of 10*pi m or a diameter of 10
% m. The last argument defines the tracks start point (in degree). Zero means, that it start at the
% positive x-axis. The next statement (t.segment_index) defines three segments, one that starts at
% the beginning, one that starts at point 40 and one at point 90. Note, that each circular track has
% initially 129 points. Each segment is then given a scenario type. The last three lines interpolate
% the positions to 100 values per meter, adds the direction information and create a plot showing
% the track. 
%
% Note: The value of initial position will be added to the values of positions. However, the
% correlated large scale parameters will only be calculated for the initial position. It is
% therefore recommended to set the starting position of the track to [0;0;0] and use the
% initial_position property to place the terminal relative to the transmitter. 
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
    name = 'track';                 % Name of the track
end

properties(Dependent)
    no_snapshots                    % Number of positions on the track
    
    % Position offset (will be added to positions)
    %   This position is given in global Cartesian coordinates (x,y, and z-component) in units
    %   of [m]. The initial position normally refers to the starting point of the track. If the
    %   track has only one segment, it is also the position for which the LSPs are calculated.
    %   The initial position is added to the values in the positions variable.
    initial_position
    
    % Ordered list of position relative to the initial position
    %   QuaDRiGa calculates an instantaneous channel impulse response (also called snapshot) for
    %   each position on the track.
    positions
    
    % Orientation angles
    %   This 3-element vector describes the orientation of the radio device for each position on
    %   the track. The reference system for aircraft principal axes is used. The first value
    %   describes the "roll angle", i.e. the rotation around an axis drawn through the body of
    %   the vehicle from tail to nose in the normal direction of movement. Positive rotation is
    %   clockwise (seen from the pilot/drivers perspective). The second value describes the
    %   "pitch angle", i.e. the vertical (tilt) angle relative to the horizontal plane; positive
    %   rotation is up. The third value describes the bearing or "yaw angle", in mathematic
    %   sense. East corresponds to 0°, and the angles increase counter-clockwise, so north is
    %   90° degrees, south is -90°, and west is 180°. All values are given in [rad]. Note that
    %   by default, QuaDRiGa antennas face east (roll = 0, pitch = 0, yaw = 0).
    orientation
    
    % Time (in sec) vs. distance (in m) for speed profile
    %   QuaDRiGa supports variable terminal speeds. This is realized by interpolating the
    %   channel coefficients at the output of the model.  The variable track.movement_profile
    %   describes the  movement along the track by associating a time-point with a
    %   distance-point on the track.
    movement_profile
    
    no_segments                     % Number of segments or states along the track
    segment_index                   % Starting point of each segment given as index in the positions vector
    
    % Scenarios for each segment along the track
    %   This variable contains the scenario names for each segment as a cell array of strings. A
    %   list of supported scenarios can be obtained by calling "qd_cahnnel.supported_scenarios".
    %   If there is only one transmitter (i.e. one base  station), the cell array has the
    %   dimension [1 x no_segments]. For multiple transmitters, the rows of the array may
    %   contain different scenarios for each transmitter. For example, in a multicell setup with
    %   three terrestrial base stations, the propagation conditions may be different to all BSs.
    %   The cell arrays than has the dimension [3 x no_segments].
    scenario
end

properties
    par                             % Field for storing additional data
end

properties(Dependent,SetAccess=protected)
    closed                          % Indicates that the track is a closed curve
end

properties(Access=private)
    Pno_snapshots       = 1;
    Pno_segments        = 1;
    Pinitial_position   = [0;0;0];
    Ppositions          = [0;0;0];
    Pmovement_profile   = [];
    Porientation        = [0;0;0];
    Psegment_index      = 1;
    Pscenario           = {''};
end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
    ReferenceCoord = [];
end

methods
    % Constructor
    function h_track = qd_track( track_type , varargin )
        if exist('track_type','var') && ~isempty( track_type ) && nargin > 0
            h_track = qd_track.generate( track_type , varargin{:} );
        end
    end
    
    % Get functions
    function out = get.no_snapshots(h_track)
        out = h_track.Pno_snapshots;
    end
    function out = get.no_segments(h_track)
        out = h_track.Pno_segments;
    end
    function out = get.initial_position(h_track)
        out = h_track.Pinitial_position;
    end
    function out = get.positions(h_track)
        out = h_track.Ppositions;
    end
    function out = get.movement_profile(h_track)
        out = h_track.Pmovement_profile;
    end
    function out = get.segment_index(h_track)
        out = h_track.Psegment_index;
    end
    function out = get.scenario(h_track)
        out = h_track.Pscenario;
    end
    function out = get.orientation(h_track)
        out = h_track.Porientation;
    end
    function out = get.closed(h_track)
        if h_track.no_snapshots > 1 &&...
                get_length( h_track ) > 0 &&...
                all( h_track.positions(:,1) == h_track.positions(:,end) ) &&...
                ( isempty( h_track.orientation ) ||...
                all( h_track.orientation(:,1) == h_track.orientation(:,end) ) )
            out = true;
        else
            out = false;
        end
    end
    
    % Set functions
    function set.name(h_track,value)
        if ~( ischar(value) )
            error('QuaDRiGa:qd_track:wrongInputValue','??? Name must be a string.')
        elseif ~isempty( strfind(value,'_') ) && isempty( strfind(value,'_seg') ) %#ok
            error('QuaDRiGa:qd_track:wrongInputValue',...
                '??? Name cannot contain the underscore "_" character.')
        end
        h_track.name = value;
    end
    
    function set.no_snapshots(h_track,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                && isreal(value) && mod(value,1)==0 && value > 0 )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "no_snapshots" must be integer and > 0')
        end
        
        if h_track.Pno_snapshots > value
            h_track.Ppositions = h_track.Ppositions(:,1:value);
            if ~isempty( h_track.Porientation )
                h_track.Porientation = h_track.Porientation(:,1:value);
            end
            
        elseif h_track.Pno_snapshots < value
            no_pos_new = value-h_track.Pno_snapshots;
            o_pos_new  = ones(1,no_pos_new);
            no_snap = h_track.no_snapshots;
            
            pos_new = [ o_pos_new * h_track.Ppositions( 1,no_snap ); ...
                o_pos_new * h_track.Ppositions( 2,no_snap );...
                o_pos_new * h_track.Ppositions( 3,no_snap )];
            h_track.Ppositions = [ h_track.Ppositions , pos_new ];
            
            orientation_new = [ o_pos_new * h_track.Porientation( 1,no_snap ); ...
                o_pos_new * h_track.Porientation( 2,no_snap );...
                o_pos_new * h_track.Porientation( 3,no_snap )];
            h_track.Porientation = [ h_track.Porientation , orientation_new ];
            
            h_track.segment_index = h_track.Psegment_index( h_track.Psegment_index <= h_track.no_snapshots );
        end
        
        h_track.par = [];
        h_track.Pno_snapshots = double( value );
    end
    
    function set.no_segments(h_track,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) ...
                && isreal(value) && mod(value,1)==0 && value > 0 )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "no_segments" must be integer and > 0')
        end
        
        if h_track.Pno_segments > value
            h_track.Pscenario  = h_track.Pscenario(:,1:value);
            h_track.Psegment_index  = h_track.Psegment_index(:,1:value);
        elseif h_track.Pno_segments < value
            no_new_segments = value-h_track.no_segments;
            no_bs = size( h_track.Pscenario , 1 );
            no_snap = h_track.no_snapshots - h_track.Psegment_index(end);
            
            new_segments = cell( no_bs , no_new_segments );
            last_segment = h_track.Pscenario( :, end );
            for n = 1 : no_new_segments
                new_segments(:,n) = last_segment;
            end
            h_track.Pscenario = [ h_track.Pscenario , new_segments ];
            
            step    = floor( no_snap / (no_new_segments+1) );
            if value == h_track.Pno_snapshots
                Tsegment_index = 1 : h_track.Pno_snapshots;
            else
                Tsegment_index = [ h_track.Psegment_index, ...
                    h_track.Psegment_index(end) + (1:no_new_segments)*step  ];
                Tsegment_index( Tsegment_index > h_track.Pno_snapshots ) = h_track.Pno_snapshots;
            end
            h_track.Psegment_index = Tsegment_index;
        end
        
        h_track.par = [];
        h_track.Pno_segments = double( value );
    end
    
    function set.initial_position(h_track,value)
        if ~( isnumeric(value) && isreal(value) )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "initial_position" must consist of real numbers')
        elseif ~all( size(value) == [3,1] )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "initial_position" must have three rows.')
        end
        h_track.Pinitial_position = double( value );
    end
    
    function set.positions(h_track,value)
        if ~( isnumeric(value) && isreal(value) && size(value,2) > 0 )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "positions" must consist of real numbers')
        elseif ~all( size(value,1) == 3 )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "positions" must have three rows.')
        end
        if size( value,2 ) ~= h_track.Pno_snapshots
            h_track.no_snapshots = size( value,2 );
        end
        h_track.Ppositions = double( value );
    end
    
    function set.movement_profile(h_track,value)
        if isempty( value )
            h_track.Pmovement_profile = [];
        else
            if ~( isnumeric(value) &&...
                    isreal(value) &&...
                    size(value,1) == 2  &&...
                    all(all( value >= 0 )) )
                error('QuaDRiGa:qd_track:wrongInputValue','??? "movement_profile" must be a vector of real positive numbers');
            end
            if ~issorted( value(1,:) )
                error('QuaDRiGa:qd_track:wrongInputValue','??? time in "movement_profile" must be sorted');
            end
            h_track.Pmovement_profile = double( value );
        end
    end
    
    function set.orientation(h_track,value)
        if ~( isnumeric(value) && isreal(value) && size(value,1) == 3 )
            error('QuaDRiGa:qd_track:wrongInputValue',...
                '??? "orientation" must be numeric and real');
        end
        if ~isempty(value)
            if all( size(value) == [ 3 1 ] )
                value = value(:,ones(1,h_track.no_snapshots));
            elseif size(value,2) ~= h_track.no_snapshots
                error('QuaDRiGa:qd_track:wrongInputValue',...
                    '??? no. of elements in "orientation" does not match no. of snapshots');
            end
        end
        h_track.Porientation = double( value );
    end
    
    function set.segment_index(h_track,value)
        if ~( isnumeric(value) &&...
                isreal(value) &&...
                all( mod(value,1)==0 ) &&...
                any( size(value) == 1 ) &&...
                all( value > 0 ) )
            error('QuaDRiGa:qd_track:wrongInputValue','??? "segment_index" must be a vector of integer numbers having values > 0')
        elseif min(value) ~= 1
            error('QuaDRiGa:qd_track:wrongInputValue','??? first segment must start at index 1')
        elseif max(value) > h_track.no_snapshots
            error('QuaDRiGa:qd_track:wrongInputValue','??? maximum segment exceeds number of snapshots')
        end
        
        % Make sure that indices are unique and sorted
        if size(value,1) ~= 1
            value = value';
        end
        value = sort( unique( value ) );
        no_segments_old = h_track.no_segments;
        
        % Set the number of segments
        if numel(value) ~= h_track.no_segments
            h_track.no_segments = numel(value);
        end
        
        % Preserve scenarios
        if no_segments_old == 1 && numel(value) > 1
            tmp = cell( size( h_track.scenario,1 ) , numel(value) );
            for n = 1:numel(value)
                tmp(:,n) = h_track.scenario(:,1);
            end
            h_track.Pscenario = tmp;
        end
        
        h_track.Psegment_index = value;
    end
    
    function set.scenario(h_track,value)
        if ~iscell(value) && ischar(value)
            value = {value};
        end
        if ~iscell(value)
            error('QuaDRiGa:qd_track:wrongInputValue','??? "scenario" must be a cell array.')
        end
        if size(value, 2) == 1 && h_track.no_segments > 1
            for n = 2:h_track.no_segments
                value(:,n) = value(:,1);
            end
        elseif size(value, 2) ~= h_track.no_segments
            error('QuaDRiGa:qd_track:wrongInputValue','??? The number of columns in "scenario" must match the number of segments.')
        end
        for n = 1:h_track.no_segments
            for m = 1:size(value, 1)
                if ~ischar(value{m, n})
                    error('QuaDRiGa:qd_track:wrongInputValue',...
                        '??? Each "scenario" must be a string.')
                end
            end
        end
        h_track.Pscenario = value;
    end
    
    % Additional methods
    function pos = positions_abs( h_track )
        if numel(h_track) > 1
            sic = size( h_track );
            pos = cell( sic );
            for n = 1 : numel(h_track)
                [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
                pos{i1,i2,i3,i4} = positions_abs( h_track(i1,i2,i3,i4) );
            end
        else
            pos = [ h_track.positions(1,:) + h_track.initial_position(1) ; ...
                h_track.positions(2,:) + h_track.initial_position(2) ;...
                h_track.positions(3,:) + h_track.initial_position(3)];
        end
    end
end

methods(Static)
    function types = supported_types
        types =  {'linear','circular','street'};
    end
    h_track = generate( track_type , track_length , direction , varargin )
end

methods % Legacy interpolation methods are replaced by "qd_track.interpolate"
    function dist = interpolate_movement( h_track , si, method )   % Legacy "interpolate_movement"
        if ~exist('method','var')
            dist = interpolate( h_track, 'time', si );
        else
            dist = interpolate( h_track, 'time', si, [], method );
        end
    end
    function interpolate_positions( h_track, samples_per_meter )   % Legacy "interpolate_positions"
        if numel(h_track) > 1
            sic = size( h_track );
            for n=1:numel(h_track)
                [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
                interpolate_positions( h_track(i1,i2,i3,i4), samples_per_meter );
            end
        else
            interpolate( h_track(1,1), 'distance', 1/samples_per_meter, [], [], true );
        end
    end
end

end
