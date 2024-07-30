function d_min = add_segment( h_track, pos, scenario, threshold )
%ADD_SEGMENT Adds segments to a qd_track object of array of objects
%
% Calling object:
%   Object array
%
% Description:
%   This method can be used to add segments to an existing qd_track object or array of qd_track
%   objects. The variable pos defines the position where the segment should start. Ideally, this
%   positions lies on a track. However, if it doesn't lie on a track, the closest point on the
%   nearest track is used. If the track has no point at this position, a new one will be created.
%   For example, a 100 m long linear track is defined by its start and end point (the qd_track
%   object contains two positions, 0 and 100 meters). If a new segment should start at 50 m
%   relative to the track start, the add_segment method will create a new point at 50 m and assign
%   the segment to it. The qd_track object will then contain 3 points (0 m, 50 m and 100 m).
%
% Input:
%   pos
%   A 3-element vector [x;y;z] in metric Cartesian coordinates defining the start-position of the
%   new segment (required).
%
%   scenario
%   A string or cell-array of strings providing the scenario name (optional). Scenario names are
%   defined by the configuration files in the config folder of the QuaDRiGa installation. A list of
%   supported scenarios can be obtained by calling "qd_builder.supported_scenarios". The scenario
%   cell-array can only have one column. Rows are for different transmitters. If scenario is not
%   defined, no new points are added to the track, but the distances are calculated.
%
%   threshold
%   A 2-element vector of real numbers. The first value is the minimum distance in meters to an
%   existing point on a track at which no new point is created, but the scenario is assigned to the
%   existing point (default = 0.1 m). The second value describes the maximum distance between "pos"
%   and the track line (default = 2 m).
%
% Output:
%   d_min
%   An array of floating point numbers describing the distance to each track in the array of
%   qd_track objects. The scenario is assigned to the object with the minimum distance.
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

if ~exist( 'threshold','var' ) || isempty( threshold )
    threshold = [ 0.1, 2 ];                         % Add new point only if distance to threshold is > 10 cm
elseif numel( threshold ) == 1
    threshold = threshold([1,1]);
end

if size( pos,1 ) ~= 3 || size( pos,2 ) ~= 1
    error('QuaDRiGa:qd_track:add_segment','"pos" must be a 3 row vector.');
end    
    

if numel(h_track) > 1                               % Array of qd_track objects
    
    sic     = size( h_track );
    d_min   = zeros( sic );
    for n = 1 : numel(h_track)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
        d_min(i1,i2,i3,i4) = add_segment( h_track(i1,i2,i3,i4), pos, [], threshold );  % Create points
    end
    
    if exist( 'scenario','var' ) && ~isempty( scenario )
        [ ~,n ] = min( d_min(:) );                  % Find track with minimum distance
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
        add_segment( h_track(i1,i2,i3,i4), pos, scenario, threshold([2,2]) );   % Set scenario
    end
    
else                                                % Single qd_track object
    h_track = h_track(1,1);                         % Fix for Octave
    
    P = pos - h_track.initial_position;             % Given point relative to track initial position
    PT = h_track.positions;                         % All positions on the track
    
    nP = size( PT,2 );                              % Number of snapshots
    oP = ones(1,nP);                                % Ones
    op = ones(1,nP-1);                              % Ones
    o3 = ones(1,3);
    
    d = sqrt(sum((PT - P(:,oP)).^2));               % Distances between given point and all points
    if d(end) < threshold(1)                        % The last point cannot start a segment
        d(end) = Inf;
    end
    [ d_min_nearest , ii ] = min( d );              % Minimum distance and nearest neighbor index
    
    use_projection = false;                         % Indicates that the projected point is not used
    if d_min_nearest <= threshold(1)                % Seach for projection
        use_nearest = true;                         % Indicates that the nearest neighbor is the best point
    else
        use_nearest = false;                        % Indicates that the projected point is better
        U = PT(:,2:end) - PT(:,1:end-1);            % Differential vectors from LineSegment-start to the LineSegment-end
        X = P(:,op) - PT(:,1:end-1);                % The vectors from P to LineSegment-start
        r = sum(X.*U) ./ sum(U.^2);                 % Orthogonal projection onto each LineSegment
        ip = r >= 0  & r < 1;                       % Proj. must be within a line segment
        
        if any( ip )
            UG = r(o3,ip) .* U(:,ip);               % The vectors from LineSegment-start to Projected point
            PG = PT(:,ip) + UG;                     % The vectors from origin to Projected point
            oG = ones(1,size(PG,2));                % Ones
            dG = sqrt(sum(( PG - P(:,oG) ).^2));    % Distances from P to Projected point
            [ d_min_proj , ii ] = min( dG );        % Minimum distance of projection
            
            if d_min_proj <= threshold(2)
                if d_min_proj < d_min_nearest       % Projection is better that nearest-neigbor
                    use_projection = true;
                    PG = PG(:,ii);                  % Select the winner
                    ip = find( ip );
                    ii = ip( ii );                  % Index of the point before the projected point
                else % d_min_proj >= d_min_nearest
                    use_nearest = true;             % Indicates that the nearest neighbor is the best point
                    [ ~ , ii ] = min( d );          % Minimum distance and nearest neighbor index
                end
            end
        end
    end
    
    if use_nearest
        PG = PT(:,ii);                              % Select the nearest neighbor
    end
    
    if exist( 'scenario','var' ) && ~isempty( scenario )
        
        if ischar( scenario )                       % Check the format of the scenario
            scenario = {scenario};
        elseif ~iscell( scenario )
            error('QuaDRiGa:qd_track:add_segment','Scenario must be a string or cell array.');
        elseif size( scenario,2 ) > 1
            error('QuaDRiGa:qd_track:add_segment','The scenario cell array can only have one column.');
        end
        
    else
        scenario = {''};
    end
    
    or = h_track.orientation;                   % Copy the orientations
    si = h_track.segment_index;                 % Copy the existing segment index
    sc = h_track.scenario;                      % Copy the scenario definition
    
    if size(sc,1) == 1 && size(scenario,1) > 1
        sc = sc( ones(1,size(scenario,1)),: );  % Duplicate exisitng scenarios
    elseif size(sc,1) > 1 &&  size(scenario,1) == 1
        scenario = scenario( ones(1,size(sc,1)),: );
    elseif size(sc,1) ==  size(scenario,1) == 1
        % Very good, nothing to be done !
    else
        error('QuaDRiGa:qd_track:add_segment','Inconsistent scenario definition.');
    end
    
    if use_nearest
        if ~any(si==ii)                         % Scenario does not exist
            h_track.segment_index = [ si( si<ii ), ii, si( si>ii ) ];
        end
        h_track.scenario = [ sc( :,si<ii ), scenario , sc( :,si>ii ) ];
        d_min = d_min_nearest;
    elseif use_projection
        h_track.positions = [ PT(:,1:ii), PG, PT(:,ii+1:end) ];
        h_track.orientation = [ or(:,[1:ii,ii]), or(:,ii+1:end) ];
        h_track.segment_index = [ si( si<=ii ), ii+1, si( si>ii )+1 ];
        h_track.scenario = [ sc( :,si<=ii ), scenario , sc( :,si>ii ) ];
        d_min = d_min_proj;
    else
        d_min = Inf;
    end
    
end

end

