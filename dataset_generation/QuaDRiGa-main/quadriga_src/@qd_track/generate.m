function h_track = generate( track_type , track_length , direction , varargin )
%GENERATE Generate tracks
%
% Calling object:
%   None (static method)
%
% Description:
%   This function creates tracks with specific properties. Currently supported are "linear",
%   "circular" and "street".
%
% Track types:
%   linear
%   Creates a linear track with given length and direction.  Direction describes the travel
%   direction along the track in [rad] in mathematical sense (i.e. 0 means east, pi/2 means north,
%   pi means west and -pi/2 south). If "track_length" or "direction" is not specified, then the
%   default track is 1 m long and has a random direction.
%
%   circular
%   Creates a circular track with given length and starting-direction.  Direction defines the
%   starting point on the circle in [rad]. Positive values define the travel direction as counter
%   clock-wise and negative values as clock-wise. E.g. 0 sets the start point in the east of the
%   circle, traveling north; -2π sets it in the east, traveling south. The default is random.
%
%   street
%   Emulates a drive route through a city grid. The mobile terminal starts at point 0, going into a
%   specified direction. The trajectory grid is build from street segments. The length of each
%   street is specified by the parameters 'street_length_min', 'street_length_mu', and
%   'street_length_sigma'. At the end of a street (i.e. at a crossing), the terminal turns with a
%   probability specified by 'turn_probability'. The change of direction is in between 75 and 105
%   degrees either left or right. The radius of the curve is given by 'curve_radius'. The track is
%   set up in a way that prevents driving in circles.
%
% Input:
%   track_type
%   The type of the track
%
%   track_length
%   the length in [m]
%
%   direction
%   specifies the driving direction in [rad] of the first segment in mathematical sense (0 means
%   east, pi/2 means north). The default value is random
%
%   street_length_min
%   the minimal street length in [m]. The default is 50 m. (for type "street" only)
%
%   street_length_mu
%   the median street length in [m]. The default is 187 m. This value was obtained from
%   measurements in Berlin, Germany. (for type "street" only)
%
%   street_length_std
%   the standard deviation of the street length in [m]. The default is 83 m. This value was
%   obtained from measurements in Berlin, Germany. (for type "street" only)
%
%   curve_radius
%   the curve radius during a turn in [m]. The default is 10 m. (for type "street" only)
%
%   turn_probability
%   the probability of a turn at a crossing. Possible values are in between 0 and 1. The default is
%   0.5. (for type "street" only)
%
% Output:
%   h_track
%   A qd_track object
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

supported_types = qd_track.supported_types;
if ~exist( 'track_type' , 'var' )
    error('"track_type" is not given.')
elseif ~( ischar(track_type) && any( strcmpi(track_type,supported_types)) )
    str = 'Track type not found. Supported types are: ';
    no = numel(supported_types);
    for n = 1:no
        str = [str,supported_types{n}];
        if n<no
            str = [str,', '];
        end
    end
    error(str);
end

if ~exist( 'track_length' , 'var' ) || isempty( track_length )
    track_length = [];
elseif ~( isnumeric(track_length) &&...
        isreal(track_length) &&...
        all( size(track_length) == 1 ) &&...
        track_length >= 0 )
    error('??? "track_length" must be a real scalar > 0');
end

if ~exist( 'direction' , 'var' ) || isempty( direction )
    direction = rand*2*pi;
elseif ~( isnumeric(direction) &&...
        isreal(direction) &&...
        all( size(direction) == 1 ) )
    error('??? "direction" must be a real scalar');
end

% Generate track
switch track_type
    
    case 'linear'
        
        h_track = qd_track( [] );
        h_track.name = 'track';
        if isempty( track_length )
            track_length = 1;
        end
        if track_length == 0
            h_track.positions = [0;0;0];
            h_track.orientation = [0;0;angle( exp( 1j * direction ) )];
        else
            r = 0.5 * track_length * exp( 1j * [ direction-pi ; direction ] );
            r = r-r(1);
            h_track.positions = [ real(r(1)) , real(r(2)) ; imag(r(1)) , imag(r(2)) ; 0,0 ];
            h_track.orientation = [ 0,0 ; 0,0 ; angle( exp( 1j * [direction,direction] ) ) ];
        end
        
    case 'circular'
        
        if isempty( track_length )
            track_length = 2*pi*10;
        end
        r = track_length/(2*pi);
        if direction >= 0
            pt = (0:1/128:1);
        else
            pt = (1:-1/128:0);
        end
        pt = pt + 0.5 * direction / pi;
        circ = r * exp( 2*pi*1j* pt );
        circ = circ - circ(1);
        
        h_track = qd_track( [] );
        h_track.name = 'track';
        h_track.positions = [real(circ) ; imag(circ) ; zeros(1,numel(circ))];
        h_track.positions( : ,h_track.no_snapshots ) = h_track.positions( : , 1 );
        h_track.calc_orientation;
        
    case 'street'
        
        if isempty( track_length )
            track_length = 1000;
        end
        
        % Parse inputs
        var_names = {'street_length_min', 'street_length_mu', 'street_length_std',...
            'curve_radius', 'turn_probability'};
        var_defaults = [50,187,83,10,0.5];
        
        for n = 1:5
            if numel( varargin ) >= n && ~isempty( varargin{n} )
                if ~(isnumeric( varargin{n} ) && varargin{n}>=0 &&...
                        isreal( varargin{n} ) && all(size(varargin{n}) == [1 1]))
                    error(['??? "',var_names{n},'"  has wrong format']);
                end
                
                eval([ var_names{n},'=',num2str(num2str(varargin{n})),';'  ]);
            else
                eval([ var_names{n},'=',num2str(var_defaults(n)),';'  ]);
            end
        end
        
        diro  = 0;                      % Initial start direction
        point = 0;                      % The start point (always at [0,0])
        m = 1;                          % A counter for the points
        acc_length = 0;                 % Accumulated length
        
        while acc_length < 1.1*track_length
            
            % Get a random street length drawn from the distribution defined above
            street_length = randn*street_length_std + street_length_mu;
            while street_length < street_length_min
                street_length = randn*street_length_std + street_length_mu;
            end
            acc_length = acc_length + street_length;
            
            % Get 3 points along the street
            point(m+1) = point(m) + exp(1j*diro) * street_length*0.1;
            point(m+2) = point(m) + exp(1j*diro) * street_length*0.9;
            point(m+3) = point(m) + exp(1j*diro) * street_length;
            m=m+3;
            
            % At a crossing, the car could change its direction. This is
            % modeled here
            if rand < turn_probability
                
                dirb = pi;
                while dirb >= pi/2 + pi/12
                    dirn = diro + sign( rand-0.5 ) * pi/2 + randn*pi/12;
                    dirb = abs( angle( exp(1j*dirn) ));
                end
                
                point(m+1) = point(m) + curve_radius*( exp(1j*diro) + exp(1j*dirn) );
                diro = dirn;
                m=m+1;
            end
        end
        point = point .* exp(1j*direction);     % Set start direction
        
        h_track = qd_track( [] );
        h_track.name = 'track';
        h_track.positions = [ real(point) ; imag(point) ; zeros(1,numel(point))];
        h_track.interpolate_positions( 1 );     % Interpolate to 1 point per meter
        h_track.no_snapshots = ceil(track_length);
        h_track.calc_orientation;
end

end
