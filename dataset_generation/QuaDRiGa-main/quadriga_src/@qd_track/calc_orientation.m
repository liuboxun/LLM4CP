function calc_orientation( h_track, roll, pitch, yaw )
%CALC_ORIENTATION Calculates the orientation of the mobile device
%
% Calling object:
%   Object array
%
% Description:
%   This function can be used to change the orientation of the mobile device. If no input variable
%   is provided, this function calculates the orientation based on the positions and stores the
%   output in the orientation property of the qd_track object. For this calculation, it is assumed
%   that the receive array antenna is fixed on a car and the car moves along the track, i.e., the
%   antenna turns with the car when the car is changing direction. This needs to be accounted for
%   when generating the channel coefficients. 
%
%   If there is at least one input variable given, the device orientation is changed by the given
%   values. For example: you can can calculate the orientation by calling the method without
%   arguments. This would fix the antenna orientation such that the antennna broadside always points
%   into the direction of movement. A second call to the function with a given pitch angle of e.g.,
%   90 degree would tilt the antenna up. All values are given in [rad]. Note that by default,
%   QuaDRiGa antennas face east (roll = 0, pitch = 0, yaw = 0).
%
% Input:
%   roll
%   The first value describes the "roll angle" in [rad], i.e. the rotation around an axis drawn
%   through the body of the vehicle from tail to nose in the normal direction of movement. Positive
%   rotation is clockwise (seen from the pilot/drivers perspective). The default value is 0.
%
%   pitch
%   The second value describes the "pitch angle" in [rad], i.e. the vertical (tilt) angle relative
%   to the horizontal plane; positive rotation is up. If this value is not given, it is calculated
%   from the snapshot positions.
%
%   yaw
%   The third value describes the bearing or "yaw angle", in mathematic sense. Values must be given
%   in [rad]. East corresponds to 0, and the angles increase counter-clockwise, so north is 90
%   degree degrees, south is -90 degree, and west is 180 degree. If this value is not given, it is
%   calculated from the snapshot positions.
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

if ~exist('roll','var') || isempty( roll )
   roll = []; 
end

if ~exist('pitch','var') || isempty( pitch )
   pitch = []; 
end

if ~exist('yaw','var') || isempty( yaw )
   yaw = []; 
end

if numel(h_track) > 1
    
    sic = size( h_track );
    prc = false( sic );
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            calc_orientation( h_track(i1,i2,i3,i4), roll, pitch, yaw );
            prc( qf.eqo( h_track(i1,i2,i3,i4), h_track ) ) = true;
        end
    end
    
else
    h_track = h_track(1,1);     % Fix for Octave

    if isempty(roll) && isempty(pitch) && isempty(yaw)
        % Calculate the orientation from the positions
        
        if h_track.no_snapshots < 2
            h_track.orientation = [0;0;0];
        else
            P = h_track.Ppositions(:,2:end) - h_track.Ppositions(:,1:end-1);
            [ az,el ] = cart2sph(P(1, :), P(2, :), P(3, :));
            
            n = h_track.no_snapshots;
            if h_track.closed
                az(n) = az(1);
                el(n) = el(1);
            else
                az(n) = az(n-1);
                el(n) = el(n-1);
            end
            
            h_track.orientation = [ zeros(1,n) ; el ; az ];
        end
    
    else % Calculate the orientation from the given values
        
        no_snapshots = h_track.no_snapshots;
        orientation  = h_track.orientation;
        if isempty( orientation )
            orientation = zeros( 3, no_snapshots );
        end
        
        if isempty( roll )
            roll = zeros( 1,no_snapshots );
        elseif numel( roll ) == 1
            roll = ones( 1,no_snapshots )*roll;
        elseif numel( roll ) ~= no_snapshots
            error('QuaDRiGa:qd_track:calc_orientation',...
                '"roll" must be either scalar or match the numer of snapshots.');
        end
        
        if isempty( pitch )
            pitch = zeros( 1,no_snapshots );
        elseif numel( pitch ) == 1
            pitch = ones( 1,no_snapshots )*pitch;
        elseif numel( pitch ) ~= no_snapshots
            error('QuaDRiGa:qd_track:calc_orientation',...
                '"pitch" must be either scalar or match the numer of snapshots.');
        end
        
        if isempty( yaw )
            yaw = zeros( 1,no_snapshots );
        elseif numel( yaw ) == 1
            yaw = ones( 1,no_snapshots )*yaw;
        elseif numel( yaw ) ~= no_snapshots
            error('QuaDRiGa:qd_track:calc_orientation',...
                '"yaw" must be either scalar or match the numer of snapshots.');
        end
        
        % Apply updates
        Rn = qf.calc_ant_rotation( yaw, -pitch );
        Ro = qf.calc_ant_rotation( orientation(3,:), -orientation(2,:) );
        for is = 1 : no_snapshots
            C = Ro(:,:,is)*Rn(:,:,is)*[1;0;0];
            C(3,C(3)> 1) = 1;         % Possible numeric instability
            C(3,C(3)<-1) = -1;        % Possible numeric instability
            orientation(2,is) = asin( C(3) );
            orientation(3,is) = atan2( C(2),C(1) );
        end
        orientation(1,:) = angle(exp(1j*(orientation(1,:)+roll)));
        h_track.orientation = orientation;
    end
end

end
