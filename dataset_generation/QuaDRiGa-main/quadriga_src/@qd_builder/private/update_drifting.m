function [ phi_d_lms, theta_d_lms, phi_a_lms, theta_a_lms, psi_lms, tau_ls, ...
    phi_d_1ms, theta_d_1ms, phi_a_1ms, theta_a_1ms, theta_r, update_tx_ant ] =...
    update_drifting( h_builder, i_snapshot, i_mobile, fbs_pos, lbs_pos )
%UPDATE_DRIFTING Updates the drifting angles for the given snapshot
%
% This function provides new drifting angles, phases and delays for each
% snapshot specified by "snapshot". It is mandatory to call
% "generate_fbs_lbs" first in order to initialize scatterer positions.
%
% Input variables:
%   i_snapshot      [ 1 , 1 ]
%       Snapshot number for which the drifting variables should be returned
%
%   i_mobile        [ 1 , 1 ]
%       Number of the mobile terminal. 
%
%   fbs_pos         [ 3, n_paths ]
%       Positions of the First-bounce scatterers in Global Cartesian Coordinates
%
%   lbs_pos         [ 3, n_paths ]
%       Positions of the Last-bounce scatterers in Global Cartesian Coordinates
%
% Output variables:
%   phi_d_lms       [ 1 , n_paths , n_tx ]
%       NLOS azimuth arrival angles
%
%   theta_d_lms     [ 1 , n_paths , n_tx ]
%       NLOS elevation arrival angles
%
%   phi_a_lms       [ 1 , n_paths , n_rx ]
%       NLOS azimuth arrival angles
%
%   theta_a_lms     [ 1 , n_paths , n_rx ]
%       NLOS elevation arrival angles
%
%   psi_lms         [ 1 , n_paths , n_rx , n_tx ]
%       Phases
%
%   tau_ls          [ 1 , n_clusters , n_rx , n_tx ]
%       Delays
%
%   phi_d_1ms       [ 1 , 1/2 , n_rx , n_tx ]
%       LOS azimuth departure angles
%
%   theta_d_1ms     [ 1 , 1/2 , n_rx , n_tx ]
%       LOS elevation departure angles
%
%   phi_a_1ms       [ 1 , 1/2 , n_rx , n_tx ]
%       LOS azimuth arrival angles
%
%   theta_a_1ms     [ 1 , 1/2 , n_rx , n_tx ]
%       LOS elevation arrival angles
%
%   theta_r         [ n_rx , n_tx ]
%       Angle between the ground and the reflected path (ground reflection only)
%
%   update_tx_ant   [ 1 , 1 ]
%       A logical scalar indicating if the Tx antenna patterns should be updated.
%
%
% The four variables (phi_d_1ms, theta_d_1ms, phi_a_1ms, theta_a_1ms) have 2 elements if
% ground reflection is used and 1 element if only LOS is used.
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

persistent i_mobile_p fbs_pos_p lbs_pos_p norm_b_tlms phi_d_lms_p theta_d_lms_p norm_c_lm e_ts e_ts0 NumSubPaths useGR

% If "i_mobile" is given as an input, we initialize all persistent variables. If it is not given, we
% use the values from the last call to the method. This reduces computing time significantly.
if exist( 'i_mobile','var' )
    i_mobile_p = i_mobile;
    
    useGR = h_builder.check_los > 1.5;
    
    % Initialize "fbs_pos_p" and "lbs_pos_p"
    if ~exist( 'fbs_pos','var' ) || isempty( fbs_pos ) || ~exist( 'lbs_pos','var' ) || isempty( lbs_pos )
        error('QuaDRiGa:qd_builder:update_drifting','Scatterer positions are not given.');
    end
    
    fbs_pos_p  = fbs_pos;
    lbs_pos_p  = lbs_pos;
    
    % The vector pointing from the FBS to the LBS
    c_lm       = -fbs_pos +lbs_pos;
    norm_c_lm  = sqrt( sum( c_lm.^2 ,1 ) );
    
    iClst = h_builder.pow(i_mobile,:) ~= 0;
    iClst(1) = true;
    if useGR
        iClst(2) = true;
    end
    NumSubPaths = h_builder.NumSubPaths( iClst );
end

% Read some common variables
n_paths = size( fbs_pos_p,2 );
o_path  = ones( 1,n_paths );
lambda  = h_builder.simpar(1,1).wavelength;                  % Wavelength in [m]

% We only update the variables for the tx-antenna if the Tx is moving.
if h_builder.tx_track(1,i_mobile_p).no_snapshots == 1 && i_snapshot > 1
    update_tx_ant = false;
else
    update_tx_ant = true;
end

% The vector pointing from the origin to the initial tx-position
tx_pos = h_builder.tx_position(:,i_mobile_p);

% The vector pointing from the origin to the initial rx-position
rx_pos = h_builder.rx_positions(:,i_mobile_p);

% Update the tx_variables
if update_tx_ant
    % The vector pointing from the tx-array center to the individual tx elements
    e_tx = h_builder.tx_array(1,i_mobile_p).element_position;
    n_tx = size( e_tx, 2 );
    o_tx = ones(1,n_tx);
    
    % Apply the tx orientation from the tx_track to the tx element positions
    O = h_builder.tx_track(1,i_mobile_p).orientation;
    R = qf.calc_ant_rotation( O(3,i_snapshot), -O(2,i_snapshot), O(1,i_snapshot) );
    e_tx = R * e_tx;
    
    % The vector from the initial tx position to the tx position at snapshot s
    e_ts0 = h_builder.tx_track(1,i_mobile_p).positions(:,i_snapshot);
    
    % The vector from the initial tx position to the tx antenna element positions at snapshot s
    e_ts = e_tx + e_ts0(:,o_tx);
    e_ts = reshape( e_ts , 3 , 1 , 1 , n_tx );
    
    % Calculate the vector from the current Tx position to the FBS position
    b_tlms = -e_ts( :,o_path,1,: ) -tx_pos(:,o_path,1,o_tx) +fbs_pos_p(:,:,1,o_tx);
    norm_b_tlms = sqrt( sum( b_tlms.^2 ,1 ) );
    
    % Update the departure NLOS angles
    phi_d_lms_p   = atan2(b_tlms(2,:,:,:), b_tlms(1,:,:,:));
    theta_d_lms_p = asin(b_tlms(3,:,:,:)./norm_b_tlms);
else
    n_tx = size( norm_b_tlms,4 );
    o_tx = ones( 1,n_tx );
end

% Copy persistent departure NLOS angles to the method output 
phi_d_lms   = phi_d_lms_p;
theta_d_lms = theta_d_lms_p;

% Update the rx_variables
% The vector pointing from the tx-array center to the individual tx elements
e_rx = h_builder.rx_array(1,i_mobile_p).element_position;
n_rx = size( e_rx, 2 );
o_rx = ones(1,n_rx);

% Apply the rx orientation rotation from the rx_track to the rx element positions
O = h_builder.rx_track(1,i_mobile_p).orientation;
R = qf.calc_ant_rotation( O(3,i_snapshot), -O(2,i_snapshot), O(1,i_snapshot) );
e_rx = R * e_rx;

% The vector from the initial position to the position at snapshot s
e_rs0 = h_builder.rx_track(1,i_mobile_p).positions(:,i_snapshot) ;

% Add the positions of the rotated elements to the Rx position on the rx_track at snapshot s.
e_rs = e_rx + e_rs0(:,o_rx);
e_rs = reshape( e_rs , 3 , 1 , n_rx );

% Calculate the vector from the current Rx position to the LBS position
a_rlms = -e_rs( :,o_path,: ) -rx_pos(:,o_path,o_rx) +lbs_pos_p(:,:,o_rx);
norm_a_rlms = sqrt( sum( a_rlms.^2 ,1 ) );

% Update the arrival NLOS angles
phi_a_lms   = atan2(a_rlms(2,:,:,:), a_rlms(1,:,:,:));
theta_a_lms = asin(a_rlms(3,:,:,:)./norm_a_rlms);

% The total path lengths for the NLOS components
d_lms  = norm_b_tlms(1,:,o_rx,:) + norm_c_lm(1,:,o_rx,o_tx) + norm_a_rlms(1,:,:,o_tx);

% LOS Drifting:
% The vector from the Tx to each Rx position.
r_rts = -e_ts(:,1,o_rx,:) -tx_pos(:,1,o_rx,o_tx) +rx_pos(:,1,o_rx,o_tx) +e_rs(:,1,:,o_tx);
norm_r_rts = sqrt( sum( r_rts.^2 ,1 ) );

% Update the LOS angles
phi_d_1ms   = atan2(r_rts(2,:,:,:), r_rts(1,:,:,:));
theta_d_1ms = asin(r_rts(3,:,:,:)./norm_r_rts);
phi_a_1ms   = atan2(-r_rts(2,:,:,:), -r_rts(1,:,:,:));
theta_a_1ms = asin(-r_rts(3,:,:,:)./norm_r_rts);

% For the LOS Tx angles, we use the first Rx array element
phi_d_lms(1,1,1,:)   = phi_d_1ms(1,1,1,:);
theta_d_lms(1,1,1,:) = theta_d_1ms(1,1,1,:);

% For the LOS Rx angles, we use the first Tx array element
phi_a_lms(1,1,:)   = phi_a_1ms(1,1,:,1);
theta_a_lms(1,1,:) = theta_a_1ms(1,1,:,1);

% The total path lengths for the LOS components
d_lms(1,1,:,:) = norm_r_rts(1,1,:,:);

% Additional calculations for the ground reflection
if useGR
    % The vector pointing from the origin to the current Rx element position
    o_gr = rx_pos(:,1,o_rx) + e_rs;
    
    % The vector pointing from the origin to the mirrored Rx element position
    o_gr(3,:) = -o_gr(3,:);
    
    % The vector pointing from the Tx element position to the mirrored Rx element position
    r_gr = -e_ts(:,1,o_rx,:) -tx_pos(:,1,o_rx,o_tx) +o_gr(:,1,:,o_tx);
    norm_r_gr = sqrt( sum( r_gr.^2 ,1 ) );
    
    % Update the ground reflection angles
    phi_d_1ms(1,2,:,:)   = atan2(r_gr(2,:,:,:), r_gr(1,:,:,:));
    theta_d_1ms(1,2,:,:) = asin(r_gr(3,:,:,:)./norm_r_gr);
    phi_a_1ms(1,2,:,:)   = atan2(-r_gr(2,:,:,:), -r_gr(1,:,:,:));
    theta_a_1ms(1,2,:,:) = -asin(-r_gr(3,:,:,:)./norm_r_gr);
    
    % For the GR Tx angles, we use the first Rx array element
    phi_d_lms(1,2,1,:)   = phi_d_1ms(1,2,1,:);
    theta_d_lms(1,2,1,:) = theta_d_1ms(1,2,1,:);
    
    % For the GR Rx angles, we use the first Tx array element
    phi_a_lms(1,2,:)   = phi_a_1ms(1,2,:,1);
    theta_a_lms(1,2,:) = theta_a_1ms(1,2,:,1);
    
    % The total path lengths for the GR components
    d_lms(1,2,:,:) = norm_r_gr(1,1,:,:);
    
    % Calculate the angle between the ground and the reflected path
    d_2d = sqrt( sum( r_gr([1,2],1,:,:).^2 ,1 ) );
    theta_r = atan( -r_gr(3,1,:,:) ./ d_2d );
    theta_r = permute( theta_r, [3,4,1,2] );
else
    theta_r = [];
end

% The phases for each sub-path
psi_lms     = 2*pi/lambda * mod(d_lms, lambda);

% The average delay for each cluster
tau_ls      = clst_avg( d_lms, NumSubPaths ) ./ h_builder.simpar(1,1).speed_of_light;

% When we use relative delays, we have to normalize the delays to the LOS tau_ls0 is the LOS delay
% at the RX-position without antennas. It is needed when the coefficients are going to be normalized
% to LOS delay. 
if ~h_builder.simpar(1,1).use_absolute_delays
    r_rts0 = -e_ts0 -tx_pos +rx_pos +e_rs0;
    norm_r_rts0 = sqrt( sum( (r_rts0).^2,1 ) );
    tau_ls0 = norm_r_rts0 ./ h_builder.simpar(1,1).speed_of_light;
    tau_ls = tau_ls - tau_ls0;
end

end
