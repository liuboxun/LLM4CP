function o2i_loss_dB = gen_o2i_loss( h_layout, method, low_loss_fraction, SC_lambda, max_indoor_dist, gen_per_tx )
%GEN_O2I_LOSS Generates the outdoor-to-indoor penetration loss
%
% Calling object:
%   Single object
%
% Description:
%   This method generates the outdoor-to-indoor (O2I) penetration loss in [dB] for the 3GPP 38.901
%   channel model. The O2I-loss is specific for each terminal. In other words, if a MT is served by
%   several BSs, the same O2I-loss applies for each BSs. The values therefore need to be generated
%   before any other LSF or SSF parameters are generated. The automatic channel generation in
%   'qd_layout.get_channels' will call all functions in the correct order. For details see: 3GPP TR
%   38.901 v14.1.0, Sec. 7.4.3, Page 27.
%   The method generates the O2I-loss and indoor 3D distance as specified by 3GPP. The values are
%   then stored with the tracks in 'qd_layout.rx_track.par.o2i_loss' and
%   'qd_layout.rx_track.par.o2i_d3din'. These two variables are then read again by 'qd_builder.get_pl'
%   and applied to the path-loss in the specific scenario.
%
% Input:
%   method
%   String selecting the indoor pathloss model. Two models are implemented: the 3GPP model 
%   '3GPP_38.901' as described in 3GPP TR 38.901 v14.1.0, Sec. 7.4.3, Page 27 and the 'mmMAGIC'
%   model as described in H2020-ICT-671650-mmMAGIC/D2.2, Sec. 4.3.
%
%   low_loss_fraction
%   3GPP TR 38.901 specifies two different formulas for the O2I-loss, one for high-loss (e.g. IRR
%   glass and concrete) and one for low-loss (standard multi-pane glass). The variable
%   'low_loss_fraction' determines the likelihood of the low-loss model. Values must be between 0
%   (high-loss only) and 1 (low-loss only). Default-value is 0.5.
%
%   SC_lambda
%   Random variables for the 'low_loss_fraction' are spatially consistent. 'SC_lambda' describes
%   the decorrelation distance of the random generator. Default: 50 m.
%
%   max_indoor_dist
%   The maximum indoor distance between the building wall and the UE. Default is 25 m.
%
%   gen_per_tx
%   Logical variable to enable or disable (default) per-BS parameter generation. By default, the
%   same O2I-loss applies for each BS. However, this can be changed so that the O2I-loss is
%   generated independently for each BS by setting this variable to 'true'.
%
% Output:
%   o2i_loss_dB
%   A cell array containing the results for each MT. These values are identical to the ones stored
%   in 'qd_layout.rx_track.par.o2i_loss'. Each cell contains an array of values, where the dimensions
%   correspond to: [ BS, Segment, Frequency ]. For outdoor-scenarios, an empty array is returned.
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

if numel( h_layout ) > 1 
   error('QuaDRiGa:qd_layout:gen_o2i_loss','gen_o2i_loss not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

% Fraction of the MTs with low-loss
if ~exist( 'method', 'var' ) || isempty( method )
    error('QuaDRiGa:qd_layout:gen_o2i_loss','O2I loss method is not defined.');
end

% Fraction of the MTs with low-loss
if ~exist( 'low_loss_fraction', 'var' ) || isempty( low_loss_fraction )
    low_loss_fraction = 0.5;
end

% Spatial consistency variable
if ~exist( 'SC_lambda', 'var' ) || isempty( SC_lambda )
    SC_lambda = 50;
end

% Spatial consistency variable
if ~exist( 'max_indoor_dist', 'var' ) || isempty( max_indoor_dist )
    max_indoor_dist = 25;
end
min_outdoor_dist = 10;

% Spatial consistency variable
if ~exist( 'gen_per_tx', 'var' ) || isempty( gen_per_tx )
    gen_per_tx = false;
end

% Parse some often used variables
no_rx   = h_layout.no_rx;
no_tx   = h_layout.no_tx;
no_freq = numel( h_layout.simpar(1,1).center_frequency );
f_GHz   = h_layout.simpar(1,1).center_frequency / 1e9;
o_tx    = ones(no_tx,1);
o_frq   = ones(1,no_freq);

% Get the material-specific losses
% See. TR 38.901 v14.1.0, Page 28, Table 7.4.3-1: Material penetration losses
% Same values apply for mmMAGIC model
L_glass    =  2    + 0.2  * f_GHz;
L_IRglass  = 23    + 0.3  * f_GHz;
L_concrete =  5    + 4    * f_GHz;

% Get the loss through external wall in [dB]
% See. TR 38.901 v14.1.0, Page 28, Table 7.4.3-2: O2I building penetration loss model
% Same values apply for mmMAGIC model
PL_tw_low  = 5 - 10*log10( 0.3 * 10.^( -L_glass/10 )   + 0.7 * 10.^( -L_concrete/10 ) );
PL_tw_high = 5 - 10*log10( 0.7 * 10.^( -L_IRglass/10 ) + 0.3 * 10.^( -L_concrete/10 ) );

% Parse the positions
[ pos, pos_tx, rx_ind ] = parse_positions( h_layout );
no_seg = size( pos,2 );

% Select Low-loss or High-Loss model
use_low_loss  = false( 1, no_seg );
use_high_loss = false( 1, no_seg );

% Generate spatially correlated uniform random variable for the "low/high loss selection"
if SC_lambda == 0
    randC = rand( 1, no_seg );
else
    randC = qd_sos.rand( SC_lambda , pos );
end

use_low_loss( 1,: )  = randC( 1,: ) <= low_loss_fraction;
use_high_loss( 1,: ) = randC( 1,: ) > low_loss_fraction;

% Assign wall penetration loss
PL_tw = zeros( no_tx, no_seg, no_freq );
for f = 1 : no_freq
    for t = 1 : no_tx
        PL_tw( t,use_low_loss,f )  = PL_tw_low(1,f);       % Low loss
        PL_tw( t,use_high_loss,f ) = PL_tw_high(1,f);      % High loss
    end
end

% 3GPP generates "d_2d_in" by drawing two random variables and using the minimum. However,
% this is incompatible with spatial consistency. Therefore, we get only one, spatially
% consistent random variable (uniform distribution) and transform it into the desired
% distribution.

% Generate spatially correlated uniform random variable for the "d_2d_in"
if SC_lambda == 0
    if gen_per_tx   % Create independent variables for each TX
        randC = rand( no_tx, no_seg );
    else
        randC = rand( 1, no_seg );
    end
else
    if gen_per_tx   % Create independent variables for each TX
        randC = qd_sos.rand( SC_lambda*ones(1,no_tx) , pos );
    else
        randC = qd_sos.rand( SC_lambda , pos );
    end
end

% Transform into target distribution
val = 0:0.1:1;  % Samples of the cdf
cdf = [ 0, 0.193, 0.36, 0.514, 0.642, 0.749, 0.84, 0.911, 0.962, 0.991, 1 ];  % Probabilities

% Test of the mapping function:
if 0
    a = rand( 1,100000 )*25;
    b = rand( 1,100000 )*25;
    c = min( a,b );
    
    d = rand( 1,100000 );
    e = qf.interp( cdf, 0, val, d )*25;
    
    bins = 0:0.01:25;
    plot( bins', [ qf.acdf( c,bins ), qf.acdf( e,bins ) ] )
end

% Calculate the total 2D and 3D distances
d_3d  = zeros( no_tx, no_seg );    % 3D distance
d_2d  = zeros( no_tx, no_seg );    % 2D distance
theta = zeros( no_tx, no_seg );    % Elevation angle
for t = 1 : no_tx
    % Vector pointing from the BS to the MT
    pos_tmp = pos - pos_tx(:,:,t);
    d_3d(t,:)  = sqrt(sum( abs( pos_tmp ).^2,1 ));
    d_2d(t,:)  = sqrt(sum( abs( pos_tmp(1:2,:) ).^2,1 ));
    theta(t,:) = atand( pos_tmp(3,:) ./ d_2d(t,:) );
end

% Use interpolation to get the desired variables
d_2d_in = qf.interp( cdf, 0, val, randC );

if gen_per_tx
    d_2d_in = reshape( d_2d_in, size(randC) );
else % Duplicate for each transmitter
    d_2d_in = o_tx * d_2d_in;
end

% Scale with the maximum indoor distance
d_2d_in = d_2d_in * max_indoor_dist;

% Indoor distance cannot larger than outdoor distance
d_2d_in( d_2d_in > d_2d ) = d_2d( d_2d_in > d_2d );

% Calculate the indoor 3D distance
d_3d_in = d_3d .* (1 - (d_2d - d_2d_in) ./ d_2d);

% Make sure the indoor 3D distance does not exceed the total 3D distance
d_3d_in( isnan( d_3d_in ) ) = inf;
d_3d_in( d_3d_in > d_3d ) = d_3d( d_3d_in > d_3d );

% Maintain the minimum outdoor distance
ii = d_3d - d_3d_in < min_outdoor_dist;
d_3d_in( ii ) = d_3d( ii ) - min_outdoor_dist;

% Set minimum indoor-distance to 0
d_3d_in( d_3d_in < 0 ) = 0;

switch method
    case '3GPP_38.901'
        % Generate indoor loss in [dB]
        PL_in = 0.5 * d_2d_in;

    case 'mmMAGIC'
        % mmMAGIC indoor loss model. See H2020-ICT-671650-mmMAGIC/D2.2, Sec. 4.3.1
        
        % Generate spatially correlated uniform random variable for "alpha" (‘indoor loss coefficient’)
        if SC_lambda == 0
            randC = rand( 1, no_seg );
        else
            randC = qd_sos.rand( SC_lambda , pos );
        end
        alpha = 0.5 + randC;
        
        % Generate spatially correlated uniform random variable for "d_room" (‘room size’)
        if SC_lambda == 0
            randC = rand( 1, no_seg );
        else
            randC = qd_sos.rand( SC_lambda , pos );
        end
        d_room = 2 + 2*randC;
        
        % Generate indoor loss in [dB]
        PL_in = alpha(o_tx,:) .* max( d_2d_in - d_room(o_tx,:) , 0 );
        
    otherwise
        error('QuaDRiGa:qd_layout:gen_o2i_loss','O2I loss method is not defined.');
end

% Copy indoor loss for all frequencies
PL_in = PL_in( :,:,o_frq );

% Elevation dependency of the penetration loss (only for mmMAGIC model)
switch method
    case 'mmMAGIC'
        PL_el = 20*abs(theta/90);
        PL_el = PL_el(:,:,o_frq);
    otherwise
        PL_el = zeros( no_tx, no_seg, no_freq );
end

% Standard deviation of the penetration-loss
switch method
    case '3GPP_38.901'
        % Std. of the penetration loss
        % Only for 3GPP TR 38.901 v14.1.0
        sig_P_low  = 4.4 * o_frq;
        sig_P_high = 6.5 * o_frq;
    case 'mmMAGIC'
        % Frequency-dependent loss model
        sig_P_low = 4 + 0.1 * f_GHz;
        sig_P_high = sig_P_low;
end

% Generate spatially correlated uniform random variable for the "N(0,sigma_p)"
if SC_lambda == 0
    if gen_per_tx   % Create independent variables for each TX
        randC = rand( no_tx, no_seg );
    else
        randC = o_tx * rand( 1, no_seg );
    end
else
    if gen_per_tx   % Create independent variables for each TX
        randC = qd_sos.rand( SC_lambda*ones(1,no_tx) , pos );
    else
        randC = o_tx * qd_sos.rand( SC_lambda , pos );
    end
end

% Generate random penetration loss
PL_sf = zeros( no_tx, no_seg, no_freq );

% Apply for each tx
for f = 1 : no_freq
    for t = 1 : no_tx
        PL_sf( t,  use_low_loss, f ) = randC( t , use_low_loss )  *  sig_P_low(1,f);
        PL_sf( t, use_high_loss, f ) = randC( t , use_high_loss ) * sig_P_high(1,f);
    end
end

% Combine the three components
PL = PL_tw + PL_el + PL_in + PL_sf;

% Write to track objects
o2i_loss_dB = cell( 1, no_rx );
for r = 1 : no_rx
    pl_tmp = PL( :, rx_ind(1,:) == r, : );
    o2i_loss_dB{ r } = pl_tmp;
    if any( abs( pl_tmp(:) ) ~= 0 )
        par_tmp = h_layout.rx_track(1,r).par;
        par_tmp.o2i_loss  = PL( :, rx_ind(1,:) == r, : );
        par_tmp.o2i_d3din = d_3d_in(:,rx_ind(1,:) == r);
        h_layout.rx_track(1,r).par = par_tmp;
    end
end

end
