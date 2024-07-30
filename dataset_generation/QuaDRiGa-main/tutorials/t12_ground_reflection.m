%% Ground reflection simulation
%
% This tutorial shows how to include a deterministic ground reflection component into the channel.
% The effects are then demonstrated for different carrier frequencies (2 GHz, 28 GHz, and 60 GHz).
%
% Simulation assumptions are in accordance with 3GPP 38.901 v14.1.0, Section 7.6.8, p.60 (Explicit
% ground reflection model). Some modifications are made as described in [Jaeckel, S.; Raschkowski,
% L.; Wu, S.; Thiele, L. & Keusgen, W.; "An Explicit Ground Reflection Model for mm-Wave Channels",
% Proc. IEEE WCNC Workshops '17, 2017 ]. For all ground reflection simulations, a random ground
% humidity is assumed, which changes the relative permittivity of the ground and, hence, the
% reflection coefficient will be different for each segment. All ground reflection properties are
% controlled by the scenario configuration files in the "config" folder of the channel model. The
% parameter "GR_enabled" activates (1) or deactivates (0) the ground reflection component. The
% parameter "GR_epsilon" can be used to fix the relative permittivity to a fixed value.


%% Basic setup
% Multiple frequencies are set in the simulation parameters by providing a vector of frequency
% sample points. A new layout is created with a 10 m high BS position. Three different model
% parametrizations are compared:
%
% * 2-ray ground reflection model without any additional NLOS components
% * 3GPP 38.901 Urban Microcell LOS
% * Modified 3GPP 38.901 Urban Microcell LOS including a ground reflection
%
% The MT is at 1.5 m height and moves along a 50 m long track starting 10 m away from the BS. The
% model is set to sample the channel every 10 cm (10 time per meter).
%
% Since the 3GPP scenarios also have non-deterministic NLOS components, there needs to be a birth /
% death process of the scattering clusters along the MT trajectory. This is done by splitting the
% track into segments. "split_segment" assumes an average segment length of 30 m with a standard
% deviation of 5 m.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change paper Size

s = qd_simulation_parameters;
s.center_frequency = [2e9 28e9 60e9];                   % Set the three carrier frequencies

l = qd_layout( s );                                     % New QuaDRiGa layout
l.no_tx = 3;                                            % One BS for each scenario
l.tx_position(3,:) = 10;                                % Set BS height for all scenarios

l.rx_track = qd_track( 'linear' , 50, 0 );              % 50 m long track
l.rx_track.initial_position = [10 ; 0 ; 1.5 ];          % Set start positions and MT height
l.rx_track.interpolate_positions(10);                   % Set sampling rate to 10 saples per meter

% Each of the 3 BS gets assigned a different scenario:
l.rx_track.scenario = { 'TwoRayGR' ; '3GPP_38.901_UMi_LOS' ; '3GPP_38.901_UMi_LOS_GR' };

l.rx_track.split_segment;                               % Split into segments
c = l.get_channels;                                     % Generate the channel coefficients
dist_2d = c(1,1,1).rx_position(1,:);                    % Extract the 2D distance

%% Plot path gain for 2-ray model
%
% The first plot shows the results for the 2-ray ground reflection model. One can see the
% differences in path gain between the 3 frequency bands. The main difference, however, are the
% rapid power fluctuations due to the interference between the 2 paths. This is very different at
% mmWave frequencies compared to 2 GHz.

H = c(1,1,1).fr(100e6,64);                              % 2 GHz broadband channel (100 MHz)
P_2ray_2Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));      % Average power

H = c(1,1,2).fr(100e6,64);                              % 28 GHz broadband channel (100 MHz)
P_2ray_28Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));     % Average power

H = c(1,1,3).fr(100e6,64);                              % 60 GHz broadband channel (100 MHz)
P_2ray_60Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));     % Average power

P = 10*log10( [ P_2ray_2Ghz , P_2ray_28Ghz , P_2ray_60Ghz ] );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist_2d , P )
axis([0, max(dist_2d)+10, min(P(:))-5, max(P(:))+5  ])
xlabel('BS-MT 2D distance in [m]')
ylabel('Path gain in [dB]')
title('Path gain for 2-ray model')
legend('2 GHz','28 GHz','60 GHz')

%% Plot path gain for 3GPP UMi LOS model
%
% The second plot sows the results for the 3GPP UMi LOS model. The path loss is similar compared to
% the 2-ray model. A shadow-fading component induces slow changes in the average received power. By
% default, the shadow fading is fully correlated between the 3 frequencies. Small-scale-fading
% correlations are done according to 3GPP TR 38.901 V14.1.0, Section 7.6.5, pp 57. This can be
% changed by not using "l.get_channels", but executing the channgel generation steps maually in a
% different order (see the 3GPP TR 38.901 full calibration for more deails). The NLOS components
% cause some fast fading wihich is averaged out by the broadband processing. No ground reflection is
% included. Hence, the fast fluctuations are absent.

H = c(1,2,1).fr(100e6,64);                              % 2 GHz broadband channel (100 MHz)
P_2ray_2Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));      % Average power

H = c(1,2,2).fr(100e6,64);                              % 28 GHz broadband channel (100 MHz)
P_2ray_28Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));     % Average power

H = c(1,2,3).fr(100e6,64);                              % 60 GHz broadband channel (100 MHz)
P_2ray_60Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));     % Average power

P = 10*log10( [ P_2ray_2Ghz , P_2ray_28Ghz , P_2ray_60Ghz ] );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist_2d , P )
axis([0, max(dist_2d)+10, min(P(:))-5, max(P(:))+5  ])
xlabel('BS-MT 2D distance in [m]')
ylabel('Path gain in [dB]')
title('Path gain for 3GPP UMi LOS')
legend('2 GHz','28 GHz','60 GHz')


%% Plot path gain for 3GPP UMi LOS model
%
% The last plot shows the modified 3GPP channel (see [1]), where the ground reflection is
% included. Hence, the typical fluctuations are now included.

H = c(1,3,1).fr(100e6,64);
P_2ray_2Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));

H = c(1,3,2).fr(100e6,64);
P_2ray_28Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));

H = c(1,3,3).fr(100e6,64);
P_2ray_60Ghz = squeeze(mean(abs(H(1,1,:,:)).^2,3));

P = 10*log10( [ P_2ray_2Ghz , P_2ray_28Ghz , P_2ray_60Ghz ] );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist_2d , P )
axis([0, max(dist_2d)+10, min(P(:))-5, max(P(:))+5  ])
xlabel('BS-MT 2D distance in [m]')
ylabel('Path gain in [dB]')
title('Path gain for 3GPP UMi LOS incl. Ground Reflection')
legend('2 GHz','28 GHz','60 GHz')

