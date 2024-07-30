%% Drifting Phases and Delays
%
% Drifting is the method used for obtaining time evolution within one segment. This tutorial
% demonstrates the effect of “drifting” on the channel coefficients. It shows how drifting can be
% enabled and disabled as well as how the resulting data can be analyzed.   
%
% Drifting is an essential feature of the channel model. Drifting enables a continuous time
% evolution of the path delays, the path phases, the departure- and arrival angles and the LSPs. It
% is thus the enabling feature for time continuous channel simulations. Although drifting was
% already available in the SCME branch of the WINNER channel model, it did not make it into the main
% branch. Thus, drifting is not available in the WIM1, WIM2 or WIM+ model. It is also not a
% feature of the 3GPP model family. Here the functionality is implemented again. This script focuses
% on the delay and the phase component of the drifting  functionality. 

%% Channel model set-up and coefficient generation
% First, we parametrize the channel model. We start with the basic simulation parameters. For the
% desired output, we need two additional options: we want to evaluate absolute delays and we need to
% get all 20 sub-paths. Normally, the sub-paths are added already in the channel builder.

clear all
close all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.8])              % Default Paper Size

s = qd_simulation_parameters;                           % New simulation parameters
s.center_frequency = 2.53e9;                            % 2.53 GHz carrier frequency
s.sample_density = 4;                                   % 4 samples per half-wavelength
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.show_progress_bars = 0;                               % Disable progress bars

%%
% Second, we define a user track. Here we choose a linear track with a length of 30 m. The track
% start 20 m east of the transmitter and runs in east direction, thus linearly increasing the
% distance from the receiver.   

l = qd_layout( s );                                     % New QuaDRiGa layout
l.tx_position(3,1) = 25;                                % 25 m BE height
l.rx_track = qd_track('linear',30,0);                   % 30 m long track facing east
l.rx_track.initial_position = [20;0;0];                 % Start position
l.set_scenario('WINNER_UMa_C2_LOS');                    % Set propagation scenario
interpolate( l.rx_track, 'distance', 1/s.samples_per_meter, [],[],1  ); % Set sampling intervals
l.visualize;                                            % Plot the layout

%%
% Now, we generate the LSPs. We set the shadow fading and K-factor to 1 and disable the path loss
% model. 

cb = l.init_builder;                                    % Create new builder object
cb.scenpar.SF_sigma = 0;                                % 0 dB shadow fading
cb.scenpar.KF_mu = 0;                                   % 0 dB K-Factor
cb.scenpar.KF_sigma = 0;                                % No KF variation
cb.plpar = [];                                          % Disable path loss model
cb.gen_parameters;                                      % Generate large- and small-scale fading

%%
% Now, we generate the channel coefficients. The first run uses the drifting module, the second run
% disables it. Note that drifting needs significantly more computing resources. In some scenarios it
% might thus be useful to disable the feature to get quicker simulation results.   

cb.simpar.use_3GPP_baseline = 0;                        % Enable drifting (=spherical waves)
c = cb.get_channels;                                    % Generate channel coefficients
c.individual_delays = 0;                                % Remove per-antenna delays

cb.simpar.use_3GPP_baseline = 1;                        % Disable drifting
d = cb.get_channels;                                    % Generate channel coefficients


%% Results and discussion
% The following plots represent the results of the test. The first plot shows the delay of the LOS
% tap (blue) and the delay of the first NLOS tap (red) vs. distance. The solid lines are from the
% channel with drifting, the dashed lines are from the channel without. The LOS delay is always
% increasing since the Rx is moving away from the Tx. However, the increase is not linear due to the
% 25 m height of the Tx. Without drifting, the delays are not updated and stay constant during the
% segment. The position of the first scatterer is in close distance to the Rx (only some m away).
% When moving, the Rx first approaches the scatterer (delay gets a bit smaller) and then the
% distance increases again.

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure

distance = c.rx_position(1,:);                          % 2D distance between Tx and Rx
plot( distance, c.delay(1,:)*1e9 , '-b' )               % Plot LOS delay with drifting
hold on
plot( distance, d.delay(1,:)*1e9 , '-.b' )              % Plot LOS delay without drifting
plot( distance, c.delay(2,:)*1e9 , '-r' )               % Plot 1st NLOS path with drifting
plot( distance, d.delay(2,:)*1e9 , '-.r' )              % Plot 1st NLOS path without drifting
hold off
xlabel('Distance from track start point')
ylabel('Delay [ns] ')
title('Path delays')
legend('LOS with drifting','LOS without drifting','NLOS with drifting','NLOS without drifting')

%%
% This plot shows the power of the first NLOS tap along the track. The fading is significantly
% higher in the beginning and becomes much less strong towards the end.

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
pow = abs(squeeze(sum( c.coeff(1,1,2,:,:) , 5 ))).^2;   % Calculate power of first NLOS path
plot( distance,10*log10(pow),'-r' )                     % Plot power of first NLOS path
xlabel('Distance from track start point')
ylabel('Tap power (dB)')
title('NLOS power with drifting')

%%
% Without drifting, the phases of the subpaths are approximated by assuming that the angles to the
% LBSs do not change. However, this only holds when the distance to the LBS is large. Here, the
% initial distance is small (ca. 5 m). When the initial angles are kept fixed along the track, the 
% error is significant. Here, the phase ramp is negative, indicating a movement direction towards
% the scatterer and thus a higher Doppler frequency. However, when the scatterer is passed, the Rx
% moves away from the scatterer and the Doppler frequency becomes lower. This is not reflected when
% drifting is turned off. 
%
% Note here, that with shorter delay spreads (as e.g. in satellite channels), the scatterers are
% placed closer to the Rxs initial position. This will amplify this effect. Hence, for correct time
% evolution results, drifting needs to be turned on.

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
pow = abs(squeeze(sum( d.coeff(1,1,2,:,:) , 5 ))).^2;   % Calculate power of first NLOS path
plot( distance,10*log10(pow),'-r' )                     % Plot power of first NLOS path
xlabel('Distance from track start point')
ylabel('Tap power (dB)')
title('NLOS power without drifting')

