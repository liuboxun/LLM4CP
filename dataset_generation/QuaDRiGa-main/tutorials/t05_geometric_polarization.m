%% Geometric Polarization
%
% This tutorial shows how to study polarization effects with QuaDRiGa. Different linearly polarized
% antennas are defined at the transmitter and the receiver, the channel between them is calculated
% and the polarization effects are evaluated.
%
% We demonstrate the polarization rotation model that calculates the path power for polarized array
% antennas. We do this by setting up the simulation with different H/V polarized antennas at the
% transmitter and at the receiver. Then we define a circular track around the receiver. When the
% receiver moves around the transmitter, it changes its antenna orientation according to the
% movement direction. In this way, all possible departure and elevation angles are sampled.
% Depending on the antenna orientation, the polarizations are either aligned (e.g. the Tx is
% V-polarized and the Rx is V-polarized), they are crossed (e.g. the Tx is V-polarized and the Rx is
% H-polarized), or the polarization orientation is in between those two. The generated channel
% coefficients should reflect this behavior.

%% Setting up the simulation environment
% First, we have to set up the simulator with some default settings. Here, we choose a center
% frequency of 2.1 GHz. We also want to use drifting in order to get the correct angles for the LOS
% component and we set the number of transmitters and receivers to one.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type

s = qd_simulation_parameters;                           % Set the simulation parameters
s.center_frequency = 2.1e9;                             % Center-frequency: 2.1 GHz
s.samples_per_meter = 360/(40*pi);                      % One sample per degree
s.show_progress_bars = 0;                               % Disable progress bars

%% Setting up the array antennas
% In the second step, we set up our array antennas. We use the synthetic dipole antennas for this
% case. Those antennas show perfect polarization characteristics. First, we generate a single dipole
% with V-polarization. Then, we create multiple copies of this antenna element and rotate them by 45
% and 90 degrees, respectively. We then use the same array antenna for the receiver.

l = qd_layout( s );                                     % Create a new Layout
l.tx_array = qd_arrayant('dipole');                     % create V-polarized dipole
l.tx_array.set_grid( (-180:10:180)*pi/180 , (-90:10:90)*pi/180 );
l.tx_array.Fa = l.tx_array.Fa ./ max(l.tx_array.Fa(:));

l.tx_array.copy_element(1,2:3);                         % Duplicate the elements
l.tx_array.rotate_pattern(45,'y',2);                    % 45 degree polarization
l.tx_array.rotate_pattern(90,'y',3);                    % 90 degree polarization
l.rx_array = l.tx_array;                                % Use the same array for the Rx

set(0,'DefaultFigurePaperSize',[14.5 5.7])              % Adjust paper size for plot
l.tx_array.visualize(1);pause(1);                       % Plot the first antenna element
l.tx_array.visualize(2);pause(1);                       % Plot the second antenna element
l.tx_array.visualize(3);pause(1);                       % Plot the third antenna element

%% Defining a track
% The third step defines the track. Here, we use a circle with 40 m diameter starting in the east,
% traveling north. We also choose a LOS scenario since we want to study the LOS polarization. The
% transmitter is located 12 m north of the center of the circle at an elevation of 6 m.

l.rx_track = qd_track('circular',40*pi,0);              % Circular track, radius 20 m
interpolate_positions( l.rx_track, s.samples_per_meter );  % Interpolate positions
l.tx_position = [ 0 ; 12 ; 6 ];                         % Tx position
l.rx_position = [ 20 ; 0 ; 0 ];                         % Start position for the Rx track
l.set_scenario('BERLIN_UMa_LOS');

set(0,'DefaultFigurePaperSize',[14.5 7.7])              % Adjust paper size for plot
l.visualize;                                            % Plot the layout

%% Generating channel coefficients
% Now, we have finished the parametrization of the simulation and we can generate the channel
% coefficients. We thus create a new set of correlated LSPs and the fix the shadow fading and the
% K-factor to some default values. This disables the drifting for those parameters. We need to do
% that since otherwise, drifting and polarization would interfere with each other.

cb = l.init_builder;                                    % Create parameter sets
cb.scenpar.KF_mu = 3;                                   % Fix KF to 3 dB
cb.scenpar.KF_sigma = 0;
cb.scenpar.SF_sigma = 0;                                % Fix SF to 0 dB
cb.plpar = [];                                          % Disable path loss model

cb.gen_parameters;                                      % Generate small-scale-fading
c = cb.get_channels;                                    % Get the channel coefficients

%% Results and Evaluation
% We now check the results and confirm, if they are plausible or not. We start with the two
% vertically polarized dipoles at the Tx and at the Rx side. The model creates 15 taps, which is the
% default for the "BERLIN_UMa_LOS" scenario. Without path-loss and shadow fading (SF=1), the power
% is normalized such that the sum over all taps is 1 W and with a K-Factor of 3 dB, we get a
% received power of 0.67W for the LOS component. The remaining 0.33 W are in the NLOS components.
% The results can be seen in the following figure.

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure

plot(abs(squeeze( c.coeff(1,1,:,:) )').^2);             % Plot the graph
axis([0 360 -0.1 1]);                                   % Set the axis
xlabel('Position [degrees]');                           % Add description
ylabel('LOS Power, linear scale');
title('Tx: Vertical , Rx: Vertical');                   % Add title

disp(['LOS power:  ',num2str(mean( abs(c.coeff(1,1,1,:)).^2 , 4))])
disp(['NLOS power: ',num2str(mean( sum(abs(c.coeff(1,1,2:end,:)).^2,3) , 4))])

%%
% The LOS power is almost constant when the Rx is south of the Tx. However, in close proximity (at
% 90 degree), the power is lowered significantly. This comes from the 6 m elevation of the Tx. When
% the Rx is almost under the Tx, the radiated power of the Dipole is much smaller compared to the
% broadside direction. The average power of the LOS is thus also lowered to 0.56 W. The average
% sum-power if the 7 NLOS components is 0.26 W. This mainly come from the XPR which leakes some
% power from the vertical- into the horizontal polarization and thus reduces the received power on
% the vertically polarized Dipole. Next, we study two cases. Either the Tx is vertical polarized and
% the Rx is at 45 degree or vise versa.

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot(abs(squeeze( c.coeff(2,1,1,:) )).^2);              % Tx vertical, Rx 45 degree
hold on
plot(abs(squeeze( c.coeff(1,2,1,:) )).^2,'--r');        % Tx 45 degree, Rx vertical
hold off
axis([0 360 -0.1 1]);
legend('Tx vertical, Rx 45 deg', 'Tx 45 deg, Rx vertical')
xlabel('Position [degrees]');
ylabel('LOS Power, linear scale');
title('Tx: Vertical , Rx: 45 deg');

%%
% The receiver changes its direction in a way that it always has the same orientation towards the
% Tx. However, due to the displacement of the Tx, the radiated power towards the Tx becomes minimal
% at around 90 degree. This minimum is visible in both curves (blue and red). However, the pole of
% the 45 degree slanted dipole now points to a different direction which explains the difference in
% the two lines. When the Rx is at 45 degeee and the Tx is vertical, the pole is in the right half
% if the circle - resulting in a lower received power. When the Rx is Vertical and the Tx is 45
% degree, the minimum power is achieved in the left half of the circle.
%
% Next, we evaluate the two dipoles which are rotated by 45 degree. When moving around the circle,
% the Tx stays fixed and the Rx rotates. Subsequently, at one position, we will have both dipoles
% aligned and at another position, both will be crossed. When they are crossed, the received power
% will be 0 and when they are aligned, the power will match the first plot (two vertical dipoles).
% This can be seen in the following figure.

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot(abs(squeeze( c.coeff(2,2,1,:) )).^2 , 'Linewidth',1);
axis([0 360 -0.1 1]);
set(gca,'XTick',0:45:360)
xlabel('Position on circle [degrees]');
ylabel('LOS Power, linear scale');
title('Tx: 45 deg , Rx: 45 deg');

%%
% In the last figure, we have the Tx-antenna turned by 90 degree. It is thus lying on the side and
% it is horizontally polarized. For the Rx, we consider three setups: Vertical (blue line), 45
% degree (green line) and 90 degree (red line). Note that the Tx is rotated around the y-axis. At
% the initial position (0 degree), the Rx (45 and 90 degree) is rotated around the x-axis. This is
% because the movement direction.

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot(abs(squeeze( c.coeff(:,3,1,:) ))'.^2);
axis([0 360 -0.1 1]);
legend('Rx: 0 deg','Rx: 45 deg','Rx: 90 deg' )
xlabel('Position [degrees]');
ylabel('LOS Power, linear scale');
title('Tx: 90 deg , Rx: 0 deg, 45 deg, 90 deg');

%%
% When the receiver is vertical (blue line), both antennas are always crossed. There is no position
% around the circle where a good link can be established. When the receiver is horizontal (red
% line), however, there are two points where the two dipoles are aligned. For the 45 degree dipole,
% the same behavior can be observed but with roughly half the power.
