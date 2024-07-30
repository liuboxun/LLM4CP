%% Typical driving course
%
% This tutorial is a step-by-step walk through of the example given in section 1.6 of the
% documentation. A 800 m long drive course is covered by a S-band satellite. A car moves along the
% trajectory where it experiences different reception conditions. The tutorial coverers:
%
% * Setting up the trajectory
% * Assigning propagation environments to different sections of the track
% * Modeling stops at traffic lights
% * Setting up antennas for the satellite and the car
% * Generating channel coefficients
% * Analyzing the received power and the cross-polarization ratio
%
% A figure illustrating the scenario can be found in the documentation in Section 1.6. There are 12
% significant points along the track that describe an event.
%
% # Start environment: Urban, LOS reception of satellite signal
% # LOS to NLOS transition
% # NLOS to LOS change
% # Turning off without change in reception condition (LOS)
% # Stopping at traffic light (LOS)
% # Turning off with change of reception condition (LOS to NLOS)
% # Crossing side street (NLOS to short LOS to NLOS)
% # Structural change in the environment without a change in the environment type (higher density of
%   buildings but still the environment remains urban)
% # Stopping at traffic lights (NLOS)
% # Houses have the same characteristics as before but are further away from the street (urban
%   environment with different reception characteristics)
% # Change of environment (Urban to Forest)
% # Turning off without change of environment (NLOS)

%% Setting up the trajectory
% The trajectory consists of 4 straight segments of 200 m, 100 m, 400 m, and 53 m length. These
% segments are connected by 90 degree turns. We models these turns by arc segments having a radius
% of 10 m, leading to 15.7 m length. Hence, the total track length is roughly 800 meters. The
% following code example shows how the track can be created. In the last step, the track is plotted.

clear all
close all

t = qd_track('linear',200,pi/4);                        % P1-P4: 200 m segment, direction NE
t.name = 'Terminal';                                    % Set track name
t.initial_position(3,1) = 2;                            % Set the Rx height to 2 meters

c = 10*exp(1j*(135:-1:45)*pi/180);                      % P4: Turn NE to SE, 10 m curve radius
c = c(2:end)-c(1);                                      % Start relative to [x,y] = [0,0]
t.positions = [t.positions,...                          % Append curve to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

c = 100*exp( -1j*pi/4 );                                % P4-P6: 200 m segment, direction SE
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

c = 10*exp(1j*(-135:-45)*pi/180);                       % P6: Turn SE to NE, 10 m curve radius
c = c(2:end)-c(1);                                      % Start relative to [x,y] = [0,0]
t.positions = [t.positions,...                          % Append curve to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

c = 400*exp( 1j*pi/4 );                                 % P6-P12: 400 m segment, direction NE
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

c = 10*exp(1j*(135:-1:45)*pi/180);                      % P12: Turn NE to SE, 10 m curve radius
c = c(2:end)-c(1);                                      % Start relative to [x,y] = [0,0]
t.positions = [t.positions,...                          % Append curve to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

c = 53*exp( -1j*pi/4 );                                 % P12-end: 53 m segment, direction SE
t.positions = [t.positions,...                          % Append curve to track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

t.calc_orientation;                                     % Calculate the receiver orientation

set(0,'defaultTextFontSize', 18)                      	% Default Font Size
set(0,'defaultAxesFontSize', 18)                     	% Default Font Size
set(0,'defaultAxesFontName','Times')               	    % Default Font Type
set(0,'defaultTextFontName','Times')                 	% Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')       	% Default Plot position
set(0,'DefaultFigurePaperType','<custom>')             	% Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.8])            	% Default Paper Size

l = qd_layout;                                          % New layout
[~,l.rx_track] = interpolate( t.copy,'distance',0.1 );  % Interpolate and assign track to layout
l.visualize([],[],0);                                   % Plot
axis equal
title('Track layout');                                  % Set plot title

%% Assigning propagation environments
% We now assign propagation environments to the track. The easiest way to do this is by using the
% "add_segment" method. This method requires 3D-coordinates of a point near the track as well as a
% scenario description. The easiest way to obtain these coordinates is to use the data cursor in the
% plot and read the coordinates from the pop-up window. Scenario descriptions for satellite
% scenarios are provided by 3GPP TR 38.811. The propagation parameters are stored in configuration
% files in the QuaDRiGa source folder. Here, we only need the scenario name.

t.scenario{1} = 'QuaDRiGa_NTN_Urban_LOS';                % P1: Start scenario: Urban LOS
t.add_segment ([64;64;2],'QuaDRiGa_NTN_Urban_NLOS',2);   % P2: LOS to NLOS change
t.add_segment ([84;84;2],'QuaDRiGa_NTN_Urban_LOS',2);    % P3: NLOS to LOS change
t.add_segment ([233;68;2],'QuaDRiGa_NTN_Urban_NLOS',2);  % P6: LOS to NLOS change
t.add_segment ([272;103;2],'QuaDRiGa_NTN_Urban_LOS',2);  % P7: NLOS to LOS change
t.add_segment ([283;114;2],'QuaDRiGa_NTN_Urban_NLOS',2); % P7: LOS to NLOS change
t.add_segment ([324;153;2],'QuaDRiGa_NTN_DenseUrban_NLOS',2);% P8: Higher density of buildings
t.add_segment ([420;250;2],'QuaDRiGa_NTN_Urban_NLOS',2); % P10: Lower density of buildings
t.add_segment ([490;320;2],'QuaDRiGa_NTN_Rural_NLOS',2); % P11: Urban to Rural

%% Modeling stops at traffic lights
% This section provides a simple way to model the movement of the car along the track. A movement
% profile describes the movement along the track by associating a time points (in seconds) with a
% traveled distance (in meters). This is assigned to the track object. The initial speed of the car
% is set to 10 m/s for the first 20 seconds. Then it slows down and stops after 30 seconds at the
% first traffic light. The stopping duration is 10 seconds. Another 6.5 second stop happens after
% 66.5 seconds or at 530 meters relative to the start. The total simulation time is 100 seconds.
% Note that accelerations are modeled. Speed changes happen suddenly as can be seen in the plot at
% the end of the section. For a smoother movement, it is advisable to sample the movement profile
% more often.

t.movement_profile = [ 0, 20, 30, 40, 66.5, 73, 100;... % Time points in seconds vs.
    0, 200, 265, 265, 530, 530, 800 ];                  %    distance in meters
dist  = t.interpolate('time',0.1);                      % Calculate travelled distance vs. time
time  = ( 0:numel(dist) - 2 )*0.1;                      % Calculate time sample points
speed = diff( dist ) * 10;                              % Calculate the speed

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Plot Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure

plot( time,speed,'Linewidth',2 );                       % Plot speed vs. time
xlabel('Simulation Time (s)'); ylabel('Speed (m/s)'); grid on;     % Annotations
axis([0,100,0,11]);

%% Simulation layout and antenna setup
% This section shows how create a simulation layout, set up the center frequency and set up
% antennas. The system operates in S-band at a 2.2 GHz carrier frequency. The satellite uses a
% parabolic dish antenna of 3 m diameter, a gain of 44 dBi, and LHCP polarization. The terminal uses
% a dual-polarized patch antenna (LHCP/RHCP) which is pointing upwards to the sky. To verify the
% correct configuration, we plot the beam footprint at the end of this section. A TX power of 100 W
% is assumed for the satellite. This would lead to an equivalent isotropically radiated power (EIRP)
% of 64 dBW for the space segment. The beam footprint takes the antenna gains as well as the
% antenna orientation at the satellite into account. The curvature of the Earth is ignored here.

l = qd_layout;                                          % Create new layout
l.simpar.center_frequency = 2.2e9;                      % Set center frequency to 2.2 GHz

l.rx_track = t;                                         % Assign terminal track for the receiver
l.rx_track.split_segment(10,50,30,12);                  % Create more segments
l.rx_track.correct_overlap;                             % Fix the segment start positions

l.set_satellite_pos( 52.3, 29.7, 172.7 );               % Set GEO satellite position
l.tx_array = qd_arrayant( 'parabolic', 3, l.simpar.center_frequency, [] , 3);       % Sat. antenna
l.tx_track.orientation = [ 0 ; -29.7 ; 97.3 ]*pi/180;   % Set the orientation of tx antenna
l.tx_name{1} = 'Sat';                                   % Set TX name

l.rx_array = qd_arrayant('patch');                      % Patch antenna for the terminal
l.rx_array.center_frequency = l.simpar.center_frequency;     % Set antenna frequency
l.rx_array.copy_element(1,2);                          	% Two identical elements
l.rx_array.rotate_pattern(90,'x',2);                   	% Rotate second element by 90 degrees
l.rx_array.coupling = 1/sqrt(2) * [1 1 ; 1j -1j];      	% Set LHCP / RHCP polarization
l.rx_array.combine_pattern;                           	% Merge polarized patterns
l.rx_array.rotate_pattern(-90,'y');                    	% Point skywards

% Calculate the beam footprint
set(0,'DefaultFigurePaperSize',[14.5 7.8])              % Adjust paper size for plot
[map,x_coords,y_coords]=l.power_map('5G-ALLSTAR_Urban_LOS','quick',2e4,-6e6,6e6,-5e6,5e6);
P = 10*log10( map{:}(:,:,1) ) + 50;                     % RX copolar power @ 50 dBm TX power
l.visualize([],[],0);                                   % Plot layout
axis([-5e6,5e6,-5e6,5e6]);                              % Axis
hold on
imagesc( x_coords, y_coords, P );                       % Plot the received power
hold off

colorbar('South')                                       % Show a colorbar
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
axis equal
set(gca,'XTick',(-5:5)*1e6);
set(gca,'YTick',(-5:5)*1e6);
caxis([-150,-90])
set(gca,'layer','top')                                  % Show grid on top of the map
title('Beam footprint in dBm');                         % Set plot title

%% Generating and analyzing channel coefficients
% Now we generate the channel coefficients and plot the power in both polarizations over time. The
% plot is annotated to show the events that happen during the simulation.

l.update_rate = 0.01;                                   % Set channel update rate to 100 Hz
c = l.get_channels;                                     % Generate channels

pow  = 10*log10( reshape( sum(abs(c.coeff(:,:,:,:)).^2,3) ,2,[] ) );    % Calculate the power
time = (0:c.no_snap-1)*0.01;                            % Vector with time samples

ar   = zeros(1,c.no_snap);                              % Shading of events
ar(900:1200) = -200;                                    % NLOS from P2 to P3
ar(3000:4000) = -200;                                   % Stop at P5
ar(4650:5050) = -200;                                   % NLOS from P6 to P7
ar(5300:5800) = -200;                                   % NLOS from P6 to P7
ar(6650:7300) = -200;                                   % Stop at P9
ar(7800:8900) = -200;                                   % Stop at P9

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Plot Size
figure('Position',[ 100 , 100 , 1200 , 400]);           % New figure
a = area(time,ar,'FaceColor',[0.7 0.9 0.7],'LineStyle','none'); % Area shading
hold on; plot(time,pow'+50); hold off;
xlabel('Simulation Time (s)'); ylabel('RX power (dBm)'); grid on; axis([0,100,[-150,-80]]);
legend('Event','RX LHCP','RX RHCP'); set(gca,'layer','top')

text( 7,-85,'P2' ); text( 11,-85,'P3' ); text( 8,-145,'NLOS' ); text( 20,-85,'P4' );
text( 33,-85,'P5' ); text( 32,-145, 'Stop' ); text( 45.5,-85,'P6' ); text( 50.5,-85,'P7' );
text( 44,-145,'NLOS' ); text( 57,-85,'P8' ); text( 53,-145,'NLOS' ); text( 69,-85,'P9' );
text( 68,-145, 'Stop' ); text( 77,-85,'P10' ); text( 80,-145, 'Urban' );text( 92,-145, 'Rural' );
