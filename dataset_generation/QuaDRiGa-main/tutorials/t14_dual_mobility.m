%% Dual Mobility
%
% In order to support moving transmitters and receivers (e.g. for car-to-car or device-to-device
% communication), QuaDRiGa 2.2 has been extended to support tracks for the transmitter. This
% tutorial demonstrates how to use the new feature. It covers the following topics:
% 
% * Set-up of a layout with two mobile transceivers (cars moving in opposite directions at different
%   speeds) and one fixed BS 
% * Plot of the coverage area of the fixed BS
% * Calculation of the channels for BS-Car1, BS-Car2, and Car1-Car2
% * Discussion of computational complexity
% * Plot of the path-loss along the trajectory
% * Calculation of the Doppler spectrum for the 3 links
% 
% An initial set of channel parameters is provided for the Urban-Device-to-Device scenario. Those
% have been adopted from the Urban-Microcell scenario al low BS heights. However, new measurements
% are needed for validating the assumptions.

%% Setting general parameters
% We set up some basic parameters such as center frequency and sample density. The minimum sample
% density (samples-per-half-wavelength) must be 1 for static transmitters and 2 for mobile
% transceivers. This ensures that the Doppler characteristics of the channel can be correctly
% captured. During the generation of the channel coefficients, interpolation is used to get the
% correct sample rate (samples-per-second). However, channel interpolation needs much less computing
% time. Increasing the sample density in the simulation parameters increases the accuracy (less
% interpolation artefacts) at the cost of much longer simulation times.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.7])            	% Default Paper Size

s = qd_simulation_parameters;                           % New simulation parameters
s.center_frequency = 2.4e9;                             % 2.4 GHz center frequency
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.sample_density = 2.1;                                 % Minimum possible sample density

%% Defining the layout
% A new layout is created and the static transmitter is defined first. In QuaDRiGa 2.2, each
% transmitter has a track. Static transmitters use zero-length tracks. However, it is possible to
% define a custom orientation for the BS in the track object. Here, the BS is oriented to the
% north-east. 

l = qd_layout( s );                                     % New layout

t = qd_track( 'linear', 0 , pi/4 );                     % Static track facing north-east
t.initial_position = [0;0;6];                           % 6 m height
t.name = 'BS';                                          % Assign unique name

a = qd_arrayant( '3gpp-3d', 8, 4, s.center_frequency, 4, 3 );   % High gain antenna
a.coupling = ones(4,1);                                 % Set horizontal coupling
a.combine_pattern;                                      % Combine radiation pattern
a.normalize_gain;                                       % Normalize gain

l.tx_track(1,1) = t;                                    % Assign static tx track
l.tx_array(1,1) = a;                                    % Tx array

% Calculate antenna footprint 
[ map, x_coords, y_coords] = l.power_map( '3GPP_38.901_UMi_LOS', 'quick',...
    1, -50, 200, -50, 200, 1.5 );
P_db = 10*log10(map{1});                                % LOS pathloss in dB

%%
% Next, we create the first mobile transceiver (Car1). It acts as a receiver for the signals from
% the "BS" and as a transmitter for "Car2". A linear track with 250 m length is created and the
% speed is set to 100 km/h. Hence, the channel is observed to 9 seconds. For the dual-mobility
% feature to work, all tracks in the layout must have the same number of snapshots. By default,
% linear tracks only have a start and an end-point. However, in order to assign segments and
% scenarios to the track, we need to create intermediate positions. Here, we interpolate the track
% so that there is a point for each 10 ms, resulting in 901 "snapshots". Segments are created along
% the track using the "qd_track.set_secenrio" method. The default settings assign a new segment
% roughly every 30 m. Since "Car1" is also a transmitter for "Car2", the same track is used as a
% transmitter track. However, segments are only defined for receiver tracks. Transmitter tracks
% "inherit" their segmentation from the receiver tracks during the channel generation. For example,
% if the receiver track for "Car2" defines a segment from snapshot 200 to snapshot 300, the
% corresponding snapshots 200 to 300 from the transmitter track are used.

t = qd_track( 'linear', 250 , pi/4 );                   % Trajectory of Car 1 moving away from BS
t.set_speed( 100/3.6 );                                 % Speed = 100 km/h
t.interpolate('time',10e-3,[],[],1);                    % Interpolate to to 10 ms grid
t.initial_position = [6;0;1.5];                         % Start position
t.name = 'Car1';                                        % Assign unique name

a = qd_arrayant('dipole');                              % Dipole antenna

l.rx_track(1,1) = t.copy;                               % Assign Rx track 1
l.rx_track(1,1).set_scenario([],[],[]);                 % Create segments (rx-track only)
l.rx_array(1,1) = a;                                    % Assign Rx array 1

l.tx_track(1,2) = t.copy;                               % Assign Rx track 2
l.tx_array(1,2) = a;                                    % Assign Rx array 2

%% 
% The second mobile receiver "Car2" receives both signals from the "BS" and from "Car1". It travels
% at 80 km/h in the opposite direction of "Car1". The track length must be shorter due to the lower
% speed. As for the first track, interpolation is used to  obtain 901 snapshots along the track and
% a different set of segments is created.

t = qd_track( 'linear', 200 , -3*pi/4 );                % Trajectory of Car 2 moving towards BS
t.set_speed( 80/3.6 );                                  % Speed = 80 km/h
t.interpolate('time',10e-3,[],[],1);                    % Interpolate to 10 ms grid
t.initial_position = [171;177;1.5];                     % Start position
t.name = 'Car2';                                        % Assign unique name

l.rx_track(1,2) = t;                                    % Assign Rx track 2
l.rx_track(1,2).set_scenario([],[],[]);                 % Create segments (rx-track only)
l.rx_array(1,2) = a;                                    % Assign Rx array 2

%%
% Now, the scenarios are assigned. The BS-Car links use the default 3GPP Urban-Microcell parameters.
% For Car-Car channels, we use initial Urban-Device-to-Device parameters. Those have not been
% confirmed by measurements yet. Since "Car1" acts as both, a transmitter and a receiver, we also
% need to remove the "Car1-Car1" link from the channel list. Lastly, a plot of the scenario is
% created showing the BS coverge and the trajectories.

l.set_scenario('3GPP_38.901_UMi',[],1,0,40);            % Static transmitter
l.set_scenario('QuaDRiGa_UD2D',  [],2,0,40);            % Mobile tranceivers

l.visualize([],[],0);                                   % Show BS and MT positions on the map
hold on; imagesc( x_coords, y_coords, P_db ); hold off  % Plot the antenna footprint
axis([-50, 200, -50, 200]);
caxis( [-80 -40] );                                     % Color range
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
set(gca,'layer','top')                                  % Show grid on top of the map
colorbar('south')
title('BS1 Path Gain (LOS) [dB]')

%% Calculate channel coefficients
% The following command calculates the channel coefficients once per millisecond. The status update
% is is shown on the command line. This involves the following steps: 
% 
% * Interpolation of the tracks to match the sample density. This avoids unnecessary computations
%   but makes sure, that the Doppler profile is completely captured. At 2.4 GHz carrier frequency,
%   250 m track length, and a sample density of 2.1, 8407 snapshots are needed.
% * Generation of channel builder objects and assigning track segments to builders.
% * Generation of large and small-scale-fading parameters, including spatial consistency.
% * Generation of drifting channel coefficients for each track-segment.
% * Merging of channel segments, including modeling the birth and death of scattering clusters.
% * Interpolation of channel coefficients to match the sample rate. This generates 9001 snapshots at
%   the output. 

l.update_rate = 1e-3;
c = l.get_channels;

%% Path gain
% Now we plot the path-gain for the 3 generated channels. As Car1 moves away from the BS, its PG
% decreases from roughly -40 dB to about -100 dB. Likewise, the PG of Car2 increases. The PG of the
% Car1-Car2 channel starts at a low vale and increases until the cars pass each other at about 4.8
% seconds simulation time. Then, the PG decreases again.

time = ( 0 : c(1,1).no_snap-1 ) * l.update_rate;        % Time axis in seconds
pg = [ c(1,1).par.pg ; c(1,2).par.pg ; c(1,3).par.pg ]; % The path-gain values

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot(time,pg','-','Linewidth',2)                        % Plot target PG
title('Path Gain vs. Time'); 
xlabel('Time [s]'); ylabel('Path Gain [dB]');
axis([0,max(time),min(pg(:))-3,max(pg(:))+3]); grid on;
legend('BS - Car1','BS - Car2','Car1 - Car2')


%% Doppler Spectrum
% The next three plots show the Doppler spectrum of the three channels. For the BS-Car1 link, the
% expected Doppler shift (car moving away from BS) is -220 Hz (v*fc/c). For BS-Car2, it is 180 Hz
% and for Car1-Car2 if goes from 400 to -400 Hz when the cars pass each other. Due to the multipath
% propagation, additional Doppler components occur.

w  = 100;                                               % Doppler analysis windows size (100 ms)
BW = 100e6;                                             % Channel bandwidth (100 MHz)
N  = 128;                                               % Number of carriers

Doppler_axis = -( (0:w-1)/(w-1)-0.5)/l.update_rate;     % The Doppler axis in Hz
time = ( 0 : c(1,1).no_snap-1 ) * l.update_rate; 
Time_axis = time( 1:w:end );                            % Time axis in seconds

no_Doppler = floor( numel(time) ./ w );                 % Number of Doppler samples

for iC = 1 : 3                                          % Repe

    Doppler_spectrum = zeros( w, no_Doppler );          % Preallocate Memory
    for n = 1 : floor( numel(time) ./ w )
        ind = (n-1)*w + 1 : n*w;                        % Snapshot indices
        H = c(1,iC).fr( BW, N, ind );                   % Frequency response of the channel
        H = permute( H,[3,4,1,2] );                     % Reorder dimensions
        G = ifft2(H);                                   % 2D IFFT
        G = fftshift( G,2);                             % Center Doppler spectrum
        Doppler_spectrum( :,n ) = 10*log10( sum( abs(G).^2 , 1 )' );    % Logrithmic power
    end
    
    figure('Position',[ 100 , 100 , 760 , 400]);        % New figure
    imagesc(Time_axis,Doppler_axis,Doppler_spectrum);   % Create images    
    colorbar
    title(['Doppler Spectrum ',regexprep(c(1,iC).name ,'_','-')]);
    xlabel('Time [s]'); ylabel('Doppler shift [Hz]');
    set(gca,'Ydir','Normal')                            % Invert y axis
    colormap jet
end

