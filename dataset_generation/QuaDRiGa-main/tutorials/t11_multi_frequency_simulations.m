%% Multi-frequency simulations
%
% This tutorial demonstrates how to perform simultaneous multi-frequency simulations at two carrier
% frequencies: 2.6 GHz and 28 GHz in an Urban-Macrocell deployment. The BS is equipped with two
% different array antennas. A conventional high-gain antenna operates at 2.6 GHz. The higher
% frequency band uses a massive-MIMO array antenna with in an 8x8 dual-polarized setup. The model is
% consistent in both, the spatial domain and the frequency domain. Simulation assumptions are in
% accordance with 3GPP 38.901 v14.1.0 (see Section 7.6.5 Correlation modeling for multi-frequency
% simulations).
%
% Identical parameters for each frequency:
%
% * LOS / NLOS state must be the same
% * BS and MT positions are the same (antenna element positions are different!)
% * Cluster delays and angles for each multi-path component are the same
% * Spatial consistency of the LSPs is identical
%
% Differences:
%
% * Antenna patterns are different for each frequency
% * Path-loss is different for each frequency
% * Path-powers are different for each frequency
% * Delay- and angular spreads are different
% * K-Factor is different
% * XPR of the NLOS components is different


%% Basic setup
% Multiple frequencies are set in the simulation parameters by providing a vector of frequency
% sample points. A new layout is created with one 25 m high BS positions and 100 MT positions. The
% MTs are placed in accordance with the 3GPP assumptions, where 80% of them are situated indoors at
% different floor levels.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.7])            	% Default Paper Size

s = qd_simulation_parameters;
s.center_frequency = [2.6e9, 28e9];                     % Assign two frequencies

l = qd_layout( s );                                     % New QuaDRiGa layout
l.tx_position = [0 0 25]';                              % 25 m BS height
l.no_rx = 100;                                          % 100 MTs

l.randomize_rx_positions( 200, 1.5, 1.5, 0 );           % Assign random user positions
l.rx_position(1,:) = l.rx_position(1,:) + 220;          % Place users east of the BS

floor = randi(5,1,l.no_rx) + 3;                         % Set random floor levels
for n = 1:l.no_rx
    floor( n ) =  randi(  floor( n ) );
end
l.rx_position(3,:) = 3*(floor-1) + 1.5;

indoor_rx = l.set_scenario('3GPP_38.901_UMa',[],[],0.8);    % Set the scenario
l.rx_position(3,~indoor_rx) = 1.5;                      % Set outdoor-users to 1.5 m height

%% Antenna set-up
% Two different antenna configurations are used at the BS. The 2.6 GHz antenna is constructed from 8
% vertically stacked patch elements with +/- 45 degree polarization. The electric downtilt is set to
% 8 degree. The mm-wave antenna uses 64 dual-polarized elements in a 8x8 massive-MIMO array
% configuration. The antennas are assigned to the BS by an array of "qd_arrayant" objects. Rows
% correspond to the frequency, columns to the BS. There is only 1 BS in the layout. The mobile
% terminal uses a vertically polarized omni-directional antenna for both frequencies.

a_2600_Mhz  = qd_arrayant( '3gpp-3d',  8, 1, s.center_frequency(1), 6, 8 );
a_28000_MHz = qd_arrayant( '3gpp-3d',  8, 8, s.center_frequency(2), 3 );

l.tx_array(1,1) = a_2600_Mhz;                           % Set 2.6 GHz antenna
l.tx_array(2,1) = a_28000_MHz;                          % Set 28 Ghz antenna

l.rx_array = qd_arrayant('omni');                       % Set omni-rx antenna

%% Coverage preview
% Next, we create a preview of the antenna footprint. We calculate the map for the two frequencies
% including path-loss and antenna patterns. The first plot is for the 2.6 GHz band.

sample_distance = 5;                                    % One pixel every 5 m
x_min           = -50;                                  % Area to be samples in [m]
x_max           = 550;
y_min           = -300;
y_max           = 300;
rx_height       = 1.5;                                  % Mobile terminal height in [m]
tx_power        = 30;                                   % Tx-power in [dBm] per antenna element
i_freq          = 1;                                    % Frequency index for 2.6 GHz

% Calculate the map including path-loss and antenna patterns
[ map, x_coords, y_coords] = l.power_map( '3GPP_38.901_UMa_LOS', 'quick',...
    sample_distance, x_min, x_max, y_min, y_max, rx_height, tx_power, i_freq );
P_db = 10*log10( sum( map{1}, 4 ) );

% Plot the results
l.visualize([],[],0);                                   % Show BS and MT positions on the map
hold on; imagesc( x_coords, y_coords, P_db ); hold off  % Plot the antenna footprint
axis([x_min,x_max,y_min,y_max]);
caxis( max(P_db(:)) + [-20 0] );                        % Color range
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
set(gca,'layer','top')                                  % Show grid on top of the map
colorbar('south')
title('Received power [dBm] for 2.6 GHz band')

%%
% For the 28 GHz, we get the complex-valued phases for each antenna element in order
% to calculate a MRT beamformer that points the towards the ground at coordinates x = 200 m and
% y = 100 m.

tx_power        = 10;                                   % Tx-power in [dBm] per antenna element
i_freq          = 2;                                    % Frequency index for 28 GHz

% Calculate the map including path-loss and antenna patterns
[ map, x_coords, y_coords] = l.power_map( '3GPP_38.901_UMa_LOS', 'phase',...
    sample_distance, x_min, x_max, y_min, y_max, rx_height, tx_power, i_freq );

% Calculate MRT beamforming weights
beam_x = find( x_coords >= 200 , 1 );                   % Point the beam at x = 200 and y = 100
beam_y = find( y_coords >= 100  , 1 );
w = conj( map{1}( beam_y, beam_x , 1 ,: ) );            % Precoding weights for a MRT beamformer
w = w ./ sqrt(mean(abs(w(:)).^2));                      % Normalize to unit power

% Apply the precoding weights to each pixel on the map and calculate the received power
P_db = map{1} .* w( ones(1,numel(y_coords)), ones(1,numel(x_coords)),:,: );
P_db = 10*log10( abs( sum( P_db ,4 ) ).^2 );

l.visualize([],[],0);                                   % Show BS and MT positions on the map
hold on; imagesc( x_coords, y_coords, P_db ); hold off  % Plot the antenna footprint
axis([x_min,x_max,y_min,y_max]);
caxis( max(P_db(:)) + [-20 0] );                        % Color range
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
set(gca,'layer','top')                                  % Show grid on top of the map
colorbar('south')
title('Received power [dBm] for 28 GHz band')

%% Generate channel coefficients
% Channel coefficients are generated by calling "l.get_channels". The output is an array of QuaDRiGa
% channel objects. The first dimension corresponds to the MTs (100). The second dimension
% corresponds to the number of BSs (1) and the third dimension corresponds to the number of
% frequencies (2).

c = l.get_channels;

