%% Time Evolution and Scenario Transitions
%
% This tutorial shows how user trajectories, segments, and scenarios are defined. Channel
% coefficients are created for each segment separately. The channel merger combines these output
% into a longer sequence. The output sequences are evaluated for different settings of the model.
% The channel model generates the coefficients separately for each segment. In order to get a
% time-continuous output, these coefficients have to be combined. This is a feature which is
% originally described in the documentation of the WIM2 channel model, but which was never
% implemented. Since this component is needed for time-continuous simulations, it was implemented
% here. This script sets up the simulation and creates such time-continuous CIRs.

%% Channel model setup and coefficient generation
% First, we set up the channel model.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type

s = qd_simulation_parameters;                           % New simulation parameters
s.center_frequency = 2.53e9;                            % 2.53 GHz carrier frequency
s.sample_density = 4;                                   % 4 samples per half-wavelength
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.show_progress_bars = 1;                               % Disable progress bars

%%
% Second, we create a more complex network layout featuring an elevated transmitter (25 m) and two
% receivers at 1.5 m height. The first Rx moves along a circular track around the receiver. The
% second receiver moves away from the Tx. Both start at the same point. Note here, that each track
% is split into three segments. The first Rx goes from an LOS area to a shaded area and back. The
% second track also start in the LOS area. Here, the scenario changes to another LOS segment and
% then to an NLOS segment. The LOS-LOS change will create new small-scale fading parameters, but the
% large scale parameters (LSPs) will be highly correlated between those two segments.

l = qd_layout(s);                                          % Create new QuaDRiGa layout
l.no_rx = 2;                                               % Two receivers
l.tx_array = qd_arrayant('dipole');                        % Dipole antennas at all Rx and Tx
l.rx_array = l.tx_array;
l.tx_position(3) = 25;                                     % Elevate Tx to 25 m

UMal = 'BERLIN_UMa_LOS';                                   % LOS scenario name
UMan = 'BERLIN_UMa_NLOS';                                  % NLOS scenario name

l.rx_track(1,1) = qd_track('circular',20*pi,0);            % Circular track with 10m radius
l.rx_track(1,1).initial_position  = [10;0;1.5];            % Start east, running north
l.rx_track(1,1).segment_index     = [1,40,90];             % Segments
l.rx_track(1,1).scenario          = { UMal, UMan, UMal };  % Scenarios
l.rx_track(1,1).name = 'Rx1';

l.rx_track(1,2) = qd_track('linear',20,pi/8);              % Linear track, 20 m length
l.rx_track(1,2).initial_position  = [10;0;1.5];            % Same start point
l.rx_track(1,2).interpolate_positions( 128/20 );
l.rx_track(1,2).segment_index     = [1,40,90];             % Segments
l.rx_track(1,2).scenario          = { UMal, UMal, UMan };  % Scenarios
l.rx_track(1,2).name = 'Rx2';

set(0,'DefaultFigurePaperSize',[14.5 7.7])                 % Adjust paper size for plot
l.visualize;                                               % Plot the layout

interpolate_positions( l.rx_track, s.samples_per_meter );  % Interpolate
calc_orientation( l.rx_track );                            % Align antenna direction with track

%%
% Now we create the channel coefficients. The fixing the random seed guarantees repeatable results
% (i.e. the taps will be at the same positions for both runs). Also note the significantly longer
% computing time when drifting is enabled.

disp('Drifting enabled:');
p = l.init_builder;                                         % Create channel builders
gen_parameters( p );                                        % Generate small-scale fading
c = get_channels( p );                                      % Generate channel coefficients
cn = merge( c );

disp('Drifting disabled:');
warning('off','QuaDRiGa:qd_builder:gen_ssf_parameters:exisitng')

s.use_3GPP_baseline = 1;                                    % Disable drifting
gen_parameters(p,2);                                        % Update small-scale fading
gen_parameters(p,3);                                        % Calc. FBS / LBS Positions
d = get_channels( p );                                      % Generate channel coefficients
dn = merge( d );

%% Results and discussion
% Now we plot the and discuss the results. We start with the power of the LOS tap along the circular
% track and compare the outcome with and without drifting.

degrees = (0:cn(1,1).no_snap-1)/cn(1).no_snap * 360;
los_pwr_drift = 10*log10(squeeze(abs(cn(1).coeff(1,1,1,:))).^2);
los_pwr_nodrift = 10*log10(squeeze(abs(dn(1).coeff(1,1,1,:))).^2);

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( degrees,los_pwr_drift )
hold on
plot(degrees,los_pwr_nodrift ,'-.r')
hold off

a = axis; axis( [0 360 a(3:4) ] );
xlabel('Position on circle [deg]');
ylabel('Power of the LOS component');
title('Power of the LOS component for the circular track');
legend('Drifting','No drifting','Location','SouthEast');

%%
% When drifting is enabled (blue curve), the channel output after merging is time-continuous. The
% variations along the track come from the drifting K-Factor and the drifting shadow fading. When
% drifting is disabled, these parameters are not updated and kept fixed at their initial value. At
% the end of each segment, both channels are cross-faded, i.e. the power of the output of the first
% segment ramps down and the power of the second segment ramps up. Since drifting guarantees a
% time-continuous evolution of the phase, this ramping process is also time continuous and no
% artifacts are visible in the blue curve. Without drifting, the phases are approximated based on
% their initial values, the initial arrival and departure angles and the traveled distance from the
% start point. However, since the Rx moves along a circular track, the angles change continuously
% which is not correctly modeled. The phase at the end of the first segment does not match the phase
% at the beginning of the second. When adding both components, artifacts appear as can be seen in
% the red curve.
%
% Next, we plot the power-delay profiles for both tracks. We calculate the frequency response of the
% channel and transform it back to time domain by an IFFT. Then, we create a 2D image of the
% received power at each position of the track. We start with the circular track.

h = cn(1,1).fr( 100e6,512 );                            % Freq.-domain channel
h = squeeze(h);                                         % Remove singleton dimensions
pdp = 10*log10(abs(ifft(h,[],1).').^2);                 % Power-delay profile

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
imagesc(pdp(:,1:256));

caxis([ max(max(pdp))-50 max(max(pdp))-5 ]); colorbar;  % Figure decorations
cm = colormap('hot'); colormap(cm(end:-1:1,:));
set(gca,'XTick',1:32:255); set(gca,'XTickLabel',(0:32:256)/100e6*1e6);
set(gca,'YTick',1:cn(1).no_snap/8:cn(1).no_snap);
set(gca,'YTickLabel', (0:cn(1).no_snap/8:cn(1).no_snap)/cn(1).no_snap * 360 );
xlabel('Delay [\mus]'); ylabel('Position on circle [deg]');
title('PDP for the circular track with drifting');

%%
% The X-axis shows the delay in microseconds and the Y-axis shows the position on the circle. For
% easier navigation, the position is given in degrees. 0 deg means east (starting point), 90 deg
% means north, 180 deg west and 270 deg south. The LOS delay stays constant since the distance to
% the Tx is also constant. However, the power of the LOS changes according to the scenario. Also
% note, that the NLOS segment has more paths due to the longer delay spread.
%
% Next, we create the same plot for the linear track. Note the slight increase in the LOS delay and
% the high similarity of the first two LOS segments due to the correlated LSPs. Segment change is at
% around 6 m.

h = cn(1,2).fr( 100e6,512 );                            % Freq.-domain channel
h = squeeze(h);                                         % Remove singleton dimensions
pdp = 10*log10(abs(ifft(h,[],1).').^2);                 % Power-delay profile

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
imagesc(pdp(:,1:256));

caxis([ max(max(pdp))-50 max(max(pdp))-5 ]); colorbar;  % Figure decorations
cm = colormap('hot'); colormap(cm(end:-1:1,:));
set(gca,'XTick',1:32:255); set(gca,'XTickLabel',(0:32:256)/100e6*1e6);
set(gca,'YTick',1:cn(2).no_snap/8:cn(2).no_snap);
set(gca,'YTickLabel', (0:cn(2).no_snap/8:cn(2).no_snap)/cn(2).no_snap * 20 );
xlabel('Delay [\mus]'); ylabel('Distance from start point [m]');
title('PDP for the linear track with drifting');

%%
% Last, we plot the same results for the linear track without drifting. Note here, that the LOS
% delay is not smooth during segment change. There are two jumps at 6 m and again at 13.5 m.

h = dn(1,2).fr( 100e6,512 );                            % Freq.-domain channel
h = squeeze(h);                                         % Remove singleton dimensions
pdp = 10*log10(abs(ifft(h,[],1).').^2);                 % Power-delay profile

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
imagesc(pdp(:,1:256));

caxis([ max(max(pdp))-50 max(max(pdp))-5 ]); colorbar;  % Figure decorations
cm = colormap('hot'); colormap(cm(end:-1:1,:));
set(gca,'XTick',1:32:255); set(gca,'XTickLabel',(0:32:256)/100e6*1e6);
set(gca,'YTick',1:cn(2).no_snap/8:cn(2).no_snap);
set(gca,'YTickLabel', (0:cn(2).no_snap/8:cn(2).no_snap)/cn(2).no_snap * 20 );
xlabel('Delay [\mus]'); ylabel('Distance from start point [m]');
title('PDP for the linear track without drifting');

