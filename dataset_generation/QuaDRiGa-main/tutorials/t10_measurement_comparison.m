%% Resimulating a Measured Scenario
%
% This more complex tutorial shows how to manually define a state sequence (i.e. a sequence of
% scenario transitions), manipulate antennas, create large-scale-parameters such as shadow fading
% and delay spread, and obtain a time series of channel coefficients. This script recreates a
% measured drive test from the Park Inn Hotel at Berlin Alexanderplatz. The transmitter was at the
% rooftop of the hotel while the mobile receiver was moving south on Grunerstraße. A simplified
% version of the scenario is recreated in the simulation where the scenarios along the track were
% classified by hand.

%% Channel model set-up and coefficient generation
% The following code configures some basic parameters.

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
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.show_progress_bars = 0;                               % Disable progress bars

%%
% We generate a track of 500 m length. This track is then interpolated to 1 snapshot per meter. In
% this way, it is possible to assign segments to the track using units of meters. The
% "segment_index" contains the segment start points in units of meters relative to the track start
% point.

t = qd_track('linear',500,-135*pi/180);                 % Track of 500 m length, direction SE
t.initial_position = [120;-120;0];                      % Start position
t.interpolate_positions( 1 );                           % Interpolate to 1 sample per meter
t.segment_index = [1,45,97,108,110,160,190,215,235,245,280,295,304,330,400,430 ]; % Segments

%%
% We now assign the the scenarios to the segments. Since the measurements were done in a satellite
% context, we use the "MIMOSA_10-45_LOS" and "MIMOSA_10-45_NLOS" scenario. The track is then
% interpolated to 3 snapshots per meter.

Sl = 'MIMOSA_10-45_LOS';
Sn = 'MIMOSA_10-45_NLOS';
t.scenario = {Sn,Sl,Sn,Sl,Sn,Sn,Sn,Sl,Sn,Sl,Sn,Sl,Sn,Sn,Sn,Sn};
t.interpolate_positions( 3 );                           % Interpolate to 3 sample per meter

%%
% A new QuaDRiGa layout is created, simulations parameters and the receiver track get assigned. When
% the channel coefficients are generated, there is a merging interval at the end of each segment
% during which paths from the old segment disappear and new paths appear. The method
% "correct_overlap" adjusts the segment start and end-points such that this transitions happens in
% the middle of the assigned segment start and end-points.

l = qd_layout( s );                                     % New QuaDRiGa layout
l.tx_position = [0;0;125];                              % Set the position of the Tx
l.rx_track = copy( t );                                 % Set the rx-track
l.rx_track.correct_overlap;                             % Adjust state change position

%%
% Now, we assign antennas and set the antenna orientations.

l.tx_array = qd_arrayant('lhcp-rhcp-dipole');           % Generate Tx antenna
l.tx_array.rotate_pattern(30,'y');                      % 30 deg downtilt
l.tx_array.rotate_pattern(-90,'z');                     % point southwards

l.rx_array = qd_arrayant('lhcp-rhcp-dipole');           % Rx-Antenna
l.rx_array.rotate_pattern(-90,'y');                     % point skywards

set(0,'DefaultFigurePaperSize',[14.5 7.7])              % Adjust paper size for plot
l.visualize;                                            % Plot the layout
view(-33, 45);                                          % 3D view

% Plot a line from the Tx to the Rx
lnk = [ l.tx_position,...
    l.rx_track.positions(:,l.rx_track.segment_index(2)) + l.rx_track.initial_position ];
hold on; plot3( lnk(1,:),lnk(2,:),lnk(3,:) , '--' ); hold off

%%
% The last step generates the channel coefficients. The warning 'Sample density in tracks does not
% fulfill the sampling theoreme.' refers to the violation of the Nyquist theoreme. Hence, it will
% not be possible to interpolate the channel coefficients. However, this is not required here since
% we are only interested in the channel statistics.

cn = l.get_channels;                                    % Generate channel coefficients
cn.individual_delays = 0;                               % Remove per-antenna delays

%% Results
% First, we plot the PDP vs. distance from the start point. For this, the channel bandwidth is
% reduced to 20 MHz. You can see how the delay of the LOS path sifts  with the distance between BS
% and MT, how the LOS segments have more power, and how NLOS paths appear and disappear along the
% track.

h =  cn.fr( 20e6,256 );                                 % Freq.-domain channel
pdp = squeeze(sum(sum( abs(ifft(h,[],3)).^2 , 1),2));
pdp = 10*log10(pdp.');

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
imagesc(pdp(end:-1:1,1:192));

caxis([ max(max(pdp))-60 max(max(pdp))-5 ]); colorbar;  % Figure decorations
cm = colormap('hot'); colormap(cm(end:-1:1,:));
set(gca,'XTick',1:32:192); set(gca,'XTickLabel',(0:32:192)/20e6*1e6);
ind = sort(cn.no_snap : -cn.no_snap/10 : 1 );
set(gca,'YTick', ind );
set(gca,'YTickLabel', round(sort(500-ind / 3,'descend')) );
xlabel('Delay [\mus]'); ylabel('Distance [m]');
title('PDP with fixed speed');

%%
% The next plot shows the total received power along the trajectory. Green shaded ares are LOS. The
% rest is NLOS. You can see that there is more power when there is LOS propagation.

dist = (1:cn.no_snap)*get_length(t)/cn.no_snap;         % Traveled distance
ind  = find(strcmp(t.scenario,Sl));                     % Find LOS scenarios
los  = [];
for n = 1:numel(ind)
    los = [los t.segment_index(ind(n)) : t.segment_index(ind(n)+1)];
end
ar = zeros(1,cn.no_snap); ar(los) = -200;

power = 10*log10( sum( reshape( abs(cn.coeff).^2 , [] , cn.no_snap ) ,1)/4 );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
a = area(dist,ar);                                      % Shading for the LOS
set(a(1),'FaceColor',[0.7 0.9 0.7]); set(a,'LineStyle','none');
hold on; plot(dist,power); hold off                     % Plot the received power
title('Position dependent power'); xlabel('Track [m]'); ylabel('Power [dB]');
axis([0 500 min(power)-5 max(power)+5]); grid on;
legend('LOS','P_{total}','Location','SouthEast')

%%
% The following plot shows the distribution (PDF) of the received power for both, the LOS and NLOS
% segments.

bins   = -150:2:-80;
p_los  = hist(power(los),bins)/cn.no_snap*100;
p_nlos = hist(power(setdiff(1:cn.no_snap,los)),bins)/cn.no_snap*100;

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
bar(bins,[p_los;p_nlos]');
axis([-124.5,-83,0,ceil(max([p_los,p_nlos]))]); grid on
title('Empirical PDF of the LOS and NLOS power')
xlabel('P_{total} [dB]'); ylabel('Probability [%]'); legend('LOS','NLOS')

%%
% The next plot shows the RMS delay spread along the path. Again, shaded ares are for the LOS
% segments. Due to the strong LOS component, the DS gets shorter during LOS areas.

pow_tap = squeeze(sum(sum(abs(cn.coeff).^2,1),2));
pow_sum = sum( pow_tap,1 );
mean_delay = sum( pow_tap.*cn.delay ,1) ./ pow_sum;
ds = sqrt( sum( pow_tap.*cn.delay.^2 ,1)./ pow_sum - mean_delay.^2 );
ar = zeros(1,cn.no_snap);
ar(los) = 10;

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
a = area(dist,ar);
set(a(1),'FaceColor',[0.7 0.9 0.7]);set(a,'LineStyle','none')
hold on; plot( dist , ds*1e6 ); hold off;               % Plot DS
ma = 1e6*( max(ds)+0.1*max(ds) );axis([0 500 0 ma]);
title('Position dependant delay spread'); grid on
xlabel('Track [m]'); ylabel('Delay Spread [dB]'); legend('LOS','\sigma_\tau');

%%
% The following plot shows the distribution (PDF) of the RMS delay spread for both, the LOS and NLOS
% segments.

bins = 0:0.03:3;
ds_los  = hist(ds(los)*1e6,bins)/cn.no_snap*100;
ds_nlos = hist(ds(setdiff(1:cn.no_snap,los))*1e6,bins)/cn.no_snap*100;

DS  = [ ds_los ; ds_nlos ];
ind = max( find( max(DS/max(DS(:)))>0.001 ) );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
bar(bins,DS');
axis([0,bins(ind),0,max(DS(:))+1]); grid on;
title('Empirical PDF of the LOS and NLOS RMSDS')
xlabel('\sigma_\tau [\mus]'); ylabel('Probability [%]'); legend('LOS','NLOS');

