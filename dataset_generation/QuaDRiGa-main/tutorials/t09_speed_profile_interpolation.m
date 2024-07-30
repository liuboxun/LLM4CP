%% Applying Varying Speeds (Channel Interpolation)
%
% This tutorial shows how to adjust the speed of the terminal, e.g. when breaking or accelerating.
% First, a simple scenario defined. Channel coefficients are calculated at a constant speed and then
% interpolated to match the varying speed of the terminal. One feature that makes the simulations
% more realistic is the function to apply arbitrary speed- and movement profiles, e.g. accelerating,
% breaking or moving at any chosen speed. These profiles are defined in the track class. The
% profiles are then converted in to effective sampling points which aid the interpolation of the
% channel coefficients.

%% Channel model set-up
% First, we set up the simulation parameters. Note the sample density of 1.2 which enables very fast
% simulations even with drifting. The sample density must fulfill the Nyquist theorem, i.e., there
% must be at least 1 sample per half-wavelength in order to be able to interpolate the channels
% correctly. Note that when both transmitter and receiver are mobile, the minimum value is 2 since
% they may move towards each other.

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
s.sample_density = 1.2;                                 % 2.5 samples per half-wavelength
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.show_progress_bars = 0;                               % Disable progress bars

%%
% Second, we define a track. It has a length of 20 m, starts at 10 m east of the transmitter and
% consists of three segments (LOS, NLOS, LOS). The positions are interpolated to match the sample
% density defined above. The track is then plugged into a network layout with one transmitter at
% position (0,0,25). Both, transmitter and receiver are equipped with dipole antennas. The last
% three lines create the large scale parameters (LSPs).

t = qd_track('linear',20,-pi/8);                        % 20 m track, direction SE
t.initial_position = [60;0;1.5];                        % Start position
t.interpolate_positions( 128/20 );                      % Interpolate
t.segment_index       = [1,40,90];                      % Assign segments
t.scenario            = {'BERLIN_UMa_LOS','BERLIN_UMa_NLOS','BERLIN_UMa_LOS'};
t.interpolate_positions( s.samples_per_meter );         % Apply sample density

l = qd_layout( s );                                     % New QuaDRiGa layout
l.tx_array = qd_arrayant('dipole');                     % Set Dipole antenna
l.rx_array = qd_arrayant('dipole');                     % Set Dipole antenna
l.tx_position(3) = 25;                                  % BE height
l.rx_track = t;                                         % Assign track

set(0,'DefaultFigurePaperSize',[14.5 7.7])              % Adjust paper size for plot
l.visualize;                                            % Plot the layout


%% Channel generation and results
% Next, we generate the channel coefficients. Note that here, the initial sample density is 1.2. We
% then interpolate the sample density to 20. It would take ten times as long to achieve the same
% result with setting the initial sample density to 20. The interpolation is significantly faster.
% It is done by first setting the speed to 1 m/s (default setting) and then creating a distance
% vector which contains a list of effective sampling points along the track.

cn = l.get_channels;                                    % Generate channels

t.set_speed( 1 );                                       % Set constant speed
dist = t.interpolate_movement( s.wavelength/(2*20) );   % Get snapshot positions
ci = cn.interpolate( dist );                            % Interpolate channels

%%
% The next plot shows the power of the first three taps from both, the original and the interpolated
% channel, plotted on top of each other. The values are identical except for the fact, that the
% interpolated values (blue line) have 17 times as many sample points.

nsnap = cn.no_snap;                                     % No. snapshots
dist_orig = (0:nsnap-1) * get_length(t)/(nsnap-1);      % Distances
pwr_orig  = 10*log10(squeeze(abs(cn.coeff(1,1,1:3,:))).^2);  % Power before interpolation
pwr_int   = 10*log10(squeeze(abs(ci.coeff(1,1,1:3,:))).^2);  % Power after interpolation

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure

plot( dist_orig,pwr_orig , 'r','Linewidth',2 )
hold on
plot( dist,pwr_int ,'b' )
hold off
axis([min(dist),max(dist), min( pwr_orig( pwr_orig>-160 ) ),...
    max( pwr_orig( pwr_orig>-160 ) )+10 ] );
xlabel('Distance from start point [m]'); ylabel('Power [dB]');

%%
% The following plot shows the power delay profile (PDP) for the interpolated channel. As defined in
% the track object, it starts with a LOS segment, going into a shaded area with significantly more
% multipath fading at around 4 seconds and then back to LOS at around 13 sec.

h = ci.fr( 100e6,512 );                                 % Freq.-domain channel
h = squeeze(h);                                         % Remove singleton dimensions
pdp = 10*log10(abs(ifft(h,[],1).').^2);                 % Power-delay profile

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
imagesc(pdp(:,1:256));

caxis([ max(max(pdp))-50 max(max(pdp))-5 ]); colorbar;  % Figure decorations
cm = colormap('hot'); colormap(cm(end:-1:1,:));
set(gca,'XTick',1:32:255); set(gca,'XTickLabel',(0:32:256)/100e6*1e6);
set(gca,'YTick',1:ci.no_snap/8:ci.no_snap);
set(gca,'YTickLabel', (0:ci.no_snap/8:ci.no_snap)/ci.no_snap * 20 );
xlabel('Delay [\mus]'); ylabel('Time [s]');
title('PDP with fixed speed');

%%
% Now, we create a movement profile. It is defined by a set of value pairs in
% track.movement_profile. The first value represents the time in seconds, the second value the
% position on the track. Here, we start at a position of 7 m, i.e. in the second (NLOS) segment. We
% then go back to the beginning of the track. This takes 5 seconds. Then, we wait there for 1 second
% and go to the end of the track, which we reach after additional 14 seconds. The next step is to
% interpolate the sample points. This is done by the interpolate_movement method. It requires the
% sample interval (in s) as an input argument. Here, we choose an interval of 1 ms which gives us
% 1000 samples per second. The plot the illustrates the results.

t.movement_profile = [ 0,7 ; 5,0 ; 6,0 ; 20,get_length(t)  ]';   % Generate movement profile
dist = t.interpolate_movement( 1e-3 );                  % Get snapshot positions
ci = cn.interpolate( dist );                            % Interpolate channels

nsnap = ci.no_snap;
time = (0:nsnap-1) * t.movement_profile(1,end)/(nsnap-1);

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( time,dist , 'r' )
xlabel('Time [s]'); ylabel('Position on track [m]');
title('Movement profile');

%%
% The last plot shows the PDP of the interpolated channel with the movement profile applied. The
% channel starts in the second segment with a lot of fading, goes back to the first while slowing
% down at the same time. After staying constant for one second, the channel starts running again,
% speeding up towards the end of the track.

h = ci.fr( 100e6,512 );                                 % Freq.-domain channel
h = squeeze(h);                                         % Remove singleton dimensions
pdp = 10*log10(abs(ifft(h,[],1).').^2);                 % Power-delay profile

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
imagesc(pdp(:,1:256));

caxis([ max(max(pdp))-50 max(max(pdp))-5 ]); colorbar;  % Figure decorations
cm = colormap('hot'); colormap(cm(end:-1:1,:));
set(gca,'XTick',1:32:255); set(gca,'XTickLabel',(0:32:256)/100e6*1e6);
set(gca,'YTick',1:ci.no_snap/8:ci.no_snap);
set(gca,'YTickLabel', (0:ci.no_snap/8:ci.no_snap)/ci.no_snap * 20 );
xlabel('Delay [\mus]'); ylabel('Time [s]');
title('PDP with variable speed');

%%
% The following code segment shows a movie of the channel response.
% (You nedd to run the code manually in MATLAB or Octave)

if 0
    h = ci.fr( 20e6,128 );
    h = squeeze(h);
    mi = -90; ma = -80;
    while true
        for n = 1:size(h,2)
            pdp  = 10*log10(abs(h(:,n)).^2);
            plot(pdp)
            ma = max( ma,max([pdp]) );
            mi = min( mi,min([pdp]) );
            axis([1,128,mi,ma])
            title(round(time(n)))
            drawnow
        end
    end
end
