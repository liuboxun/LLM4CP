%% Spatial consistency
%
% Version 2.0 of the QuaDRiGa channel model supports spatial consistency as specified by 3GPP 38.901
% v14.0.0, Section 7.6.3, pp45. This tutorial demonstrates the properties of this feature and how it
% can be used. Spatial consistency can be seen in three aspects of wireless channels:
%
% * The LOS  / NLOS state of a link.
% * The large-scale parameters, such as shadow-fading and delay spread
% * The positions of the scattering clusters as a function of the mobile terminal (MT) position
%
% Here, points 2 and 3 are covered. The large-scale parameter (point 2) are always spatially
% consistent. They change slowly when the terminal moves. For example, two MTs that are close
% together will have similar SFs, DSs and angular spreads. The rate at which the LSPs change is
% adjusted by the "lambda" parameters in the configuration file. For example: "DS_lambda = 20" means
% that the delay spread of two terminal at 20 meters distance will be correlated with correlation
% coefficient of exp(-1) = 0.36. Two terminals at the same positions will see the same DS
% (correlation coefficient is 1). 
%
% The small-scale fading (SSF) is governed by the position of the scattering clusters. Two closely
% spaced terminals will not only have a similar DS, they will also see the same scattering clusters.
% This will have an effect on the achievable data rate. QuaDRiGa implements a 3D correlated random
% process the correlates all random variables that are used to generate the scattering clusters. The
% decorrelation distance of this process (i.e. the distance where the correlation of the same
% variable for 2 users drops to 0.36) is controlled by the parameter "SC_lambda" in the
% configuration files. A value of 0 disables the spatial consistency for SSF.


%% Model setup and channel generation
% First, a new layout is created. The center frequency is set to 2 GHz, the BS height is
% set to 10 m. By default, vertically polarized omni-directional antennas are used.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change paper Size

l = qd_layout;                                          % Create new QuaDRIGa layout
l.simpar.center_frequency = 2e9;                        % Set center frequency to 2 GHz
l.simpar.use_absolute_delays = 1;                       % Enables true LOS delay
l.simpar.show_progress_bars = 0;                        % Disable progress bars
l.tx_position = [ 0,0,10 ]';                            % Set BS posittions

%%
% Next, a new receiver trajectory is created. The track is 50 meters long and starts in
% the north-east of the BS.

l.rx_track = qd_track( 'linear' , 50, pi/2 );           % 50 m long track going north
l.rx_track.initial_position = [20 ; 30 ; 1.5 ];         % Set start position and MT height
l.rx_track.interpolate_positions(10);                   % One channel sample every 10 cm
l.rx_track.scenario = '3GPP_38.901_UMi_NLOS';           % Set propagation scenario

%%
% QuaDRiGa supports two different MT mobility options. By default, drifting is used. This keeps the
% scattering positions fixed for a short segment of the track. Along a segment, path delays and
% angles are updated when the terminal is moving. However, 3GPP 38.901 proposed a different mobility
% option (3GPP 38.901 v14.0.0, Section 7.6.3.2, Option B, pp47). This is implemented in QuaDRiGa as
% well. It is enabled by setting the number of segments on a track equal to the number of snapshots.
% Hence, a new channel realization is created for each position on the track. Mobility is then
% obtained by the spatially consistency procedure.

l.rx_track.no_segments = l.rx_track.no_snapshots;       % Use spatial consisteny for mobility

%%
% Now, a channel builder object is created. The scenario parameters can then be edited to study
% their effects on the results. 

b = l.init_builder;                                     % Initializes channel builder

%%
% 3GPP specifies a cluster delay spread for the two strongest clusters (3GPP 38.901 v14.0.0, Table
% 7.5-5, pp37). When "PerClusterDS" in the configuration file is set to values > 0, the clusters are
% split into three sub-clusters with different delays. However, this is incompatible with spatial
% consistency because the strongest cluster changes over time. Therefore, QuaDRiGa applies the
% cluster delay spread to all clusters which avoids this problem. Here, the cluster delay spread is
% disabled avoid cluttering the results. You can find out what happens when you set to a different
% value. 

b.scenpar.PerClusterDS = 0;                     % Disable per-cluster delay spread
b.scenpar.NumClusters = 5;                      % Only generate 5 clusters
b.scenpar.KF_mu = -3;                           % Set los power to 33 % of the total power
b.scenpar.KF_sigma = 0.5;
b.scenpar.SC_lambda = 5;                        % Set SSF decorrelation distance to 5 m

b.gen_parameters;                               % Generate small-scale-fading parameters

c = get_channels( b );                          % Generate channel coefficients
c = merge( c, [], 0 );                          % Combine output channels
c.individual_delays = 0;                        % Remove per-antenna delays

dl = c.delay.';                                 % Extract path delays from the channel
pow = squeeze( abs(c.coeff).^2 )';              % Calculate path powers from the channel

[len,dist] = get_length( l.rx_track );          % Store the length and distances from start point

%% Path powers
% The first plot shows the path powers along the receiver track. The path parameters (delays,
% angles, power) are generated as described in 3GPP 38.901 v14.0.0, Section 7.6.3.2, Option B,
% pp47). As you can see, path powers do not suddenly "jump", but they change relatively smoothly
% when the MT moves. 

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist,10*log10(pow(:,1)),'-b','Linewidth',2)
hold on; plot( dist,10*log10(pow(:,2:end)),'--'); hold off
axis( [ 0, len, 10*log10(min(pow(:)))-1, 10*log10(max(pow(:)))+1 ] )
grid on
title('Path powers'); xlabel('Distance from start point [m]'); ylabel('Path power [dB]');
legend('LOS path','NLOS paths')


%% Delay spread
% The second plot sows the path delays and the delay spread. As for the powers, delays change
% smoothly over time. NLOS delays can never be smaller than the LOS delay. In addition, the thick
% black line shows the DS at the input of the model and the red, dashed line shows the DS that is
% calculated from the channel coefficients. Both should be identical.

% Calculate DS from the channel coefficients
pow_normalized = pow ./ (sum(pow,2) * ones( 1,size(pow,2) ));
ds = sqrt( sum( pow_normalized .* dl.^2 , 2 ) - sum( pow_normalized .* dl , 2 ).^2 );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist,b.ds*1e6,'-k','Linewidth',2 )
hold on
plot( dist,ds*1e6,'-.r','Linewidth',3 )
plot( dist,dl(:,1)*1e6,'-b','Linewidth',2)
plot( dist,dl(:,2:end)*1e6,'--')
hold off; xlim([0,len]); grid on
title('Delay spread and delays'); xlabel('Distance from start point [m]'); ylabel('Delay [mus]')
legend('Requested DS','Actual DS','LOS delay','NLOS delays');


%% Azimuth of Arrival
% The third plot shows the Azimuth of Arrival (AoA) angles of the paths. As for the DS, the black
% line shows the angular spread (AS) at the model input and the red dashed line the AS at the
% output. Those two lines might be different. The arrival angles are distributed on a sphere and
% therefore, it is not possible to achieve arbitrary angular spreads. At some point the angles just
% wrap around the circle. Therefore, the maximum AS is limited to vales around 80 degrees.

% Calculate AS from the channel coefficients
mean_angle = angle( sum( pow_normalized.*exp( 1j*b.AoA ) , 2 ) );
ang = b.AoA - mean_angle * ones( 1,b.NumClusters );
ang = angle( exp( 1j*ang ) );
as = sqrt( sum(pow_normalized.*ang.^2,2) - sum( pow_normalized.*ang,2).^2 ) * 180/pi;

% Unwrap the angles to illustrate spatial consistency
ang_unwrapped = unwrap(b.AoA,1)*180/pi;

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist,b.asA','-k','Linewidth',2 )
hold on
plot( dist,as,'-.r','Linewidth',3 )
plot( dist, ang_unwrapped(:,1) ,'Linewidth',2)
plot( dist, ang_unwrapped(:,2:end),'--')
plot( dist, zeros(b.no_rx_positions,1),'-k')
hold off; xlim([0,len]); grid on
title('Azimuth of Arrival'); xlabel('Distance from start point [m]'); ylabel('Angle [deg]')
legend('Requested AS','Actual AS','LOS angle', 'NLOS angles')


%% Elevation of Arrival
% Elevation angles are bound between -90 and +90 degrees. The angles also do not change rapidly.  

% Calculate AS from the channel coefficients
mean_angle = angle( sum( pow_normalized.*exp( 1j*b.EoA ) , 2 ) );
ang = b.EoA - mean_angle * ones( 1,b.NumClusters );
ang = angle( exp( 1j*ang ) );
as = sqrt( sum(pow_normalized.*ang.^2,2) - sum( pow_normalized.*ang,2).^2 ) * 180/pi;

% Get angles in degres
ang = b.EoA*180/pi;

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist,b.esA','-k','Linewidth',2 )
hold on
plot( dist,as,'-.r','Linewidth',3 )
plot( dist, ang(:,1) ,'Linewidth',2)
plot( dist, ang(:,2:end),'--')
plot( dist, zeros(b.no_rx_positions,1),'-k')
hold off; xlim([0,len]); grid on
title('Elevation of Arrival'); xlabel('Distance from start point [m]'); ylabel('Angle [deg]')
legend('Requested AS','Actual AS','LOS angle', 'NLOS angles')


%% Azimuth of Departure
% QuaDRiGa calculates the exact positions of the scattering clusters. However, this is not always
% possible. For example, when the path delay is very short and the departure and arrival angles have
% too large values, the cluster positions do not exist. In this case, QuaDRiGa uses a single-bounce
% model, where the departure angles depend on the arrival angles. In this case, the angles of some
% clusters might suddenly change. However, this happens only for spherical waves. You can deactivate
% the sperical waves by setting "l.simpar.use_spherical_waves = 0". In this case, no cluster
% positions are calculated. 

% Calculate AS from the channel coefficients
mean_angle = angle( sum( pow_normalized.*exp( 1j*b.AoD ) , 2 ) );
ang = b.AoD - mean_angle * ones( 1,b.NumClusters );
ang = angle( exp( 1j*ang ) );
as = sqrt( sum(pow_normalized.*ang.^2,2) - sum( pow_normalized.*ang,2).^2 ) * 180/pi;

% Unwrap the angles to illustrate spatial consistency
ang_unwrapped = unwrap(b.AoD,1)*180/pi;

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist,b.asD','-k','Linewidth',2 )
hold on
plot( dist,as,'-.r','Linewidth',3 )
plot( dist, ang_unwrapped(:,1) ,'Linewidth',2)
plot( dist, ang_unwrapped(:,2:end),'--')
plot( dist, zeros(b.no_rx_positions,1),'-k')
hold off; xlim([0,len]); grid on
title('Azimuth of Departure'); xlabel('Distance from start point [m]'); ylabel('Angle [deg]')
legend('Requested AS','Actual AS','LOS angle', 'NLOS angles')


%% Elevation of Departure
% Elevation angles are bound between -90 and +90 degrees. The angles also do not change rapidly
% except for the sudden changes when the model uses single-bounce propagation. 

% Calculate AS from the channel coefficients
mean_angle = angle( sum( pow_normalized.*exp( 1j*b.EoD ) , 2 ) );
ang = b.EoD - mean_angle * ones( 1,b.NumClusters );
ang = angle( exp( 1j*ang ) );
as = sqrt( sum(pow_normalized.*ang.^2,2) - sum( pow_normalized.*ang,2).^2 ) * 180/pi;

% Get angles in degres
ang = b.EoD*180/pi;

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot( dist,b.esD','-k','Linewidth',2 )
hold on
plot( dist,as,'-.r','Linewidth',3 )
plot( dist, ang(:,1) ,'Linewidth',2)
plot( dist, ang(:,2:end),'--')
plot( dist, zeros(b.no_rx_positions,1),'-k')
hold off; xlim([0,len]); grid on
title('Elevation of Departure'); xlabel('Distance from start point [m]'); ylabel('Angle [deg]')
legend('Requested AS','Actual AS','LOS angle', 'NLOS angles')

%% Video
% The last plot shows an visualization of the cluster positions. The spatial consistency model
% ensures that path delays, angles and power change smoothly with time. However, doe to this, all
% cluster appear to be moving through the environment. When angles and delays change rapidly,
% cluster positions change rapidly as well. Sometimes, the speed of the clusters exceed the speed of
% the MT by several order of magnitude. This violates the WSS conditions which state, that for short
% time intervals, the cluster positions stay fixed. Hence, a combination of drifting and spatial
% consistency is needed to achieve realistic channels. (You need to run the code in the loop
% manually) 

set(0,'DefaultFigurePaperSize',[14.5 7.8])            	% Default Paper Size
b.visualize_clusters;
if 0
    for n = 1 : b.no_rx_positions
        b.visualize_clusters(n,[],0);
        title(['Distance from start: ',num2str( dist(n),'%1.1f' ),' m'])
        axis([-100 100 -50 150])
        drawnow
    end
end

