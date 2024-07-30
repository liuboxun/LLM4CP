%% Network Setup and Parameter Generation
%
% The tutorial demonstrates how to setup a simple layout with multiple receivers, how to adjust
% parameters manually, generate channel coefficients, and how to calculate parameters from
% the data. The channel model class 'qd_builder' generates correlated values for the LSPs. The
% channel builder then uses those values to create coefficients that have the specific properties
% defined in the builder objects. One important question is therefore: Can the same properties which
% are defined in the builder also be found in the generated coefficients? This is an important test
% to verify, if all components of the channel builder work correctly.

%% Channel model setup and coefficient generation
% We first set up the basic parameters.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type

s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 1;                               % Show progress bars
s.center_frequency = 2.53e9;                            % Set center frequency
s.samples_per_meter = 1;                                % 1 sample per meter
s.use_absolute_delays = 1;                              % Include delay of the LOS path

%% Receive antenna
% In order to verify the angular spreads, we need to calculate the angles and the resulting angular
% spreads from the channel coefficients. However, the arrival angle information is embedded in the
% channel coefficients. In order to obtain the angles, we need a special antenna that allows us to
% calculate the arrival angles from the channel response. Such an "ideal" antenna is generated here.
% It consists of 31 elements that allow us to calculate the azimuth and elevation direction of a
% path as well as the polarization.

[ theta, phi, B, d_phi ] = qf.pack_sphere( 27 );        % Generate equidistant directions
N = numel( theta );                                     % Store number of directions
a = qd_arrayant('custom',20,20,0.05);                   % Main beam opening and front-back ratio
a.element_position(1) = 0.2;                            % Element distance from array phase-center
a.copy_element(1,2:N+3);                                % Set number of elements
for n = 1:N                                             % Create sub-elements
    a.rotate_pattern( theta(n)*180/pi,'y',n,1);         % Apply elevation direction
    a.rotate_pattern( phi(n)*180/pi,'z',n,1);           % Apply azimuth direction
end
a.center_frequency = s.center_frequency;                % Set center frequency
a.combine_pattern;                                      % Apply far field transformation
P = sum( abs(a.Fa(:,:,1:N)).^2,3 );                     % Normalize to unit power
a.Fa(:,:,1:N) = a.Fa(:,:,1:N) ./ sqrt(P(:,:,ones(1,N)));

a.Fb(:,:,N+1) = 1;                                      % Add horizontal polarization
a.Fa(:,:,N+1) = 0;

a.Fb(:,:,N+2) = 1/sqrt(2);                              % Add LHCP receive polarization
a.Fa(:,:,N+2) = 1j/sqrt(2);
a.Fb(:,:,N+3) = 1/sqrt(2);                              % Add RHCP receive polarization
a.Fa(:,:,N+3) = -1j/sqrt(2);

%% Layout and Channel Generation
% We have one transmitter and 250 receiver positions. Each receiver gets a specific channel.
% However, the receivers LSPs will be correlated. The BS useses a 2-element antenna that transmits a
% linear polarized signal and an left-hand circular polarized signal. This will allow us to verify
% the correct functionality for both polarizations.

l = qd_layout(s);                                       % Create new QuaDRiGa layout
l.no_rx = 250;                                          % Set number of MTs
l.randomize_rx_positions( 200 , 1.5 , 1.5 , 1.7 );      % 200 m radius, 1.5 m Rx height
l.set_scenario('BERLIN_UMa_NLOS');                      % Use NLOS scenario

l.tx_position(3) = 20;                                  % 20 m tx height
l.tx_array = qd_arrayant( 'omni' );                     % Omni-directional BS antenna
l.tx_array.copy_element(1,2);
l.tx_array.Fa(:,:,2) = 1/sqrt(2);                       % Send additional LHCP signal
l.tx_array.Fb(:,:,2) = 1j/sqrt(2);
l.rx_array = a;                                         % Omni-directional MT antenna

set(0,'DefaultFigurePaperSize',[14.5 7.7])              % Adjust paper size for plot
l.visualize([],[],0);                                   % Plot the layout
view(-33, 60);                                          % Enable 3D view

%%
% We set up the scenario and adjust the parameter range. Then, we generate the channel coefficients.
% In the last step, the arrival angles are obtained from the channel coefficients. This uses only
% the linear polarized transmit singal. 

p = l.init_builder;                                     % Initialize builder
p.plpar = [];                                           % Disable path-loss
p.scenpar.NumClusters = 15;                             % Reduce paths (for faster processing)
p.lsp_xcorr = eye(8);                                   % Disable inter-parameter correlation

p.scenpar.XPR_mu    = 2;                                % Set XPR range
p.scenpar.XPR_sigma = 10;
p.scenpar.KF_mu     = -5;                               % Set KF-Range
p.scenpar.KF_sigma  = 10;
p.scenpar.DS_mu     = log10(0.6e-6);                    % Median DS = 600 ns
p.scenpar.DS_sigma  = 0.3;                              % 300-1200 ns range

p.scenpar.asA_kf = -0.6;                                % Set some inter-parameter correlations
p.scenpar.esA_kf = -0.6;
p.scenpar.esA_asA = 0.5;

p.scenpar.PerClusterAS_A = 1;                           % Limit the per cluster AS to 1 degree
p.scenpar.PerClusterAS_D = 1;
p.scenpar.PerClusterES_A = 1;
p.scenpar.PerClusterES_D = 1;

p.gen_parameters;                                       % Generate small-scale-fading parameters
c = p.get_channels;                                     % Generate channel coefficients

coeff = cat( 5, c.coeff );                              % Extract amplitudes and phases
delay = cat( 5, c.delay );                              % Extract path delays

cf = reshape( coeff(:,1,:,:,:), a.no_elements, 1, [] ); % Format input for angle estimation
[ az, el, J ] = qf.calc_angles( cf, a, 1, [], 1 );      % Calculate angles

%% Results and discussion
% In the following plots, we extract parameters from the generated coefficients and compare them
% with the initial ones which were generated by the 'qd_builder' object (p). The values in (p) can
% be seen as a request to the channel builder and the values in the generated coefficients (c) as a
% delivery. We first calculate the SF from the channel data by summing up the power over all 20
% taps. We see, that the values are almost identical.

sf = mean(sum(sum(abs(coeff(1:29,1,:,:,:)).^2,3),1),4);    % Calculate shadow fading
sf = sf(:);

set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Paper Size
figure('Position',[ 100 , 100 , 760 , 400]);            % New figure

plot(-35:35,-35:35,'k')
hold on
plot([-35:35]+3,-35:35,'--k')
plot([-35:35]-3,-35:35,'--k')
plot( 10*log10(p.sf) , 10*log10(sf) , 'ob','Markerfacecolor','r')
hold off
axis([ -15 , 15 , -15, 15 ])
legend('Equal','+/- 3dB','Location','SouthEast')
xlabel('SF_P [dB]'); ylabel('SF_C [dB]');
title('Shadow Fading - Requested vs. generated value');

%%
% Next, we repeat the same calculation for the K-Factor. Again, we see that the values are almost
% identical.

p_nlos = mean(sum(sum(  abs( coeff(1:29,1,2:end,:,:) ).^2  ,3),1),4);     % Calculate NLOS power
p_los  = mean(sum(sum(  abs( coeff(1:29,1,  1  ,:,:) ).^2  ,3),1),4);     % Calculate LOS power
kf = p_los./p_nlos;                                     % Calculate K-Factor
kf = kf(:);

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot(-35:35,-35:35,'k')
hold on
plot([-35:35]+3,-35:35,'--k')
plot([-35:35]-3,-35:35,'--k')
plot( 10*log10(p.kf) , 10*log10(kf) , 'ok','Markerfacecolor','r')
hold off
axis([ -30 , 30 , -30, 30 ])
legend('Equal','+/- 3dB','Location','SouthEast')
xlabel('KF_P [dB]');
ylabel('KF_C [dB]');
title('K-Factor - Requested vs. generated value');

%%
% Now we repeat the calculation for the RMS delays spread.

pow = reshape( permute( sum(abs(coeff(:,1,:,:,:)).^2,1) , [5,4,3,1,2] ), [], c(1).no_path );
tau = reshape( permute( mean( delay(:,1,:,:,:) ,1)      , [5,4,3,1,2] ), [], c(1).no_path );
pow_sum = sum(pow,2);                                   % Calculate sum-power

pow_tap = abs(coeff).^2;                                % Calculate path powers

mean_delay = sum( pow.*tau,2) ./ pow_sum;               % Calculate mean delay
ds = sqrt( sum( pow.*tau.^2 ,2)./ pow_sum - mean_delay.^2 );
ds = mean( reshape( ds, l.no_rx,[] ),2 );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot([0:0.1:2],[0:0.1:2],'k')
hold on
plot([0:0.1:2]*1.1,[0:0.1:2],'--k')
plot([0:0.1:2],[0:0.1:2]*1.1,'--k')
plot( p.ds'*1e6 , (ds')*1e6 , 'ok','Markerfacecolor','r')
hold off
axis([ 0,1.5,0,1.5 ])
legend('Equal','+/- 10% Error','Location','SouthEast')
xlabel('DS_P [\mus]');
ylabel('DS_C [\mus]');
title('Delay Spread - Requested vs. generated value');

%%
% Now we compare the angular spreads calculated from the channel coefficients with the values in the
% builder. Most values are in the 10% error corridor. The deviations come from the per-cluster
% angular spreads, the limited resolution of the antenna and the dependency of the maximal angular
% spread on the K-Factor.

az = reshape( az, c(1).no_path, [], l.no_rx );

ang = reshape( permute( az, [3,2,1] ), [], c(1).no_path );
asa = qf.calc_angular_spreads( ang,pow );
asa = mean( reshape( asa, l.no_rx,[] ),2 );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot([0:180],[0:180],'k')
hold on
plot([0:180]*1.1,[0:180],'--k')
plot([0:180],[0:180]*1.1,'--k')
plot( p.asA' , asa*180/pi , 'ok','Markerfacecolor','r')
hold off
axis([ 0,80,0,80 ])
legend('Equal','+/- 10% Error','Location','SouthEast')
xlabel('ASA_P [deg]');
ylabel('ASA_C [deg]');
title('Azimuth Spread of Arrival Angles - Requested vs. generated value');

%%
% The same calculations are made for the elevation angles.

el = reshape( el, c(1).no_path, [], l.no_rx );

ang = reshape( permute( el, [3,2,1] ), [], c(1).no_path );
esa = qf.calc_angular_spreads( ang,pow );
esa = mean( reshape( esa, l.no_rx,[] ),2 );

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot([0:180],[0:180],'k')
hold on
plot([0:180]*1.1,[0:180],'--k')
plot([0:180],[0:180]*1.1,'--k')
plot( p.esA' , esa*180/pi , 'ok','Markerfacecolor','r')
hold off
axis([ 0,40,0,40 ])
legend('Equal','+/- 10% Error','Location','SouthEast')
xlabel('ESA_P [deg]');
ylabel('ESA_C [deg]');
title('Elevation Spread of Arrival Angles - Requested vs. generated value');

%%
% The transmitter at the BS sends a vertically polarized and a LHCP wave. When the wave is
% scattered, the polarization is changed. The array antenna is able to measure the Jones-vector of
% the incoming wave (after the reflection). Hence, it is possible to calculate the XPR of the
% scattering events. Likewise, the polarization of the LHCP singal is changed during scattering. We
% use the power-ratio of the RHCP receiv antenna to the LHCP receive antenna to determine the
% circular XPR. 

xpr = abs(J(1,1,:)).^2 ./ abs(J(2,1,:)).^2;
xpr = reshape( xpr, c(1).no_path, [], l.no_rx );
xpr = mean(mean(xpr(2:end,:,:),1),2);
xpr = xpr(:);

xprC = abs(coeff(31,2,:,:,:)).^2 ./ abs(coeff(30,2,:,:,:)).^2;
xprC = mean(mean(xprC(1,1,2:end,:,:),3),4);
xprC = xprC(:);

figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
plot(-35:35,-35:35,'k')
hold on
plot([-35:35]+3,-35:35,'--k')
plot( 10*log10(p.xpr)' , 10*log10(xpr) , 'ok','Markerfacecolor','b')
plot( 10*log10(p.xpr)' , 10*log10(xprC) , 'ok','Markerfacecolor','r')
plot([-35:35]-3,-35:35,'--k')
hold off
axis([ -30 , 30 , -30, 30 ])
legend('Equal','+/- 3dB','Linear XPR','Circular XPR','Location','SouthEast')
xlabel('XPR_P [dB]');
ylabel('XPR_C [dB]');
title('XPR - Requested vs. generated value');

%%
% Lastly, it is checked if the requested inter-parameter correlations are also found in the
% channel coefficients.

disp(['Corr. KF  - ASA :  ',num2str(qf.xcorrcoeff( 10*log10(kf) , log10(asa)    ),'%1.2f')])
disp(['Corr. KF  - ESA :  ',num2str(qf.xcorrcoeff( 10*log10(kf) , log10(esa)    ),'%1.2f')])
disp(['Corr. KF  - SF  :  ',num2str(qf.xcorrcoeff( 10*log10(kf) , 10*log10(sf) ),'%1.2f')])
disp(['Corr. KF  - DS  :  ',num2str(qf.xcorrcoeff( 10*log10(kf) , log10(ds)     ),'%1.2f')])
disp(['Corr. DS  - SF  :  ',num2str(qf.xcorrcoeff( log10(ds)    , 10*log10(sf) ),'%1.2f')])
disp(['Corr. DS  - ASA :  ',num2str(qf.xcorrcoeff( log10(ds)    , log10(asa)    ),'%1.2f')])
disp(['Corr. DS  - ESA :  ',num2str(qf.xcorrcoeff( log10(ds)    , log10(esa)    ),'%1.2f')])
disp(['Corr. ASA - ESA :  ',num2str(qf.xcorrcoeff( log10(asa)   , log10(esa)    ),'%1.2f')])

