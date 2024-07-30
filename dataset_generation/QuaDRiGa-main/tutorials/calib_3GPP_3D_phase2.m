%% 3GPP 36.873 Phase 2 Calibration
%
% This section performs the 3GPP calibration as described in 3GPP TR 36.873 V12.5.0, Section 8.2,
% Page 42 for the phase 2 of the calibration exercise. It is shown how the model is set up to
% obtain the required results, how the output is processed and how the results compare with the 3GPP
% baseline.

%% Antenna setup
% 3GPP uses two antenna configurations for the phase 2 calibration. The first BS array antenna is an
% 2x2 array of vertically polarized patch antennas (0.5 lambda spacing). The second antenna is a
% high-gain panel antenna with 10 coupled elements in elevation and two plus/minus 45 degree
% polarized columns. The electric downtilt is set to 12 degree. Note: The 102 degree electrical tilt
% in Table 8.2-2 refer to spheric coordinates, whereas QuaDRiGa uses geographic coordinates. The
% first MS antenna is an two-element ULA with vertical polarization. The second antenna is an 0/90
% degree cross-polarized array antenna.

clear all
close all
warning('off','all');

s = qd_simulation_parameters;               % Set general simulation parameters
s.center_frequency = 2e9;                   % 2 GHz center frequency

% BS antenna configuration 1
% 2 elements in elevation, 2 elements in azimuth, vertical pol., 0.5 lambda spacing
a_bs_1 = qd_arrayant( '3gpp-3d', 2, 2, s.center_frequency, 1, 0, 0.5 );
a_bs_1.element_position(1,:) = 0.5;         % Distance from pole
a_bs_1.name = 'K=1, M=2';                   % Antenna name

% BS antenna configuration 2
% 10 elements in elevation, 2 element in azimuth, vertical pol., 12 deg downtilt, 0.5 lambda spacing
a_bs_2 = qd_arrayant( '3gpp-3d', 10, 2, s.center_frequency, 6, 12, 0.5 );
a_bs_2.element_position(1,:) = 0.5;         % Distance from pole
a_bs_2.name = 'K=M=10';                     % Antenna name

% MT antenna configuration 1
% 1 element in elevation, 2 elements in azimuth, vertical pol., 0.5 lambda spacing
a_mt_1 = qd_arrayant('omni');
a_mt_1.copy_element(1,2);
a_mt_1.element_position(2,:) = [ -s.wavelength/2 , s.wavelength/2 ]*0.5;

% MT antenna configuration 2
% 1 element in elevation, 2 elements in azimuth, X-pol. 0/90, 0.5 lambda spacing
a_mt_2 = qd_arrayant('omni');
a_mt_2.copy_element(1,2);
a_mt_2.Fa(:,:,2) = 0;
a_mt_2.Fb(:,:,2) = 1;

%% QuaDRiGa Setup
% Here, the channel model is configured. The simulation assumptions are given in Table 8.2-2 in 3GPP
% TR 36.873 V12.5.0. 3GPP specifies to perform simulations for 3D-UMa and 3D-UMi. The scenario
% parameters are given in Table 6.1, page 14. Combined with the two antenna configurations, there
% are four simulation setups. Hence, we define four QuaDRIGa layouts. All 3GPP scenarios define a a
% hexagonal grid with 19 sites and three sectors per site. This is implemented in
% "qd_layout.generate", using the "regular" layout.

tic
no_rx = 2000;                               % Number of MTs (directly scales the simulation time)
create_curves = 1:4;                        % The number of curves to create

s.use_3GPP_baseline = 1;                    % Disable spherical waves
s.show_progress_bars = 0;                   % Enable / disable status display

isd = [ 200, 200, 500, 500 ];                                   % ISD in each layout
no_go_dist = [ 10, 10, 35, 35 ];                                % Min. UE-eNB 2D distance

l(1,1) = qd_layout.generate( 'regular', 19, isd(1), a_bs_1);    % 200 m ISD, K=M=1
l(1,1).simpar = s;                                              % Set simulation parameters
l(1,1).tx_position(3,:) = 10;                                   % 10 m BS height
l(1,1).name = '3D-UMi (K=1,M=2)';

l(1,2) = qd_layout.generate( 'regular', 19, isd(2), a_bs_2);    % 200 m ISD, K=M=10
l(1,2).tx_position(3,:) = 10;                                   % 10 m BS height
l(1,2).simpar = s;                                              % Set simulation parameters
l(1,2).name = '3D-UMi (K=M=10)';

l(1,3) = qd_layout.generate( 'regular', 19, isd(3), a_bs_1);    % 500 m ISD, K=M=1
l(1,3).tx_position(3,:) = 25;                                   % 25 m BS height
l(1,3).simpar = s;                                              % Set simulation parameters
l(1,3).name = '3D-UMa (K=1,M=2)';

l(1,4) = qd_layout.generate( 'regular', 19, isd(4), a_bs_2);    % 500 m ISD, K=M=10
l(1,4).tx_position(3,:) = 25;                                   % 25 m BS height
l(1,4).simpar = s;                                              % Set simulation parameters
l(1,4).name = '3D-UMa (K=M=10)';

% Dorp users in each layout
for il = create_curves
    l(1,il).no_rx = no_rx;                                      % Number of users
    l(1,il).randomize_rx_positions( 0.93*isd(il),1.5,1.5,0, [], no_go_dist(il) );
    
    % Set random height of the users
    floor = randi(5,1,l(1,il).no_rx) + 3;                       % Number of floors in the building
    for n = 1 : l(1,il).no_rx
        floor( n ) =  randi(  floor( n ) );                     % Floor level of the UE
    end
    l(1,il).rx_position(3,:) = 3*(floor-1) + 1.5;               % Height in meters
    
    % Set the scenario and assign LOS probabilities (80% of the users are inddor)
    % "set_scenario" returns an indicator if the user is indoors (1) or outdoors (0)
    switch il
        case {1,2} % UMi
            indoor_rx = l(1,il).set_scenario('3GPP_3D_UMi',[],[],0.8);
            
        case {3,4} % UMa
            indoor_rx = l(1,il).set_scenario('3GPP_3D_UMa',[],[],0.8);
    end
    l(1,il).rx_position(3,~indoor_rx) = 1.5;                    % Set outdoor-users to 1.5 m height
    
    switch il                                                   % Set user antenna
        case {1,3} % ULA
            l(1,il).rx_array = a_mt_1;
        case {2,4} % X-POL
            l(1,il).rx_array = a_mt_2;
    end
end
toc

%% Generate channels
% Channels are now generated using the default QuaDRiGa method (phase 1 only used the LOS path).
% This will take quite some time.

tic                                                     % Time the simulations
clear c
for il = create_curves
    cl = l(1,il).get_channels;                          % Generate channels
    nEl = l(1,il).tx_array(1,1).no_elements / 3;        % Number of elements per sector
    nEl = { 1:nEl , nEl+1:2*nEl , 2*nEl+1:3*nEl  };     % Element indices per sector
    c(:,:,il) = split_tx( cl,nEl );                     % Split channels from each sector
end
toc

%% Coupling Loss
% In the second phase of the calibration, the SSF model is enabled. Hence, all NLOS paths are
% included in the evaluations. For this reason, the coupling loss changes compared to phase 1.
% Multiple paths are now differently weighted by the antenna pattern, depending on the departure
% angles at the BS. The path gain is calculated by averaging the power of all sublinks of the MIMO
% channel matrix. As for phase 1, the coupling loss is the path gain of the serving BS. MTs are
% assigned to BSs based on the maximum path gain value.

calib_3GPP_ref_data;                                                    % Load reference data

legend_names = { l(1,1).name, l(1,2).name, l(1,3).name, l(1,4).name };  % Legend entries
line_col = {'b','r','k','m'};                                           % Color of the lines

set(0,'defaultTextFontSize', 18)                                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                                        % Default Font Size
set(0,'defaultAxesFontName','Times')                                    % Default Font Type
set(0,'defaultTextFontName','Times')                                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')                          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')                              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 6.9])                              % Default Paper Size

pg_eff = zeros( no_rx, 19*3, 4 );       % Calculate the effective path gain from the channels
for il = create_curves
    % Get the number of MIMO sob-channels in the channel matrix
    no_mimo_links = l(1,il).tx_array(1,1).no_elements / 3 * l(1,il).rx_array(1,1).no_elements;
    for ir = 1 : no_rx                                  % Extract effective PG vor each BS-MT link
        for it = 1 : 19*3
            pg_eff( ir,it,il ) = sum( abs(c(ir,it,il).coeff(:)).^2 )/no_mimo_links;
        end
    end
end

coupling_loss = zeros( no_rx, 4 );      % Calculate the coupling loss from the effective PG
for il = create_curves
    coupling_loss(:,il) = 10*log10(max( pg_eff(:,:,il),[],2 ));
end

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = -150; wx = 100; tx = 0.01; ty = 97;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    ln(end+1) = plot( bins, 100*qf.acdf(coupling_loss(:,il),bins),['-',line_col{il}],'Linewidth',2);
    plot( cl36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(coupling_loss(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(cl36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Coupling loss (dB)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%% Wideband SINR
% The wideband SINR is essentially the same as the GF. However, the 3GPP model uses the RSRP values
% for the calculation of this metric. The calculation method is described in 3GPP TR 36.873 V12.5.0
% in Section 8.1 on Page 38. Essentially, the RSRP values describe the average received power (over
% all antenna elements at the receiver) for each transmit antenna port. Hence, in the phase 2
% calibration, there are 4 RSRP values, one for each transmit antenna. The wideband SINR is the GF
% calculated from the first RSRP value, i.e. the average power for the first transmit antenna port.

% Calculate the RSRP value from the first transmit antenna
rsrp_p0 = zeros( no_rx, 19*3, 4 );
for il = create_curves
    for ir = 1 : no_rx
        for it = 1 : 19*3
            tmp = c(ir,it,il).coeff(:,1,:);                 % Coefficients from first Tx antenna
            rsrp_p0( ir,it,il ) = sum( abs( tmp(:) ).^2 ) / 2;      % Divide by 2 Rx antennas
        end
    end
end

% Calculate wideband SINR
sinr = zeros( no_rx, 4 );
for il = create_curves
    sinr(:,il) = 10*log10( max( rsrp_p0(:,:,il),[],2 ) ./ ...
        ( sum( rsrp_p0(:,:,il),2 ) - max( rsrp_p0(:,:,il),[],2 ) ) );
end

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = -10; wx = 40; tx = 0.01; ty = 97;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    ln(end+1) = plot( bins, 100*qf.acdf(sinr(:,il),bins),['-',line_col{il}],'Linewidth',2);
    plot( sinr36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(sinr(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(sinr36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Wideband SINR (dB)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%% Zenith of Departure Spread
% The zenith of departure spread is calculated without the influence of the antenna patterns. Only
% the raw value before weighting the path powers with the antenna gain is used. This is not
% immediately clear from 3GPP TR 36.873, because the calculation method is not specified. However, a
% high gain pattern, such as used for the K=M=10 cases, would significantly decrease the angular
% spread compared to the low-gain patterns (K=1, M=2) since may paths get less power due to the
% weighting with the antenna pattern. Hence, we conclude that the angular spreads are calculated
% without influence of the antenna patterns. Unfortunately, 3GPP also does not define how the
% angular spread is calculated. Here, we extract the angles and the path powers from the QuaDRiGa
% SSF model and calculate the RMS angular spread as
%
% $$\bar{\phi} = \arg\left( \sum_{l=1}^L  P_{l} \cdot \exp\left( j{\phi}_{l} \right) \right)$$
% $$\phi_{l}^{[*]} = \left( \phi_{l} - \bar{\phi} +\pi \mod 2\pi \right)-\pi$$
% $$\sigma_{\phi} = \sqrt{ \frac{1}{P} \cdot \sum_{l=1}^{L} P_{l} \cdot \left(\phi_{l}^{[*]}\right)^2 -
%     \left( \frac{1}{P} \cdot \sum_{l=1}^{L} P_{l} \cdot \phi_{l}^{[*]}\right)^2 }$$
%
% where $\phi_l$ is the raw departure or arrival angle of a path obtained from the model,
% $\bar{\phi}$ is the mean angle of all paths belonging to a CIR, and $\phi_{l}^{[*]}$ is the angle
% where the mean angle is equal to 0 degree. $P_{l}$ is the power of a path, $P$ is the total power
% in the CIR, and $L$ is the number of paths.
%
% To gain some information about the expected values, we can use the formulas in 3GPP TR 36.873,
% page 37. Most of the users are in NLOS conditions and 80 percent of them are situated indoors.
% Simulation results show that the average distance between the MT and the serving BS is 0.65 times
% the ISD. Also, the average height for the indoor users is 9 m. With those values, the expected
% median ZSD for this case are:
%
% $$\mu_{ZSD}(\text{UMa, NLOS, O2I}) = 10^{-2.1(d_{2D}/1000) -0.01(h_{MT}-1.5) + 0.9} \approx 1.4^\circ$$
% $$\mu_{ZSD}(\text{UMi, NLOS, O2I}) = 10^{-2.1(d_{2D}/1000) +0.01\cdot\max(h_{MT}-h_{BS},0) + 0.9} \approx 4.4^\circ$$
%
% Results in the figure show that the median ZOD values for the 3GPP calibration are around 4 degree
% for UMi and 2 degree for UMa. However, QuaDRiGa produces smaller values of 3 degree for UMi and
% 1.7 degree for UMa.

zsd = zeros( no_rx, 4 );
for il = create_curves
    for ir = 1 : no_rx
        [~,it] = max(pg_eff(ir,:,il),[],2);                 % Determine the serving BS
        eod = c(ir,it,il).par.EoD_cb *pi/180;               % EoD angle in [rad]
        pow = c(ir,it,il).par.pow_cb;                       % Normalized power w/o antenna
        zsd(ir,il) = qf.calc_angular_spreads( eod,pow );    % ZSD = ESD in [rad]
    end
end
zsd = zsd * 180 / pi;                                       % Convert to [deg]

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = 0; wx = 50; tx = 0.51; ty = 77;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    ln(end+1) = plot( bins, 100*qf.acdf(zsd(:,il),bins),['-',line_col{il}],'Linewidth',2);
    plot( zsb36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(zsd(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(zsb36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('ZSD (deg)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%% Zenith of Arrival Spread
% The ZSA is calculated in the same way as the ZSD. It is notable here, that the for all O2I
% scenarios, identical values were proposed for the ZSA in 3GPP TR 36.873. The median value is given
% as 10.2 degree. Since 80 percent of the MTs are indoors, the median value should be around 10.2
% for all scenarios and antenna configurations. Surprisingly, the results show differences in the
% median ZSA, depending on the antenna and scenario settings for both, the 3GPP-3D reference curves
% and the QuaDRiGa results. The reason for this is currently subject to speculation. As for the ZSD,
% QuaDRiGa tends to predict slightly lower median values compared to the 3GPP-3D reference.

zsa = zeros( no_rx, 4 );
for il = create_curves
    for ir = 1 : no_rx
        [~,it] = max(pg_eff(ir,:,il),[],2);                 % Determine the serving BS
        eod = c(ir,it,il).par.EoA_cb *pi/180;               % EoD angle in [rad]
        pow = c(ir,it,il).par.pow_cb;                       % Normalized power w/o antenna
        zsa(ir,il) = qf.calc_angular_spreads( eod,pow );    % ZSD = ESD in [rad]
    end
end
zsa = zsa * 180 / pi;                                       % Convert to [deg]

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = 0; wx = 50; tx = 0.51; ty = 77;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    ln(end+1) = plot( bins, 100*qf.acdf(zsa(:,il),bins),['-',line_col{il}],'Linewidth',2);
    plot( zsa36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(zsa(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(zsa36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('ZSA (deg)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%% Largest and smallest singular values
% The singular values of a MIMO channel matrix describe how many parallel spatial data streams can
% be transmitted to one user and what the individual capacity of each streams is. The simulation
% settings propose two settings: One with four vertically polarized antennas at the BS and two
% vertically polarized antennas at the receiver (configuration 1), and one with two cross-polarized
% high-gain an antennas at the BS and an ideal cross-polarized array antenna at the receiver
% (configuration 2). Both configurations result in a 2x4 MIMO channel. Hence, the channel has two
% singular values and supports at most two streams. The 3GPP-3D report does not mention, how the
% singular values are calculated from the channel matrix. It was only discussed internally. The
% method is as follows:
%
% * The results are reported for the channel matrix of the serving BS. The serving BS is determined
%   at the MT by the highest received power of all BS in the layout.
% * The calculations are done in the frequency domain. The bandwidth is set to 10 MHz, which is
%   further split into 50 resource blocks (RBs) of 200 kHz bandwidth, each. Each RB can further be
%   divided inton sub-carriers. However, for the QuaDRiGa results, we only used one subcarrier per
%   RB.
% * The singular values are reported for channels without path-gain, but with antenna patterns
%   included. Hence, one needs to extract the path-gain at the MT position from the channel model
%   and % normalize the channel matrix accordingly, i.e.
%   $$\mathbf{H} = \frac{\mathbf{H}^{[raw]}}{\sqrt{10^{0.1\cdot PG_{dB}}}}$$
% * The ``singular values'' are calculated for each RB by an Eigen-value decomposition of the
%   receive covariance matrix as
%   $$s_{1,2} = \frac{1}{n_{RB}} \cdot \mathrm{eig}\left(\sum_{n=1}^{n_{RB}}
%   \mathbf{H}_n\mathbf{H}_n^H \right)$$
%   for one single carrier, the relationship between the eigenvalues of the covariance matrix and
%   the singular values of the channel matrix is given by
%   $$s_{1,2} = \mathrm{eig}\left( \mathbf{H}_n\mathbf{H}_n^H \right) =
%   \left\{\mathrm{svd}\left(\mathbf{H}\right)\right\}^2$$
% * Results are presented in logarithmic scale, i.e. as $10\cdot\log_{10}(s_{1,2})$.

sv = zeros( 2,50,no_rx,4 );
for il = create_curves
    for ir = 1 : no_rx
        [~,it] = max(pg_eff(ir,:,il),[],2);                 % Determine the serving BS
        
        % Frequency-Domain channel matrix @ 50 RBs, 10 MHz
        H = c(ir,it,il).fr( 10e6, 50 );
        
        % Get the PG without antenna pattern. This is stored in c.par.pg_parset.
        pg = c(ir,it,il).par.pg_parset;                     % in [dB]
        H = H ./ sqrt(10.^(0.1*pg));                        % Normalize channel matrix
        
        for m = 1:size(H,3)
            sv(:,m,ir,il) = svd(H(:,:,m)).^2;
        end  % NOTE: eig( H(:,:,m)*H(:,:,m)' ) == svd(H(:,:,m)).^2
    end
end

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = -5; wx = 40; tx = 0.01; ty = 97;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    sv_max = 10*log10( reshape(sv(1,:,:,il),[],1) );
    ln(end+1) = plot( bins, 100*qf.acdf(sv_max,bins),['-',line_col{il}],'Linewidth',2);
    plot( sv1_36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(sv_max),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(sv1_36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Largest singular value (10log10)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%%
% The generated figure shows the distribution of the largest singular value. For the results with
% co-polar antennas (the blue and black curve), there is an almost perfect match between QuaDRiGa
% and the 3GPP calibration curves. The results for the cross-polar antennas (red and magenta line)
% show some differences. However, the results from individual partners in R1-143469-2014also show a
% significant spread in this case. Median results for the UMi scenario (red curve) ranged from 9 to
% 15 dB. QuaDRiGa predicts 10.6 dB, which is still well within the reported range.


%% Smallest singular value
% The results for the smallest singular value are shown in the following figure. Here, QuaDRiGa
% performs very close to the median results reported in R1-143469-2014.

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = -20; wx = 40; tx = 0.01; ty = 97;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    sv_min = 10*log10( reshape(sv(2,:,:,il),[],1) );
    ln(end+1) = plot( bins, 100*qf.acdf(sv_min,bins),['-',line_col{il}],'Linewidth',2);
    plot( sv2_36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(sv_min),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(sv2_36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Smallest singular value (10log10)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%% Ratio of singular values
% Probably a more important measure than the singular values themselves is the ratio between the
% singular values, which is calculated as $$SR = 10\cdot\log_{10}\left( \frac{s_1}{s_2} \right)$$
% This measure is closely linked to the condition number of the channel matrix $C =
% \sqrt{\frac{s_1}{s_2}}$. The larger this number is, the more difficult it is to invert the matrix
% $\mathbf{H}$. However, inverting this matrix is required in order to separate the two data streams
% at the receiver.

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = 0; wx = 40; tx = 0.51; ty = 37;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    sv_rat = 10*log10( reshape( sv(1,:,:,il) ./ sv(2,:,:,il) ,[],1) );
    ln(end+1) = plot( bins, 100*qf.acdf(sv_rat,bins),['-',line_col{il}],'Linewidth',2);
    plot( svR_36873b(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(sv_rat),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(svR_36873b(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Ratio of singular values (10log10)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%%
% As can be seen, the ratio is much higher for the co-polar antenna configuration (blue and black
% curve). For cross-polar channels, the ratio is about one order of magnitude lower, since an
% additional degree of freedom is provided by the second polarization. Results from QuaDRiGa
% generally agree well. However, there is one exception for the 3D-UMa cross-polar case, where
% QuaDRiGa predicts a SV-ratio of 6.1 dB. The lowest reported value in R1-143469-2014 is 6.3 dB.
