%% 3GPP 36.873 Phase 1 Calibration
%
% This section performs the 3GPP calibration as described in 3GPP TR 36.873 V12.5.0, Section 8.2,
% Page 39 for the phase 1 of the calibration exercise. It is shown how the model is set up to
% obtain the required results, how the output is processed and how the results compare with the 3GPP
% baseline. The purpose of the phase 1 calibration is to show the correct working of the path-loss
% models, the antenna model, the user placement in 3D coordinates.

%% Antenna setup
% The antenna model consists of a 2D planar array structure with M rows and N columns of patch
% elements. Each element has an azimuth and elevation FWHM of 65 degree. The elements can either be
% vertically polarized or cross-polarizes with plus/minus 45 degree polarization. In the latter, the
% number of antenna ports is doubled. Optionally, vertically stacked elements can be coupled using
% fixed complex-valued weights. In order to reduce computational complexity, effective antenna
% patterns are calculated in QuaDRiGa that include the coupling and downtilt settings.
%
% 3GPP uses two antenna configurations for the phase 1 calibration. The first defines a high-gain
% panel antenna with 10 coupled elements in elevation and 12 degree electric down-tilt. Note: The
% 102 degree electrical tilt in Table 8.2-1 refer to spheric coordinates, whereas QuaDRiGa uses
% geographic coordinates. The second antenna is a patch antenna. Both are defined in 3GPP TR 36.873,
% Section 7.1, Page 17 and implemented "qd_arrayant.generate".

clear all
close all
warning('off','all');

s = qd_simulation_parameters;               % Set general simulation parameters
s.center_frequency = 2e9;                   % 2 GHz center frequency
s.show_progress_bars = 0;                   % Disable progress bars

% Antenna configuration 1
% 10 elements in elevation, 1 element in azimuth, vertical pol., 12 deg downtilt, 0.5 lambda spacing
a1 = qd_arrayant( '3gpp-3d', 10, 1, s.center_frequency, 4, 12, 0.5 );
a1.element_position(1,:) = 0.5;             % Distance from pole
a1.name = 'K=M=10';                         % Antenna name

% Antenna configuration 2
% 1 element in elevation, 1 element in azimuth, vertical pol.
a2 = qd_arrayant( '3gpp-3d', 1, 1, s.center_frequency, 1, 0, 0.5 );
a2.element_position(1,:) = 0.5;             % Distance from pole
a2.name = 'K=M=1';                          % Antenna name


%% QuaDRiGa Setup
% Here, the channel model is configured. The simulation assumptions are given in Table 8.2-1 in 3GPP
% TR 36.873 V12.5.0. 3GPP specifies to perform simulations for 3D-UMa and 3D-UMi. The scenario
% parameters are given in Table 6.1, page 14. Combined with the two antenna configurations, there
% are four simulation setups. Hence, we define 4 QuaDRIGa layouts. All 3GPP scenarios define a a
% hexagonal grid with 19 sites and three sectors per site. This is implemented in
% "qd_layout.generate", using the "regular" layout.

tic
no_rx = 2000;                               % Number of MTs (directly scales the simulation time)
create_curves = 1:4;                        % The number of curves to create

s.use_3GPP_baseline = 1;                    % Disable spherical waves and geometric polarization

isd = [ 200, 200, 500, 500 ];                               % ISD in each layout
no_go_dist = [ 10, 10, 35, 35 ];                            % Min. UE-eNB 2D distance

l(1,1) = qd_layout.generate( 'regular', 19, isd(1), a2);    % 200 m ISD, K=M=1
l(1,1).simpar = s;                                          % Set simulation parameters
l(1,1).tx_position(3,:) = 10;                               % 10 m BS height
l(1,1).name = '3D-UMi (K=M=1)';

l(1,2) = qd_layout.generate( 'regular', 19, isd(2), a1);    % 200 m ISD, K=M=10
l(1,2).tx_position(3,:) = 10;                               % 10 m BS height
l(1,2).simpar = s;                                          % Set simulation parameters
l(1,2).name = '3D-UMi (K=M=10)';

l(1,3) = qd_layout.generate( 'regular', 19, isd(3), a2);    % 500 m ISD, K=M=1
l(1,3).tx_position(3,:) = 25;                               % 25 m BS height
l(1,3).simpar = s;                                          % Set simulation parameters
l(1,3).name = '3D-UMa (K=M=1)';

l(1,4) = qd_layout.generate( 'regular', 19, isd(4), a1);    % 500 m ISD, K=M=10
l(1,4).tx_position(3,:) = 25;                               % 25 m BS height
l(1,4).simpar = s;                                          % Set simulation parameters
l(1,4).name = '3D-UMa (K=M=10)';

% Drop users in each layout
for il = create_curves
    l(1,il).no_rx = no_rx;                                  % Number of users
    l(1,il).randomize_rx_positions( 0.93*isd(il), 1.5, 1.5, 0, [], no_go_dist(il) );
    
    % Set random height of the users
    floor = randi(5,1,l(1,il).no_rx) + 3;                   % Number of floors in the building
    for n = 1 : l(1,il).no_rx
        floor( n ) =  randi(  floor( n ) );                 % Floor level of the UE
    end
    l(1,il).rx_position(3,:) = 3*(floor-1) + 1.5;           % Height in meters
    
    % Set the scenario and assign LOS probabilities (80% of the users are inddor)
    % "set_scenario" returns an indicator if the user is indoors (1) or outdoors (0)
    switch il
        case {1,2} % UMi
            indoor_rx = l(1,il).set_scenario('3GPP_3D_UMi',[],[],0.8);
        case {3,4} % UMa
            indoor_rx = l(1,il).set_scenario('3GPP_3D_UMa',[],[],0.8);
    end
    l(1,il).rx_position(3,~indoor_rx) = 1.5;             	% Set outdoor-users to 1.5 m height
    
    % Set user antenna
    l(1,il).rx_array = qd_arrayant('omni');                	% Omni-Antenna, vertically polarized
end
toc

%% Generate channels
% Now, the required metric are generated by the model. The MT is always connected to the strongest
% serving BS. The coupling loss describes the received power to this BS relative to 0 dBm transmit
% power. Only the LOS path is considered. Other metrics are the geometry factor (GF) and the zenith
% angle at the BS.

tic
pg_eff =zeros( no_rx, 19*3, 4 );                            % Effective PG for each MT and BS
zod    = zeros( no_rx*19, 4 );                              % Zenith angles for each MT and BS site
for il = create_curves
    coeff  = zeros( no_rx * 19 , 3 );                       % Raw channel coefficients
    name   = cell( no_rx * 19, 1  );                        % Name in the form "Tx_Rx"
    
    b = l(1,il).init_builder;                               % Initialze channel builder objects
    init_sos( b );                                          % Initialize random generators
    gen_lsf_parameters( b );                                % Generat shadow fading
    cf = get_los_channels( b );                             % Get the LOS channel coefficients only
    
    cnt = 1;                                                % Counter
    sic = size(b);
    for i_cb = 1 : numel(b)
        [ i1,i2 ] = qf.qind2sub( sic, i_cb );
        tx_name = ['Tx',num2str(i2,'%02d')];                % Tx name, e.g. "Tx01"
        
        if b(i1,i2).no_rx_positions > 1
            tmp = b(i1,i2).get_angles;                      % 3D angles btween BS and MT
            zod( cnt : cnt+b(i1,i2).no_rx_positions-1,il) = 90-tmp(3,:);
        end
        
        for i_mt = 1 : b(i1,i2).no_rx_positions
            rx_name = b( i1,i2 ).rx_track(1,i_mt).name;     % Rx name, e.g. "Rx0001"
            name{ cnt }  = [tx_name,'_',rx_name];           % Link name, e.g. "Tx01_Rx0001"
            coeff(cnt,:) = cf(i1,i2).coeff(1,:,1,i_mt);     % Channel coefficients
            cnt = cnt + 1;                                  % Increase counter
        end
    end
    
    [~,ii] = sort( name );                                  % Get the correct order of the channels
    zod(:,il) = zod(ii,il);                                 % Sort ZODs by name
    
    tmp = reshape( coeff(ii,:), no_rx, 19, 3 );             % Split the 3 sectors from each BS site
    tmp = permute( tmp, [1,3,2] );                          % Reorder the channels
    pg_eff(:,:,il) = reshape( tmp, no_rx, [] );
end
pg_eff = abs( pg_eff ).^2;                                  % Amplitude --> Power
zod = reshape( zod, no_rx, 19, 4 );
toc

%% Coupling Loss
% The coupling loss is defined as the path gain of a MT to its serving BS, i.e. the strongest BS
% seen by the MT. Here, the term BS refers to one sector of a 3-sector site. In the proposed layout,
% there are 19 sites, each consisting of three BSs. MTs were placed in the first ring of
% interferers, i.e. around the first site. The phase 1 calibration does not consider a SSF model,
% but includes the antenna patterns. Hence, the results shown in the following figure were obtained
% by running the simulations with only one path (the LOS path). The thick lines were obtained using
% the QuaDRiGa model, the thin dashed line are taken from 3GPP 36.873. They represent the median of
% all 3GPP calibration results. The QuaDRiGa results fit almost perfectly. The remaining differences
% are well within the tolerances visible in the individual result curves.

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

% Calculate the coupling loss from the effective PG
coupling_loss = zeros( no_rx, 4 );
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
    plot( cl36873a(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(coupling_loss(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(cl36873a(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Coupling loss (dB)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );


%% Geometry Factor
% The GF is a lower bound for the actual SINR. It is defined as the power ratio of the serving BS
% and the sum power of all interfering BSs. The results in  the following Figure agree well with the
% 3GPP calibrations curves.

% Calculate the GF
gf = zeros( no_rx, 4 );
for il = create_curves
    gf(:,il) = 10*log10( max( pg_eff(:,:,il),[],2 ) ./ ( sum( pg_eff(:,:,il),2 ) -...
        max( pg_eff(:,:,il),[],2 ) ) );
end

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = -10; wx = 40; tx = 0.01; ty = 97;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    ln(end+1) = plot( bins, 100*qf.acdf(gf(:,il),bins),['-',line_col{il}],'Linewidth',2);
    plot( gf36873a(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(gf(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(gf36873a(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('Geometry (dB)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );

%% Evaluation: Zenith of Departure Angle
% The ZoD is calculated from the LOS path between the serving BS and the MT position. The values in
% the follwing Figure prove that the model is a 3D model. Users are placed on different floors and
% the serving BS is determined based on the power of the LOS path. Note that this power value
% changes when different antenna patterns are used. Hence, the assignment of MTs to BSs is
% different, depending on which antennas are used at the BS, which explains why the curves differ
% from each other. The results obtained from QuaDRiGa agree almost perfectly with the 3GPP
% calibration curves (tolerances are within 0.1 degree).

% Determine the serving site
zod_serving = zeros( no_rx, 4 );
for il = create_curves
    [~,serving] = max(pg_eff(:,:,il),[],2);
    serving = ceil( serving / 3 - 0.1 );
    for ir = 1 : no_rx
        zod_serving( ir,il ) = zod( ir, serving( ir ),il );
    end
end

figure('Position',[ 50 , 550 , 950 , 600]);
axes('position',[0.09 0.12 0.88 0.86]); hold on;

xm = 70; wx = 40; tx = 0.01; ty = 97;
text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
ln = []; bins = (-0.1:0.01:1.1)*wx+xm;
for il = create_curves
    ln(end+1) = plot( bins, 100*qf.acdf(zod_serving(:,il),bins),['-',line_col{il}],'Linewidth',2);
    plot( zod36873a(il,:), 0:5:100,['--',line_col{il}],'Linewidth',1 )
    text((tx+0.1*il)*wx+xm,ty,num2str(median(zod_serving(:,il)),'%1.1f'),'Color',line_col{il});
    text((tx+0.1*il)*wx+xm,ty-4,num2str(zod36873a(il,11),'%1.1f'),'Color',line_col{il});
end

hold off; grid on; box on;
set(gca,'YTick',0:10:100); set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
xlabel('LOS ZOD (deg)')
ylabel('CDF [%]')
legend(ln,legend_names( create_curves ),'Location', 'SouthEast')
text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
