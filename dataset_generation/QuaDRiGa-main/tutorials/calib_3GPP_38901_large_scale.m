%% 3GPP 38.901 Large Scale Calibration
%
% This section performs the 3GPP calibration as described in 3GPP TR 38.901 V14.1.0, Section 7.8.1,
% Page 74 for the large scale calibration. It is shown how the model is set up to obtain the
% required results, how the output is processed and how the results compare with the 3GPP baseline.
% The purpose of this calibration is to show the correct working of the path-loss models, the
% antenna model, the user placement in 3D coordinates.

%% Antenna setup
% 3GPP uses a high-gain panel antenna with 10 coupled elements in elevation and 12 degree electric
% down-tilt for UMa and UMi scenarios. Indoor scenarios use 20 degree downtilt. Note: The 102 or
% 110 degree electrical tilt in Table 7.8-1 refer to spheric coordinates, whereas QuaDRiGa uses
% geographic coordinates.

close all
clear all
warning('off','all');

% Antenna configuration 1 (UMa and UMi)
% 10 elements in elevation, 1 element in azimuth, vertical pol., 12 deg downtilt, 0.5 lambda spacing
a1 = qd_arrayant( '3gpp-3d', 10, 1, [], 4, 12, 0.5 );
a1.element_position(1,:) = 0.5;             % Distance from pole in [m]

% Antenna configuration 1 (Indoor)
% 10 elements in elevation, 1 element in azimuth, vertical pol., 20 deg downtilt, 0.5 lambda spacing
a2 = qd_arrayant( '3gpp-3d', 10, 1, [], 4, 20, 0.5 );
a2.element_position(1,:) = 0.2;             % Distance from pole in [m]

%% QuaDRiGa Setup
% Here, the channel model is configured. The simulation assumptions are given in Table 7.8.1 in 3GPP
% TR 38.901 V14.1.0. 3GPP specifies to perform simulations for UMa, UMi and Indoor at 3 frequencies:
% 6 GHz, 30 GHz and 70 GHz. The scenario parameters for UMa and UMi are given in Table 7.2-1, page
% 20. Hence, we define three QuaDRIGa layouts. UMa and UMi use a hexagonal grid with 19 sites and
% three sectors per site. This is implemented in "qd_layout.generate", using the "regular" layout.
% The indoor scenario layout is speciefied in Table 7.2-2.

no_rx = 2000;                               % Number of MTs (directly scales the simulation time)
select_scenario = 2;                      % Scenario: 1 = UMi, 2 = UMa, 3 = Indoor
select_fequency = 2;                      % Frequency: 1 = 6 GHz, 2 = 30 GHz, 4 = 70 GHz

s = qd_simulation_parameters;               % Set general simulation parameters
s.center_frequency = [ 6e9, 30e9, 70e9 ];   % Set center frequencies for the simulations
s.center_frequency = s.center_frequency( select_fequency );
no_freq = numel( s.center_frequency );

s.use_3GPP_baseline = 1;                    % Disable spherical waves
s.show_progress_bars = 0;                   % Disable progress bar

isd = [ 200, 500, 20 ];                                         % ISD in each layout
no_go_dist = [ 10, 35, 0 ];                                     % Min. UE-eNB 2D distance

l(1,1) = qd_layout.generate( 'regular', 19, isd(1), a1);        
l(1,1).simpar = s;                                              % Set simulation parameters
l(1,1).tx_position(3,:) = 10;                                   % 10 m BS height
l(1,1).name = 'UMi';

l(1,2) = qd_layout.generate( 'regular', 19, isd(2), a1);     
l(1,2).tx_position(3,:) = 25;                                   % 25 m BS height
l(1,2).simpar = s;                                              % Set simulation parameters
l(1,2).name = 'UMa';

l(1,3) = qd_layout.generate( 'indoor', [2,6], isd(3), a2, 3, 30);   
l(1,3).tx_position(3,:) = 3;                                    % 3 m BS height
l(1,3).simpar = s;                                              % Set simulation parameters
l(1,3).name = 'Indoor Open Office';

for il = select_scenario                                        % Dorp users in each layout
    l(1,il).no_rx = no_rx;                                      % Number of users
    if il == 3
        ind = true( 1,no_rx );                                  % Indoor placement
        while any( ind )
            l(1,il).randomize_rx_positions( sqrt(60^2+25^2), 1, 1, 0, ind );
            ind = abs( l(1,il).rx_position(1,:) ) > 60 | abs( l(1,il).rx_position(2,:) ) > 25;
        end
    else
        ind = true( 1,no_rx );                                  % UMa / UMi placement
        while any( ind )
            l(1,il).randomize_rx_positions( 0.93*isd(il), 1.5, 1.5, 0, ind );
            ind = sqrt(l(1,il).rx_position(1,:).^2 + l(1,il).rx_position(2,:).^2) < no_go_dist(il);
        end
        floor = randi(5,1,l(1,il).no_rx) + 3;                   % Number of floors in the building
        for n = 1 : l(1,il).no_rx
            floor( n ) =  randi(  floor( n ) );                 % Floor level of the UE
        end
        l(1,il).rx_position(3,:) = 3*(floor-1) + 1.5;           % Height in meters
    end
    switch il   % Set the scenario and assign LOS probabilities (80% of the users are indoor)
        case 1
            indoor_rx = l(1,il).set_scenario('3GPP_38.901_UMi',[],[],0.8);
            l(1,il).rx_position(3,~indoor_rx) = 1.5;            % Set outdoor-users to 1.5 m height
        case 2
            indoor_rx = l(1,il).set_scenario('3GPP_38.901_UMa',[],[],0.8);
            l(1,il).rx_position(3,~indoor_rx) = 1.5;            % Set outdoor-users to 1.5 m height
        case 3
            l(1,il).set_scenario('3GPP_38.901_Indoor_Open_Office');
    end
    l(1,il).rx_array = qd_arrayant('omni');                	    % Omni-Antenna, vertically polarized
end

%% Generate channels
% The following code generates the channel coefficients. However, by default, QuaDRiGa always uses
% the full small-scale-fading model as well as spatial consistency. These two features are disabled
% by setting the number of paths to 1 and the decorrelation distance for the SSF (SC_lambda) to 0 m.

tic
pg_eff = cell( 1,3 );
for il = select_scenario
    pg_eff{il} = zeros( no_rx , l(1,il).no_tx*3 , no_freq );
    b = l(1,il).init_builder;                       % Generate builders
    
    sic = size( b );
    for ib = 1 : numel(b)
        [ i1,i2 ] = qf.qind2sub( sic, ib );
        scenpar = b(i1,i2).scenpar;                 % Read scenario parameters
        scenpar.NumClusters = 1;                    % Only LOS path, disable SSF model
        scenpar.SC_lambda = 0;                      % Disable spatial consistency of SSF
        b(i1,i2).scenpar_nocheck = scenpar;         % Save parameters without check (faster)
    end
    
    b = split_multi_freq( b );                      % Split the builders for multiple frequencies
    gen_parameters( b );                            % Generate LSF (SF) and SSF (LOS path only)
    cm = get_channels( b );                         % Generate channels
    cm = split_tx( cm, {1,2,3} );                   % Split sectors
    cm = qf.reshapeo( cm, [ no_rx, l(1,il).no_tx*3, no_freq ] );
    
    for ir = 1 : no_rx                              % Extract effective PG vor each BS-MT link
        for it = 1 : l(1,il).no_tx*3
            for iF = 1 : no_freq
                pg_eff{il}( ir,it,iF ) = abs( cm( ir,it,iF ).coeff ).^2;
            end
        end
    end
end
toc

%% Coupling Loss
% The coupling loss is defined as the path gain of a MT to its serving BS, i.e. the strongest BS
% seen by the MT. The thick lines were obtained using the QuaDRiGa model, the thin dashed line are
% taken from 3GPP R1-165974. They represent the median of all 3GPP calibration results. Results
% agree well for UMi and Indoor Open Office. However, there are some significant differences in the
% UMa calibration curves. This is probably due to the fact that the original calibration was done
% using the parameters from 3GPP 38.900 v14.0.0 (2016-06). The parameters for UMa-LOS have changed
% in 3GPP 38.901 v14.1.0 (2017-06).

calib_3GPP_ref_data;                                                    % Load reference data

set(0,'defaultTextFontSize', 18)                                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                                        % Default Font Size
set(0,'defaultAxesFontName','Times')                                    % Default Font Type
set(0,'defaultTextFontName','Times')                                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')                          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')                              % Default Paper Type
set(0,'DefaultFigurePaperSize',[16.5 7.3])                              % Default Paper Size

legend_names = { '6 GHz','30 GHz','70 GHz' };
line_col = {'b','r','k'};                                               % Color of the lines

figure('Position',[ 50 , 550 , 1400 , 640]);
for il = select_scenario
    cl = zeros( no_rx, 3 );             % Calculate the coupling loss from the effective PG
    for iF = 1 : no_freq
        cl(:,iF) = 10*log10(max( pg_eff{il}(:,:,iF),[],2 ));
    end
    if il == 3
        figure('Position',[ 50 , 550 , 1400 , 640]);
        axes('position',[0.3, 0.12 0.44 0.81]); hold on;
        xm = -105; wx = 70; tx = 0.01; ty = 97;
    else
        xm = -210; wx = 150; tx = 0.01; ty = 97;
        axes('position',[0.06+(il-1)*0.48 0.12 0.44 0.81]); hold on;
    end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq
        iFs = select_fequency(iF);
        ln(end+1) = plot( bins, 100*qf.acdf(cl(:,iF),bins),['-',line_col{iFs}],'Linewidth',2);
        plot( cl38900a(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text((tx+0.12*iF)*wx+xm,ty,num2str(median(cl(:,iF)),'%1.1f'),'Color',line_col{iFs});
        text((tx+0.12*iF)*wx+xm,ty-4,num2str(cl38900a(iFs,10,il),'%1.1f'),'Color',line_col{iFs});
    end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('Coupling loss (dB)')
    if il==1 || il==3; ylabel('CDF [%]'); end;
    title(l(1,il).name)
    legend(ln,legend_names(select_fequency),'Location', 'SouthEast')
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end

%% Geometry Factor
% The GF is a lower bound for the actual SINR. It is defined as the power ratio of the serving BS
% and the sum power of all interfering BSs. The results in  the following Figure agree well with the
% 3GPP calibrations curves.

figure('Position',[ 50 , 550 , 1400 , 640]);
for il = select_scenario
    gf = zeros( no_rx, 3 );
    for iF = 1 : no_freq
        gf(:,iF) = 10*log10( max( pg_eff{il}(:,:,iF),[],2 ) ./ ( sum( pg_eff{il}(:,:,iF),2 ) -...
            max( pg_eff{il}(:,:,iF),[],2 ) ) );
    end
    if il == 3
        figure('Position',[ 50 , 550 , 1400 , 640]);
        axes('position',[0.3, 0.12 0.44 0.81]); hold on;
    else
        axes('position',[0.06+(il-1)*0.48 0.12 0.44 0.81]); hold on;
    end
    xm = -10; wx = 40; tx = 0.01; ty = 97;
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1:no_freq
        iFs = select_fequency(iF);
        ln(end+1) = plot( bins, 100*qf.acdf(gf(:,iF),bins),['-',line_col{iFs}],'Linewidth',2);
        plot( gf38900a(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text((tx+0.1*iF)*wx+xm,ty,num2str(median(gf(:,iF)),'%1.1f'),'Color',line_col{iFs});
        text((tx+0.1*iF)*wx+xm,ty-4,num2str(gf38900a(iFs,10,il),'%1.1f'),'Color',line_col{iFs});
    end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('Geometry (dB)')
    if il==1 || il==3; ylabel('CDF [%]'); end;
    title(l(1,il).name)
    legend(ln,legend_names(select_fequency),'Location', 'SouthEast')
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end
