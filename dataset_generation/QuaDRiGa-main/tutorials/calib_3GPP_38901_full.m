%% 3GPP 38.901 Full Calibration
%
% This section performs the 3GPP calibration as described in 3GPP TR 38.901 V14.1.0, Section 7.8.2,
% Page 75 for the full calibration. It is shown how the model is set up to obtain the
% required results, how the output is processed and how the results compare with the 3GPP baseline. 
% The 3GPP calibration reference results were published in the TDOC R1-165975 in August 2016. These
% results were obtained using model parameters from 3GPP TR 38.900 v14.0.0 (2016-06). Unfortunately,
% some parameters were changed in the year following the the publication of the results and
% therefore, different calibration results will be obtained when using the parameters from 38.901
% V14.1.0 which are included in QuaDRiGa.

%% Antenna setup
% 3GPP uses a nested panel antenna. One panel consists of 16 dual-polarized antenna elements (+/- 45
% degree polarization) with 0.5 lambda element spacing. The panel is duplicated along the y-axis. In
% order to reuse the same antenna object for all four frequencies in the simulation, we do not
% specify a carrier frequency. In this case the model assumes that the element positions are given
% in multiples of the wavelength. The method "combine_pattern" calculates the array response with
% respect to the phase-center of the antenna, i.e. the phase in the antenna pattern then includes
% the element positions in the array. The effective element positions are set to 0 and the same same
% antenna can be used for multiple frequencies.

close all
clear all
warning('off','all');

% The mapping function of antenna elements to CRS port (0 degree panning angle)
port_mapping = [ 1,0;0,1 ; 1,0;0,1 ;1,0;0,1 ;1,0;0,1 ]; 
port_mapping = [ port_mapping , zeros( 8,2 ) ; zeros( 8,2 ), port_mapping ] / 2;

% BS antenna configuration 1 (UMa and UMi), 12 degree downtilt
aBS  = qd_arrayant( '3gpp-mmw', 4, 4, [], 6, 12, 0.5, 1, 2, 2.5, 2.5 );
aBS.coupling = port_mapping;                % Assign port mapping
aBS.combine_pattern;                        % Calculate array response
aBS.element_position(1,:) = 0.5;            % Distance from pole in [m]

% BS antenna configuration 1 (Indoor), 20 degree downtilt
aBSi  = qd_arrayant( '3gpp-mmw', 4, 4, [], 6, 20, 0.5, 1, 2, 2.5, 2.5 );
aBSi.coupling = port_mapping;               % Assign port mapping
aBSi.combine_pattern;                       % Calculate array response
aBSi.element_position(1,:) = 0.2;           % Distance from pole in [m]

% BS antenna configuration 2 (UMa, UMi, Indoor)
a2 = qd_arrayant( '3gpp-3d', 2, 2, [], 1, [], 0.5 );
a2.combine_pattern;                         % Calculate array response
a2.element_position(1,:) = 0.5;             % Distance from pole in [m]

append_array( aBS ,a2 );                    % Concatenate arrays for both configurations
append_array( aBSi,a2 );

aMT = qd_arrayant('omni');                  % MT antenna configuration
aMT.copy_element(1,2);
aMT.Fa(:,:,2) = 0;
aMT.Fb(:,:,2) = 1;

%% QuaDRiGa Setup
% Here, the channel model is configured. The simulation assumptions are given in Table 7.8-2 in 3GPP
% TR 38.901 V14.1.0. 3GPP specifies to perform simulations for UMa, UMi and Indoor at four
% frequencies: 6 GHz, 30 GHz, 60 GHz and 70 GHz. Hence, we define three QuaDRIGa layouts. UMa and
% UMi use a hexagonal grid with 19 sites and three sectors per site. The scenario parameters for UMa
% and UMi are given in Table 7.2-1, page 20. This is implemented in "qd_layout.generate", using the
% "regular" layout. The indoor scenario layout is specified in Table 7.2-2 and implemented using the
% "indoor" layout. 3GPP defines two different approaches for the LOS probability for InH users (page
% 27). Here we assume that "open office" should be used.

no_rx = 2000;                               % Number of MTs (directly scales the simulation time)
select_scenario = 1:3;                      % Scenario: 1 = UMi, 2 = UMa, 3 = Indoor
select_fequency = 1:4;                      % Freq.: 1 = 6 GHz, 2 = 30 GHz, 3 = 60 GHz, 4 = 70 GHz

s = qd_simulation_parameters;               % Set general simulation parameters
s.center_frequency = [ 6e9, 30e9, 60e9, 70e9 ];   % Set center frequencies for the simulations
s.center_frequency = s.center_frequency( select_fequency );
no_freq = numel( s.center_frequency );

s.use_3GPP_baseline = 1;                    % Disable spherical waves
s.show_progress_bars = 0;                   % Disable progress bar

isd = [ 200, 500, 20 ];                                         % ISD in each layout
no_go_dist = [ 10, 35, 0 ];                                     % Min. UE-eNB 2D distance

l(1,1) = qd_layout.generate( 'regular', 19, isd(1), aBS);      
l(1,1).simpar = s;                                              % Set simulation parameters
l(1,1).tx_position(3,:) = 10;                                   % 10 m BS height
l(1,1).name = 'UMi';

l(1,2) = qd_layout.generate( 'regular', 19, isd(2), aBS);       
l(1,2).tx_position(3,:) = 25;                                   % 12 m BS height
l(1,2).simpar = s;                                              % Set simulation parameters
l(1,2).name = 'UMa';

l(1,3) = qd_layout.generate( 'indoor', [2,6], isd(3), aBSi, 3, 30);  
l(1,3).tx_position(3,:) = 3;                                    % 3 m BS height
l(1,3).simpar = s;                                              % Set simulation parameters
l(1,3).name = 'InH';

for il = select_scenario                                        % Drop users in each layout
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
    l(1,il).rx_array = aMT;                                     % MT antenna setting
end

%% Generate channels
% The following code generates the channel coefficients. The calibration case assumes that no
% spatial consistency for the SSF is used. Hence, we deactivate the feature by setting the
% decorrelation distance of the SSF parameters "SC_lambda" to 0. The method "split_multi_freq"
% separates the builder objects so that each builder creates channels for only one frequency. If you
% call "split_multi_freq" before any LSF and SSF parameters are generated as it is done the the
% following code, then LSF parameters (e.g. SF, DS, AS) will be uncorrelated for each frequency. If
% you call "gen_lsf_parameters" before "split_multi_freq", then all LSF parameters will be fully
% correlated. However, frequency dependent averages and variances still apply. If you call
% "gen_lsf_parameters" and "gen_ssf_parameters" before "split_multi_freq", then SSF will also
% correlated, i.e. the same paths will be seen at each frequency. Correlated SSF for multi-frequency
% simulations is an additional feature of the 3GPP model (see Section 7.6.5, pp 57 of TR 38.901
% V14.1.0).

tic
clear c
for il = select_scenario
    b = l(1,il).init_builder;                       % Generate builders
    
    sic = size( b );
    for ib = 1 : numel(b)
        [ i1,i2 ] = qf.qind2sub( sic, ib );
        scenpar = b(i1,i2).scenpar;                 % Read scenario parameters
        scenpar.SC_lambda = 0;                      % Disable spatial consistency of SSF
        b(i1,i2).scenpar_nocheck = scenpar;         % Save parameters without check (faster)
    end
    
    b = split_multi_freq( b );                      % Split the builders for multiple frequencies
    gen_parameters( b );                            % Generate LSF and SSF parameters (uncorrelated)
    cm = get_channels( b );                         % Generate channels
    
    cs = split_tx( cm, {1:4,9:12,17:20} );          % Split sectors for Antenna configuration 1
    c{1,il} = qf.reshapeo( cs, [ no_rx, l(1,il).no_tx*3, no_freq ] );
    cs = split_tx( cm, {5:8,13:16,21:24} );         % Split sectors for Antenna configuration 2
    c{2,il} = qf.reshapeo( cs, [ no_rx, l(1,il).no_tx*3, no_freq ] );
end
toc

%% Coupling Loss
% The coupling loss is defined as the path gain of a MT to its serving BS, i.e. the strongest BS
% seen by the MT. The thick lines were obtained using the QuaDRiGa model, the thin dashed line are
% taken from 3GPP R1-165975. They represent the median of all 3GPP calibration results. Results
% agree well for UMi and Indoor Open Office. However, there are some differences in the UMa
% calibration curves. This is due to the fact that the 3GPP reference curves were generated using
% parameters from 3GPP 38.900 v14.0.0 (2016-06). The parameters for UMa-LOS have changed in 3GPP
% 38.901 v14.1.0 (2017-06).

calib_3GPP_ref_data;                                                    % Load reference data

set(0,'defaultTextFontSize', 18)                                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                                        % Default Font Size
set(0,'defaultAxesFontName','Times')                                    % Default Font Type
set(0,'defaultTextFontName','Times')                                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')                          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')                              % Default Paper Type
set(0,'DefaultFigurePaperSize',[16.5 7.3])                              % Default Paper Size

legend_names = { '6 GHz','30 GHz','60 GHz','70 GHz' };
line_col = {'b','r','m','k'};                                           % Color of the lines

for il = select_scenario                                                % Scenario
    figure('Position',[ 50 , 550 , 1400 , 640]);                        % New figure
    pg_eff = []; cl = [];                                               % Clear variables
    if il  < 3; xm = -210; wx = 150; tx = 0.01; ty = 97; end            % UMa and UMi
    if il == 3; xm = -105; wx =  70; tx = 0.01; ty = 97; end            % InH
    for ic = 1:2                                                        % Configuration
        axes('position',[0.06+(ic-1)*0.48 0.12 0.44 0.81]); hold on;    % New sub-figure
        text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');           % Result text
        ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
        for iF = 1 : no_freq                                            % Frequency
            for ir = 1 : no_rx                                          % Receiver
                for it = 1 : size(c{ic,il},2)                           % Calc. path gain
                    pg_eff( it ) = sum(abs(c{ic,il}(ir,it,iF).coeff(:)).^2) / 8;
                end
                cl(ir) = 10*log10(max( pg_eff ));                       % Calc. coupling loss
            end
            iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
            ln(end+1) = plot( bins, 100*qf.acdf(cl,bins),['-',line_col{iFs}],'Linewidth',2);
            plot( cl38900b(iFs,:,il,ic), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
            text( tX,ty,   num2str(median(cl)             ,'%1.1f'),'Color',line_col{iFs});
            text( tX,ty-4, num2str(cl38900b(iFs,10,il,ic) ,'%1.1f'),'Color',line_col{iFs});
        end
        hold off; grid on; box on; set(gca,'YTick',0:10:100);           % Decorations
        set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
        xlabel('Coupling loss (dB)'); title([ l(1,il).name,' - Config ',num2str( ic ) ] );
        if ic==1; ylabel('CDF [%]'); end 
        legend(ln,legend_names(select_fequency),'Location', 'SouthEast');
        text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
    end
end

%% Wide-band SINR
% The wide-band SINR is essentially the same as the GF. However, the 3GPP model uses the RSRP values
% for the calculation of this metric. The calculation method is described in 3GPP TR 36.873 V12.5.0
% in Section 8.1 on Page 38. Essentially, the RSRP values describe the average received power (over
% all antenna elements at the receiver) for each transmit antenna port. Hence, there are 4 RSRP
% values, one for each transmit antenna. The wideband SINR is the GF calculated from the first RSRP
% value, i.e. the average power for the first transmit antenna port. 

for il = select_scenario                                                % Scenario
    figure('Position',[ 50 , 550 , 1400 , 640]);
    rsrp_p0 = []; cl = [];
    if il==1 || il==2;  xm = -10; wx = 40; tx = 0.01; ty = 97; end
    if il==3;           xm = -10; wx = 25; tx = 0.01; ty = 97; end
    for ic = 1 : 2                                                      % Configuration
        axes('position',[0.06+(ic-1)*0.48 0.12 0.44 0.81]); hold on;
        text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
        ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
        for iF = 1 : no_freq                                            % Frequency
            for ir = 1 : no_rx                                          % Calc. coupling loss
                for it = 1 : size(c{ic,il},2)
                    tmp = c{ic,il}(ir,it,iF).coeff(:,1,:);              % Coeff. from first Tx ant.
                    rsrp_p0( it ) = sum(abs( tmp(:) ).^2) / 2;          % Divide by 2 Rx ant.
                end
                sinr(ir) = 10*log10( max(rsrp_p0)/(sum(rsrp_p0)-max(rsrp_p0)) );
            end
            iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
            ln(end+1) = plot( bins, 100*qf.acdf(sinr,bins),['-',line_col{iFs}],'Linewidth',2);
            plot( sinr38900(iFs,:,il,ic), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
            text( tX,ty,   num2str(median(sinr)            ,'%1.1f'),'Color',line_col{iFs});
            text( tX,ty-4, num2str(sinr38900(iFs,10,il,ic) ,'%1.1f'),'Color',line_col{iFs});
        end
        hold off; grid on; box on; set(gca,'YTick',0:10:100);
        set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
        xlabel('Wideband SINR (dB)'); title([ l(1,il).name,' - Config ',num2str( ic ) ] );
        if ic==1; ylabel('CDF [%]'); end
        legend(ln,legend_names(select_fequency),'Location', 'SouthEast');
        text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
    end
end

%% Delay Spread
% The following plots show the delay spread without antenna patterns for the serving BS, i.e. only
% the multi-path components are generated by the SSF model, but the paths are not weights by the
% antenna patterns. For the UMi and UMa scenarios, 80% of the users are indoors. Hence, the results
% are dominated by the O2I parameters, which are not frequency dependent and are identical for LOS
% or NLOS propagation of the outdoor link. The green curve therefore shows the O2I distributions of
% the DS. One can see that the results for UMi and UMa are very similar.

legend_ref = {'O2I only','O2I only','InH LOS'};

ref_O2I = 10.^( randn(1,10000)*0.32-6.62 )*1e9;
mu      = (-7.692  -0.01 *log10(1+s.center_frequency'/1e9));
ref_InH = 10.^( 0.18*randn(no_freq,10000) + mu * ones(1,10000) )*1e9;
for il = select_scenario                                                % Scenario
    pg_eff = []; ds = []; ds_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.41; ty = 47;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = 0; wx = 1000; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = 0; wx = 1000; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = 0; wx = 80; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        for ir = 1 : no_rx                                              % Calc. coupling loss
            for it = 1 : size(c{1,il},2)
                pg_eff( it ) = sum(abs(c{1,il}(ir,it,iF).coeff(:)).^2) / 8;
                ds_tmp( it ) = c{1,il}(ir,it,iF).par.ds_cb;
            end
            [~,ii] = max( pg_eff ); ds(ir) = ds_tmp(ii)*1e9;
        end
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(ds,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( ds38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(ds)         ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(ds38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
        if il==3;plot(bins,100*qf.acdf(ref_InH(iF,:),bins),'-.','Color',line_col{iFs});end
    end
    if il<3; ln(end+1)=plot(bins,100*qf.acdf(ref_O2I,bins),'-.','Color',[0 .5 0],'Linewidth',2);end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('Delay Spread (nsec)'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,{legend_names{select_fequency},legend_ref{il}},'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end

%%
% For the InH case, parameters changed in TR 38.901. The original parameterization in 3GPP TR 38.900
% V14.0.0 included some significant dependence of the STD of the DS on the carrier frequency. This
% was removed in TR 38.901. In addition, due to the open office LOS probabilities and the UE
% attachment to the strongest BS, there are 98% of the users in LOS conditions. Hence, the results
% heavily depend on the LOS DS, which is shown for the four frequencies are dashed-dotted thin
% lines. One can observe, that the DS values for the serving BS are always smaller compared to the
% expected values from the LOS distributions. This comes from the negative correlation of the DS
% with the SF (-0.8 for InH-LOS). If the link has a high SF, it also has a low DS. However, if the
% SF is high, the BS gets selected for the UE attachment. As a result, DS values for the serving BS
% are always smaller compared to the average LOS-DS from all BS.

%% Azimuth Angle Spread of Departure
% The next plot shows the ASD. The same assumptions as for the DS apply, i.e. no antenna patterns,
% UE attachment to the strongest BS, O2I-dominance for UMa and UMi and LOS dominance for InH.
% Results agree well for UMa and UMi.

ref_O2I = 10.^( randn(1,10000)*0.42 +1.25 );
ref_InH = 10.^( randn(1,10000)*0.18 +1.60 );
for il = select_scenario                                                % Scenario
    pg_eff = []; asd = []; asd_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.01; ty = 97;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = 0; wx = 80; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = 0; wx = 80; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = 0; wx = 80; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        for ir = 1 : no_rx                                              % Calc. coupling loss
            for it = 1 : size(c{1,il},2)
                pg_eff( it ) = sum(abs(c{1,il}(ir,it,iF).coeff(:)).^2) / 8;
                asd_tmp( it ) = c{1,il}(ir,it,iF).par.asD_cb;
            end
            [~,ii] = max( pg_eff ); asd(ir) = asd_tmp(ii);
        end
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(asd,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( asd38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(asd)         ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(asd38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
    end
    if il<3; ln(end+1)=plot(bins,100*qf.acdf(ref_O2I,bins),'-.','Color',[0 .5 0],'Linewidth',2);end
    if il==3;ln(end+1)=plot(bins,100*qf.acdf(ref_InH,bins),'-.','Color',[0 .5 0],'Linewidth',2);end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('ASD (degrees)'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,{legend_names{select_fequency},legend_ref{il}},'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end

%%
% For the InH results, the green curve shows the expected results when only taking the parameters
% for the LOS-ASD into account. One can see that the results obtained from the model are much lower.
% This can have two reasons. First, the ASD is negatively correlated with the SF (-0.4). Hence, BSs
% with a high SF are more likely to become the serving BSs which leads to decreased ASD values for
% the serving link. Second, the maximum achievable angular spread depends on the KF. The average KF
% in the indoor LOS scenario is 7 dB. In this case, the maximal achievable azimuth spread is around
% 50 degree. However, the positive correlation between SF and KF (+0.5) leads to increased KF values
% for the serving link. As a result, the median ASD for the serving link gets reduced to roughly 30
% degree compared to the 40 degree that would be expected from the InH-LOS parameters.

%% Elevation / Zenith Angle Spread of Departure
% The next plot shows the ESD / ZSD. The same assumptions as for the DS and ASD apply, i.e. no
% antenna patterns, UE attachment to the strongest BS, O2I-dominance for UMa and UMi and LOS
% dominance for InH. Results agree well for UMi and UMa as well as for the higher Frequencies for
% InH. 

mu  = (2.228 - 1.43 *log10(1+s.center_frequency'/1e9));
sig = (0.3   + 0.13 *log10(1+s.center_frequency'/1e9));
ref_InH = 10.^( sig * randn(1,10000) + mu * ones(1,10000) );
for il = select_scenario                                                % Scenario
    pg_eff = []; esd = []; esd_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.41; ty = 37;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = 0; wx = 6; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = 0; wx = 15; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = 0; wx = 30; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        for ir = 1 : no_rx                                              % Calc. coupling loss
            for it = 1 : size(c{1,il},2)
                pg_eff( it ) = sum(abs(c{1,il}(ir,it,iF).coeff(:)).^2) / 8;
                esd_tmp( it ) = c{1,il}(ir,it,iF).par.esD_cb;
            end
            [~,ii] = max( pg_eff ); esd(ir) = esd_tmp(ii);
        end
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(esd,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( zsd38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(esd)         ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(zsd38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
        if il==3;plot(bins,100*qf.acdf(ref_InH(iF,:),bins),'-.','Color',line_col{iFs});end
    end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('ZSD (degrees)'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,legend_names(select_fequency),'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end

%% 
% InH at 6 GHz shows some differences for the same reason, the ASD was smaller then expected.
% The ZSD at lower frequencies can have values above 30 degrees. However, with a KF of 7 dB, the
% maximum achievable ZSD is around 30 degree. Due to the correlation between SF and KF, the serving
% link gets even higher KF values and, as a consequence, lower angular spreads.

%% Azimuth Angle Spread of Arrival
% The next plot shows the ASA. The same assumptions as for the DS apply, i.e. no antenna patterns,
% UE attachment to the strongest BS, O2I-dominance for UMa and UMi and LOS dominance for InH.
% Results agree well for UMa and UMi. For InH, the ASA parameters changed from TR 38.900 to 38.901.
% Hence, different results are obtained at the output of the model compared to the 3GPP calibration
% reference. At 6 GHz, where the largest ASAs are achieved for InH-LOS, the upper limit for the
% angle spread is reached due to the correlations of ASA vs. SF (-0.5) and SF vs. KF (+0.5).

ref_O2I = 10.^( randn(1,10000)*0.16 +1.76 );
mu  = (1.781 - 0.19 *log10(1+s.center_frequency'/1e9));
sig = (0.119 + 0.12 *log10(1+s.center_frequency'/1e9));
ref_InH = 10.^( sig * randn(1,10000) + mu * ones(1,10000) );
for il = select_scenario                                                % Scenario
    pg_eff = []; asa = []; asa_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.01; ty = 97;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = 0; wx = 100; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = 0; wx = 100; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = 0; wx = 80; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        for ir = 1 : no_rx                                              % Calc. coupling loss
            for it = 1 : size(c{1,il},2)
                pg_eff( it ) = sum(abs(c{1,il}(ir,it,iF).coeff(:)).^2) / 8;
                asa_tmp( it ) = c{1,il}(ir,it,iF).par.asA_parset;
            end
            [~,ii] = max( pg_eff ); asa(ir) = asa_tmp(ii);
        end
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(asa,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( asa38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(asa)         ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(asa38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
        if il==3;plot(bins,100*qf.acdf(ref_InH(iF,:),bins),'-.','Color',line_col{iFs});end
    end
    if il<3; ln(end+1)=plot(bins,100*qf.acdf(ref_O2I,bins),'-.','Color',[0 .5 0],'Linewidth',2);end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('ASA (degrees)'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,{legend_names{select_fequency},legend_ref{il}},'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end

%% Elevation / Zenith Angle Spread of Arrival
% The next plot shows the ESD / ZSD results. Again, UMi and UMa results agree well since mostly, O2I
% parameters apply (green curve). For InH, the 3GPP parameters changed from TR 38.900 to 38.901.
% Hence, different results are obtained at the output of the model compared to the 3GPP calibration
% reference.

ref_O2I = 10.^( randn(1,10000)*0.43 +1.01 );
mu  = (1.44  - 0.26 *log10(1+s.center_frequency'/1e9));
sig = (0.264 - 0.04 *log10(1+s.center_frequency'/1e9));
ref_InH = 10.^( sig * randn(1,10000) + mu * ones(1,10000) );
for il = select_scenario                                                % Scenario
    pg_eff = []; esa = []; esa_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.01; ty = 97;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = 0; wx = 50; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = 0; wx = 50; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = 0; wx = 25; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        for ir = 1 : no_rx                                              % Calc. coupling loss
            for it = 1 : size(c{1,il},2)
                pg_eff( it ) = sum(abs(c{1,il}(ir,it,iF).coeff(:)).^2) / 8;
                esa_tmp( it ) = c{1,il}(ir,it,iF).par.esA_cb;
            end
            [~,ii] = max( pg_eff ); esa(ir) = esa_tmp(ii);
        end
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(esa,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( zsa38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(esa)         ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(zsa38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
        if il==3;plot(bins,100*qf.acdf(ref_InH(iF,:),bins),'-.','Color',line_col{iFs});end
    end
    if il<3; ln(end+1)=plot(bins,100*qf.acdf(ref_O2I,bins),'-.','Color',[0 .5 0],'Linewidth',2);end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('ZSA (degrees)'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
     legend(ln,{legend_names{select_fequency},legend_ref{il}},'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end

%% First Singular Value (Configuration 2)
% The singular values of a MIMO channel matrix describe how many parallel spatial data streams can
% be transmitted to one user and what the individual capacity of each streams is. The antenna
% configuration results in a 2x4 MIMO channel. Hence, the channel has two singular values and
% supports at most two streams. The singular values are calculated as follows:
%
% * The results are reported for the channel matrix of the serving BS. 
% * The calculations are done in the frequency domain. The bandwidth is set to 20 MHz (at 6 GHz) or
%   100 MHz (above 6 GHz), which is further split into 50 resource blocks (RBs). 
% * The singular values are reported for channels without path-gain, but with antenna patterns
%   included. 
% * The singular values are calculated for each RB by an Eigenvalue decomposition of the
%   receive covariance matrix as
%   $$s_{1,2} = \frac{1}{n_{RB}} \cdot \mathrm{eig}\left(\sum_{n=1}^{n_{RB}}
%   \mathbf{H}_n\mathbf{H}_n^H \right)$$
% * Results are presented in logarithmic scale, i.e. as $10\cdot\log_{10}(s_{1,2})$.
%
% Results for the first singular value agree perfectly for UMi and UMa. Only minor differences can
% be seen for InH. 

clear sv                                                        % Calculate singular values
BW = [20,100,100,100]*1e6;                                      % Bandwidth for each 
for il = select_scenario
    sv{il} = zeros( 2,50,no_rx,4 );
    pg_eff = [];
    for iF = 1 : no_freq
        for ir = 1 : no_rx
            for it = 1 : size(c{2,il},2)
                pg_eff( it ) = sum(abs(c{2,il}(ir,it,iF).coeff(:)).^2) / 8;
            end
            [~,it] = max( pg_eff );                             % Select serving BS
            H  = c{2,il}(ir,it,iF).fr( BW(iF), 50 );
            pg = c{2,il}(ir,it,iF).par.pg_parset;
            H = H ./ sqrt(10.^(0.1*pg));                        % Normalize channel matrix
            for m = 1:size(H,3)
                sv{il}(:,m,ir,iF) = svd(H(:,:,m)).^2;
            end  % NOTE: eig( H(:,:,m)*H(:,:,m)' ) == svd(H(:,:,m)).^2
        end
    end
end

for il = select_scenario                                                % Scenario
    pg_eff = []; esa = []; esa_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.01; ty = 97;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = -10; wx = 30; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = -10; wx = 30; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = -10; wx = 30; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        sv_max = 10*log10( reshape(sv{il}(1,:,:,iF),[],1) );
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(sv_max,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( sv1_38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(sv_max)      ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(sv1_38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
    end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('10log10( 1st singular value )'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,legend_names(select_fequency),'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end


%% Second Singular Value (Configuration 2)
% Results are slightly larger for UMi and UMa indicating a slightly higher channel capacity as
% reported by the average 3GPP results. However, the results presented here are still well within
% the range of the results reported by different partners in R1-165975. For InH, the larger
% differences are probably due to the changed parameterization TR 38.901.

for il = select_scenario                                                % Scenario
    pg_eff = []; esa = []; esa_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.01; ty = 97;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = -30; wx = 40; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = -30; wx = 40; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = -50; wx = 50; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        sv_min = 10*log10( reshape(sv{il}(2,:,:,iF),[],1) );
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(sv_min,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( sv2_38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(sv_min)      ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(sv2_38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
    end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('10log10( 2nd singular value )'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,legend_names(select_fequency),'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end


%% Ratio of Singular Values (Configuration 2)
% Probably a more important measure than the singular values themselves is the ratio between the
% singular values, which is calculated as $$SR = 10\cdot\log_{10}\left( \frac{s_1}{s_2} \right)$$
% This measure is closely linked to the condition number of the channel matrix $C =
% \sqrt{\frac{s_1}{s_2}}$. The larger this number is, the more difficult it is to invert the matrix
% $\mathbf{H}$. However, inverting this matrix is required in order to separate the two data streams
% at the receiver. Due to the larger second SV our results are better (lower values are better) that
% the 3GPP baseline for UMa and UMi. The InH cannot be discussed properly due to the changed
% parameterization TR 38.901.

for il = select_scenario                                                % Scenario
    pg_eff = []; esa = []; esa_tmp = [];
    if il == 1 || il == 3; figure('Position',[ 50 , 550 , 1400 , 640]); end; tx = 0.01; ty = 97;
    if il == 1; axes('position',[0.06 0.12 0.44 0.81]); hold on; xm = 0; wx = 40; end
    if il == 2; axes('position',[0.54 0.12 0.44 0.81]); hold on; xm = 0; wx = 40; end
    if il == 3; axes('position',[0.30 0.12 0.44 0.81]); hold on; xm = 0; wx = 60; end
    text( tx*wx+xm,ty,'QD.'); text( tx*wx+xm,ty-4,'3GP');
    ln = []; bins = (-0.1:0.005:1.1)*wx+xm;
    for iF = 1 : no_freq                                                % Frequency
        sv_rat = 10*log10( reshape( sv{il}(1,:,:,iF)./sv{il}(2,:,:,iF) ,[],1) );
        iFs = select_fequency(iF); tX = (tx+0.12*iF)*wx+xm;
        ln(end+1) = plot( bins, 100*qf.acdf(sv_rat,bins),['-',line_col{iFs}],'Linewidth',2);
        plot( svR_38900(iFs,:,il), 5:5:95,['+--',line_col{iFs}],'Linewidth',1 )
        text( tX,ty,   num2str(median(sv_rat)      ,'%1.1f'),'Color',line_col{iFs});
        text( tX,ty-4, num2str(svR_38900(iFs,10,il) ,'%1.1f'),'Color',line_col{iFs});
    end
    hold off; grid on; box on; set(gca,'YTick',0:10:100);
    set(gca,'XTick',xm : wx/10 : xm+wx); axis([xm xm+wx 0 100]);
    xlabel('10log10( ratio singular values )'); title([ l(1,il).name ] );
    if il==1 || il==3; ylabel('CDF [%]'); end
    legend(ln,legend_names(select_fequency),'Location', 'SouthEast');
    text( 0.01*wx+xm, 3, ['v.',qd_simulation_parameters.version] );
end
