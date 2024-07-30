%% 3GPP 38.821 NTN Calibration
%
% In this section, the model calibration is done for the 3GPP NTN specifications as described in
% 3GPP TR 38.821 v16.0.0, Section 6. All essential components for the non-terrestrial networks are
% implemented in QuaDRiGa. It is shown how the model is set up to obtain the required results, how
% the output is processed and how the results compare with the 3GPP baseline.
%
% This section is structured as follows: First, the basic system parameters are configured. This
% includes the satellite parameters, the antennas, the positions of the satellite in orbit and the
% multi-beam layout. Then, a list of study cases is set up and the corresponding QuaDRiGa layouts
% are configured. Lastly, the simulations are executed and evaluated.

%% Satellite parameters and antennas
% Two sets of satellite parameters are considered as the baseline for system level simulator
% calibration. The corresponding parameters are given in Tables 6.1.1.1-1 (Set-1 satellite
% parameters) and 6.1.1.1-2 (Set-2 satellite parameters) of 3GPP TR 38.821. The tables define the
% payload characteristics for downlink (DL) and uplink (UL) transmissions. Two scenarios are
% considered for the calibration: one in the S-band (2 GHz) for direct access to a mobile terminal,
% and one in the KA-band for fixed access (DL at 20 GHz, UL at 30 GHz). There are 12 options for the
% satellite antenna, all of which are parabolic reflector antennas. Here, the antenna
% characteristics are defined. This includes the carrier frequency, the reflector radius and the
% antenna gain. The properties of the three UE antennas are given in Table 6.1.1.1-3.

satant = qd_arrayant([]);
satant(1,1)  = qd_arrayant('parabolic', 11.0,  2e9,[],5,1,[], 51.0 );   % Set1, GSO,  S, UL/DL
satant(1,2)  = qd_arrayant('parabolic',  1.0,  2e9,[],5,1,[], 30.0 );   % Set1, LEO,  S, UL/DL
satant(1,3)  = qd_arrayant('parabolic',  2.5, 20e9,[],5,1,[], 58.5 );   % Set1, GSO, KA, DL
satant(1,4)  = qd_arrayant('parabolic', 0.25, 20e9,[],5,1,[], 38.5 );   % Set1, LEO, KA, DL
satant(1,5)  = qd_arrayant('parabolic', 1.67, 30e9,[],5,1,[], 58.5 );   % Set1, GSO, KA, UL
satant(1,6)  = qd_arrayant('parabolic',0.167, 30e9,[],5,1,[], 38.5 );   % Set1, LEO, KA, UL

satant(1,7)  = qd_arrayant('parabolic',  6.0,  2e9,[],5,1,[], 45.5 );   % Set2, GSO,  S, UL/DL
satant(1,8)  = qd_arrayant('parabolic',  0.5,  2e9,[],5,1,[], 24.0 );   % Set2, LEO,  S, UL/DL
satant(1,9)  = qd_arrayant('parabolic',  1.0, 20e9,[],5,1,[], 50.5 );   % Set2, GSO, KA, DL
satant(1,10) = qd_arrayant('parabolic',  0.1, 20e9,[],5,1,[], 30.5 );   % Set2, LEO, KA, DL
satant(1,11) = qd_arrayant('parabolic', 0.67, 30e9,[],5,1,[], 50.5 );   % Set2, GSO, KA, UL
satant(1,12) = qd_arrayant('parabolic',0.065, 30e9,[],5,1,[], 30.5 );   % Set2, LEO, KA, UL

ueant(1,1) = qd_arrayant('parabolic',  0.6, 20e9,[],5,1,[], 39.7 );     % VSAT, KA, DL
ueant(1,2) = qd_arrayant('parabolic',  0.6, 30e9,[],5,1,[], 43.2 );     % VSAT, KA, UL
ueant(1,3) = qd_arrayant('omni');                                       % Handheld, S, UL/DL
ueant(1,3).copy_element(1,2);
ueant(1,3).rotate_pattern(45,'x',1);
ueant(1,3).rotate_pattern(-45,'x',2);
ueant(1,3).rotate_pattern(90,'z');
ueant(1,3).center_frequency = 2e9;


%% Satellite orbital positions
% Compared to 3GPP TR 38.821, QuaDRiGa provides a more detailed satellite orbital model which allows
% precise orbit tracking of multiple satellites. However, for the calibration, fixed satellite
% positions are sufficient. A geostationary satellite (GSO) is set at a elevation angle target of 45
% degree and two low-earth-orbit (LEO) satellites are defined at 90 degree elevation angle target,
% one at 600 km height and one at 1200 km height. QuaDRiGa requires a reference position on Earth
% for which the simulations are done. In order to achieve the 45 degree elevation target for the GSO
% satellite, the latitude must be set to 38.85 degree. With -5 degree longitude, the reference
% position lies in central Spain. The same reference position is chosen for the LEO satellites,
% where the 90 degree elevation angle target is achieved by adjusting the satellite orbit such that
% the satellite is placed directly above that reference position. Furthermore, it is necessary to
% adjust the satellite orientation such that the antenna is pointed towards the reference position
% on Earth. This is done using the "calc_orientation" method of the "qd_track"-class. The GSO
% antenna must face north at a 45 degree down-tilt angle, whereas the LEO antenna must be tilted
% down by 90 degrees. 

sat.GSO = qd_satellite( 'gso', 1, -5);                	% Satellite orbital position
sat.GSO = sat.GSO.init_tracks( [-5,38.85] );           	% UE reference position on Earth
sat.GSO.calc_orientation( [], -45*pi/180, 90*pi/180);  	% Satellite antenna orientation
sat.GSO.name = 'GSO';

sat.LEO600 = qd_satellite( 'custom', qd_satellite.R_e + 600, 0, 63.4, -28.8, 44.55, 0);
sat.LEO600 = sat.LEO600.init_tracks( [-5,38.85] );   	% UE reference position on Earth
sat.LEO600.calc_orientation( [], -90*pi/180 );        	% Satellite antenna orientation
sat.LEO600.name = 'LEO600';

sat.LEO1200 = qd_satellite( 'custom', qd_satellite.R_e + 1200, 0, 63.4, -28.8, 44.55, 0);
sat.LEO1200 = sat.LEO1200.init_tracks( [-5,38.85] );  	% UE reference position on Earth
sat.LEO1200.calc_orientation( [], -90*pi/180 );        	% Satellite antenna orientation
sat.LEO1200.name = 'LEO1200';


%% Multibeam layout
% Table 6.1.1.1-4 of 3GPP TR 38.821 defines the beam layout for single satellite simulations. There
% are 19 serving beams, i.e. 18 beams surrounding a central beam, allocated on 2 distinct "tiers".
% In addition, up to 4 additional tiers of beams cause inter-beam interference (see Figure
% 6.1.1.1-1). For the metrics statistic (e.g. coupling loss, geometry), only the UEs placed in the
% inner-19 beams are considered. The adjacent beam spacing is based on the 3 dB beam-width
% parameters which can be calculated from the parabolic antennas. Here, we define the beam-layout
% and the frequency-reuse pattern.
% 
% The frequency-reuse pattern is defined in Table 6.1.1.1-5. There are three options: FRF1 allocates
% the whole system bandwidth in each beam, which will result in significant interference at the
% edges of the beams. FRF3 uses only 1/3 of the available spectrum in each beam, but interference is
% reduced significantly. FRF4 includes polarization-reuse (LHCP or RHCP beams). Hence, 1/2 of the
% total spectrum is available in each beam and interference is reduced at the same time. However,
% UEs need to have dual-polarized antennas to separate the satellite signals. Also, multi-path
% propagation and improper antenna alignment lead to cross-talk between the polarization
% multiplexed signals, which reduces the effectiveness of polarization-reuse in S-band.  
% 
% The following algorithm calculates the beam offsets and the frequency-reuse indicator for each
% beam within six tiers (127 beams). The indicator is an integer number ranging from 1 to 4. The
% FRF3 beams can have numbers between 1 and 3 which are allocated such that neighboring beams cannot
% have the same frequency. FRF4 beams are set such that odd numbers (1 or 3) are RHCP and even
% numbers (2 and 4) are for LHCP beams. 

Pb     = zeros( 1,127 );                                % Maximum number of beams is 127 (6 tiers)
Pb(2)  = 1j;                                            % Second beam is at x = 0, y = 1
FRF1    = ones( 1,127 );                                % Frequency reuse 1 index
FRF3    = FRF1;                                         % Initialize FRF3 index                       
FRF3(2) = 2;                                            % Second beam frequency index for FRF3
FRF4    = FRF3;                                         % Initialize FRF4 index
d0     = 90;                                            % Walking direction from B1 to B2 = 90 deg
for n = 3 : 127                                         % Loop for remaining beams
    dn = 120;                                           % Change direction by 120 deg clock-wise
    while dn > -0.5
        Pn = Pb(n-1) + exp(1j*(d0-dn)*pi/180);          % Create new beam offset
        if all( abs(Pb-Pn) > 0.1 )                      % Beam does not exist yet
            Pb(n) = Pn;                                 % Add beam to list
            d0 = mod( d0-dn,360 );                      % Update moving direction
            NG = find( abs(Pb(1:n-1)-Pb(n))< 1.1 );     % Find all neighbors with dstance 1
            FRF3(n) = setdiff( 1:3, FRF3(NG) );         % Neigbors must have different frequencies
            FRF4n = setdiff( 1:4, FRF4(NG) );           % Get candidates for FRF4 frequencies
            if numel( FRF4n ) == 1                      % Only one FRF4 candidate (3 neighbors)
                FRF4(n) = FRF4n;
            elseif abs(d0-330)<1 || abs(d0-150)<1       % Keep parity
                FRF4(n) = FRF4n( mod(FRF4n,2) == mod(FRF4(n-1),2) );
            elseif abs(d0-30)<1 || abs(d0-210)<1        % Use first value
                FRF4(n) = FRF4n(1);
            else                                        % Change parity
                FRF4(n) = FRF4n( mod(FRF4n,2) ~= mod(FRF4(n-1),2) );
            end
            dn = -1;                                    % End while loop and find next beam
        else                                            % Beam exists already
            dn = dn - 60;                               % Change direction by 60 deg and try again
        end
    end
end


%% List of calibration study cases
% The list of calibration study cases is given by Table 6.1.1.1-9. There are 30 cases, each defined
% by the satellite orbit (GSO, LEO600, or LEO1200), the parameter set (set-1 or set-2), the
% frequency band (S-band or KA-band), and the frequency/polarization reuse option. In addition,
% there are differences for the UL and DL configurations. The following data structure maps the
% calibration cases to simulations by selecting one of the three orbits, one of the 12 satellite
% antennas, one of the three UE antennas and a reuse option for each case. In addition, the
% adjacent-beam spacing must be given in degree and the total transmit power must be given in dBm. 
% The noise figure (NF) describes how the receiver noise influences the SNR. It is given in dB
% relative to the thermal noise floor, which is calculated based on the Boltzmann constant of
% -228.6 dBW/K/Hz and an ambient temperature of 290 K. In the KA-band uplink, a different antenna
% temperature of 150 K is assumed, leading to an improved noise figure. Contrary to 3GPP TR 38.821,
% we do not use the antenna-gain-to-noise-temperature (G/T) metric here, but include this effect in
% a different noise figure. The bandwidth allocation is done as in 3GPP TR 38.821 Table 6.1.3.3-1.


% ABS      = 0.5 * sqrt(3)* satant(1, 3).calc_beamwidth(1)
% PTx_DL   = dBW/MHZ + 30 + 10*log10(MHZ) - Gain
% NF_UL_KA = Nf + 10*log10(T0+(Ta-T0)*10^(-0.1*Nf) ) - 10*log10(T0)
%          = 1.2 + 10*log10(290+(150-290)*10^(-0.1*1.2) ) - 10*log10(290) = -0.8 dB

sc={};                                                      % Init. data structure (cell array)
noise_thermal = -228.6 + 10*log10(290) + 30;                % Thermal noise in [dBm/Hz]

% Set 1 DL         Orbit, SAT, UE,FRF,    ABS,  PTx,   NF,  BW
sc(end+1,:) = {    'GSO',   3,  1,  1, 0.1527, 37.5,  1.2, 400   };
sc(end+1,:) = {    'GSO',   3,  1,  2, 0.1527, 37.5,  1.2, 133.3 };
sc(end+1,:) = {    'GSO',   3,  1,  3, 0.1527, 37.5,  1.2, 200   };
sc(end+1,:) = {    'GSO',   1,  3,  1, 0.3470, 52.8,  7.0,  30   };
sc(end+1,:) = {    'GSO',   1,  3,  2, 0.3470, 52.8,  7.0,  10   };
sc(end+1,:) = { 'LEO600',   4,  1,  1, 1.5260, 21.5,  1.2, 400   };
sc(end+1,:) = { 'LEO600',   4,  1,  2, 1.5260, 21.5,  1.2, 133.3 };
sc(end+1,:) = { 'LEO600',   4,  1,  3, 1.5260, 21.5,  1.2, 200   };
sc(end+1,:) = { 'LEO600',   2,  3,  1, 3.8174, 48.8,  7.0,  30   };
sc(end+1,:) = { 'LEO600',   2,  3,  2, 3.8174, 48.8,  7.0,  10   };
sc(end+1,:) = {'LEO1200',   4,  1,  1, 1.5260, 27.5,  1.2, 400   };
sc(end+1,:) = {'LEO1200',   4,  1,  2, 1.5260, 27.5,  1.2, 133.3 };
sc(end+1,:) = {'LEO1200',   4,  1,  3, 1.5260, 27.5,  1.2, 200   };
sc(end+1,:) = {'LEO1200',   2,  3,  1, 3.8174, 54.8,  7.0,  30   };
sc(end+1,:) = {'LEO1200',   2,  3,  2, 3.8174, 54.8,  7.0,  10   };

% Set 2 DL         Orbit, SAT, UE,FRF,    ABS,  PTx,   NF,  BW
sc(end+1,:) = {    'GSO',   9,  1,  1, 0.3817, 37.5,  1.2, 400   };
sc(end+1,:) = {    'GSO',   9,  1,  2, 0.3817, 37.5,  1.2, 133.3 };
sc(end+1,:) = {    'GSO',   9,  1,  3, 0.3817, 37.5,  1.2, 200   };
sc(end+1,:) = {    'GSO',   7,  3,  1, 0.6357, 52.8,  7.0,  30   };
sc(end+1,:) = {    'GSO',   7,  3,  2, 0.6357, 52.8,  7.0,  10   };
sc(end+1,:) = { 'LEO600',  10,  1,  1, 3.8174, 21.5,  1.2, 400   };
sc(end+1,:) = { 'LEO600',  10,  1,  2, 3.8174, 21.5,  1.2, 133.3 };
sc(end+1,:) = { 'LEO600',  10,  1,  3, 3.8174, 21.5,  1.2, 200   };
sc(end+1,:) = { 'LEO600',   8,  3,  1, 7.6397, 48.8,  7.0,  30   };
sc(end+1,:) = { 'LEO600',   8,  3,  2, 7.6397, 48.8,  7.0,  10   };
sc(end+1,:) = {'LEO1200',  10,  1,  1, 7.6397, 27.5,  1.2, 400   };
sc(end+1,:) = {'LEO1200',  10,  1,  2, 3.8174, 27.5,  1.2, 133.3 };
sc(end+1,:) = {'LEO1200',  10,  1,  3, 3.8174, 27.5,  1.2, 200   };
sc(end+1,:) = {'LEO1200',   8,  3,  1, 7.6397, 54.8,  7.0,  30   };
sc(end+1,:) = {'LEO1200',   8,  3,  2, 7.6397, 54.8,  7.0,  10   };

% Set 1 UL         Orbit, SAT, UE,FRF,    ABS,  PTx,   NF,  BW
sc(end+1,:) = {    'GSO',   5,  2,  1, 0.1524, 33.0, -0.8, 400   };
sc(end+1,:) = {    'GSO',   5,  2,  2, 0.1524, 33.0, -0.8, 133.3 };
sc(end+1,:) = {    'GSO',   5,  2,  3, 0.1527, 33.0, -0.8, 200   };
sc(end+1,:) = {    'GSO',   1,  3,  1, 0.3470, 23.0,  7.0,   0.4 };
sc(end+1,:) = {    'GSO',   1,  3,  2, 0.3470, 23.0,  7.0,   0.4 };
sc(end+1,:) = { 'LEO600',   6,  2,  1, 1.5260, 33.0, -0.8, 400   };
sc(end+1,:) = { 'LEO600',   6,  2,  2, 1.5260, 33.0, -0.8, 133   };
sc(end+1,:) = { 'LEO600',   6,  2,  3, 1.5260, 33.0, -0.8, 200   };
sc(end+1,:) = { 'LEO600',   2,  3,  1, 3.8174, 23.0,  7.0,   0.4 };
sc(end+1,:) = { 'LEO600',   2,  3,  2, 3.8174, 23.0,  7.0,   0.4 };
sc(end+1,:) = {'LEO1200',   6,  1,  1, 1.5260, 33.0, -0.8, 400   };
sc(end+1,:) = {'LEO1200',   6,  1,  2, 1.5260, 33.0, -0.8, 133.3 };
sc(end+1,:) = {'LEO1200',   6,  1,  3, 1.5260, 33.0, -0.8, 200   };
sc(end+1,:) = {'LEO1200',   2,  3,  1, 3.8174, 23.0,  7.0,   0.4 };
sc(end+1,:) = {'LEO1200',   2,  3,  2, 3.8174, 23.0,  7.0,   0.4 };

% Set 2 UL         Orbit, SAT, UE,FRF,    ABS,  PTx,   NF,  BW
sc(end+1,:) = {    'GSO',  11,  2,  1, 0.3817, 33.0, -0.8, 400   };
sc(end+1,:) = {    'GSO',  11,  2,  2, 0.3817, 33.0, -0.8, 133.3 };
sc(end+1,:) = {    'GSO',  11,  2,  3, 0.3817, 33.0, -0.8, 200   };
sc(end+1,:) = {    'GSO',   7,  3,  1, 0.6357, 23.0,  7.0,   0.4 };
sc(end+1,:) = {    'GSO',   7,  3,  2, 0.6357, 23.0,  7.0,   0.4 };
sc(end+1,:) = { 'LEO600',  12,  2,  1, 3.8174, 33.0, -0.8, 400   };
sc(end+1,:) = { 'LEO600',  12,  2,  2, 3.8174, 33.0, -0.8, 133.3 };
sc(end+1,:) = { 'LEO600',  12,  2,  3, 3.8174, 33.0, -0.8, 200   };
sc(end+1,:) = { 'LEO600',   8,  3,  1, 7.6397, 23.0,  7.0,   0.4 };
sc(end+1,:) = { 'LEO600',   8,  3,  2, 7.6397, 23.0,  7.0,   0.4 };
sc(end+1,:) = {'LEO1200',  12,  2,  1, 7.6397, 33.0, -0.8, 400   };
sc(end+1,:) = {'LEO1200',  12,  2,  2, 3.8174, 33.0, -0.8, 133.3 };
sc(end+1,:) = {'LEO1200',  12,  2,  3, 3.8174, 33.0, -0.8, 200   };
sc(end+1,:) = {'LEO1200',   8,  3,  1, 7.6397, 23.0,  7.0,   0.4 };
sc(end+1,:) = {'LEO1200',   8,  3,  2, 7.6397, 23.0,  7.0,   0.4 };


%% Layout setup
% A simulation layout is created for each simulation case. 

l = qd_layout;                                              % Initialize layout variable
for isc = 1 : size(sc,1)
    l(1,isc) = qd_layout;                                   % Create new QuaDRiGa layout
    l(1,isc).simpar(1,1).center_frequency = ...
        satant(1,sc{isc,2}).center_frequency;               % Set Frequency
    l(1,isc).simpar(1,1).show_progress_bars = 0;            % Disable Progress bars
    
    if sc{isc,4} == 1                                       % Set number of beams
        l(1,isc).no_tx = 61;                                % FRF1: Two tiers of interfering beams
    else
        l(1,isc).no_tx = 127;                               % FRF3/4: Four tiers of interfering beams
    end
    
    for itx = 1 : l(1,isc).no_tx                                        % Set satellite position
        l(1,isc).tx_track(1,itx) = copy( sat.(sc{isc,1}) );             % Copy satellite track obj.
        l(1,isc).tx_track(1,itx).name = ['B',num2str(itx,'%03d')];      % Set beam ID
        l(1,isc).tx_track(1,itx).calc_orientation( [],...               % Apply beam offset
            imag(Pb(itx))*sc{isc,5}*pi/180, -real(Pb(itx))*sc{isc,5}*pi/180 );
    end
end


%% UE setup
% The calibration requires that each of the 19 serving beams gets assigned a fixed number of UEs.
% This is done by dropping 100 random UEs in the coverage area of the inner 19 beams, calculating
% the strongest (serving) beam for each UE and assigning users to their serving beams until the
% beams have reached their set number of UEs. 
%
% Then, we assign the antennas to the beams and the UEs. For FRF4, a distinction must be made for the
% RHCP and LHCP beams. In the KA-band, the UEs use directional parabolic antennas. Those antennas
% must be pointed towards the satellite in order to allow communication. We read the satellite and
% the UE positions from the qd_layout object, calculate the ideal pointing angle and apply this
% orientation to the UE.  

no_ms_per_beam = 10;                                        % Set number of terminals per beam

beam_separation = zeros(size(sc,1),1);                      % Initialize beam separation
for isc = 1 : size(sc,1)
    
    % UE placement
    ls = copy(l(1,isc));                                    % Temporary copy of the layout
    ls.no_tx = 37;                                          % One tier of interferers
    ls.tx_array = sub_array( satant(1,sc{isc,2}),1 );       % LHCP polarization
    beam_separation(isc) = tand( sc{isc,5} ) * sqrt(sum(ls.tx_position(:,1).^2));
    [~,gain] = calc_gain( ls.tx_array(1,1) );               % Sat antenna gain
     
    rx_assigned = zeros(19,no_ms_per_beam);                 % Init. list of assigned UEs
    rx_pos = zeros( 3,numel( rx_assigned ));                % List of rx positions
    while any(rx_assigned(:) == 0) 
        ls.no_rx = 100;                                     % Drop 100 random UEs
        ls.randomize_rx_positions(beam_separation(isc)*3.1,1.5,1.5,0);
        ls.set_scenario('LOSonly');                         % Consider only antenna (no path-gain)
        [~,pow] = ls.set_pairing('power',0);                % Calculate RX power
        for irx = 1 : ls.no_rx                              % Do for each UE
            [pm,ib] = max(pow(irx,:));                      % Find serving beam
            if ib < 19.5 && any( rx_assigned(ib,:) == 0 ) && gain-pm < 12   % Inner 2 tiers ?
                ii = find(rx_pos(3,:) == 0,1);              % Check if beam is already full
                rx_pos(:,ii) = ls.rx_position(:,irx);       % Assign UE
                rx_assigned( ib, find(rx_assigned(ib,:)==0,1) ) = ii;
            end
        end
    end
    ii = reshape(rx_assigned',1,[]);                        % Order UEs by beam assignment
    l(1,isc).rx_position = rx_pos(:,ii);                    % Store list of UEs
    l(1,isc).set_scenario('QuaDRiGa_NTN_Rural_LOS');        % Set propagation conditions

    % Antenna assignment
    if sc{isc,4} ~= 3                                       % FRF1 and FRF3 use single polarization
        l(1,isc).tx_array = sub_array( satant(1,sc{isc,2}),1 );
        l(1,isc).rx_array = sub_array(  ueant(1,sc{isc,3}),1 );
    else                                                    % FRF4 uses dual polarization
        satant1 = sub_array( satant(1,sc{isc,2}),1 );       % RHCP @ satellite
        satant2 = sub_array( satant(1,sc{isc,2}),2 );       % LHCP @ satellite
        ueant1  = sub_array(  ueant(1,sc{isc,3}),1 );       % RHCP @ UE
        ueant2  = sub_array(  ueant(1,sc{isc,3}),2 );       % LHCP @ UE
        for itx = 1 : l(1,isc).no_tx
            if mod( FRF4(itx), 2) == 1                      % FRF4 index is an odd number
                l(1,isc).tx_array(1,itx) = satant1;         % ... use RHCP
            else                                            % FRF4 index is an even number
                l(1,isc).tx_array(1,itx) = satant2;         % ... use LHCP
            end
        end
        for irx = 1 : l(1,isc).no_rx                        % Set the corresponding UE antennas
            if mod( FRF4( ceil((irx-0.5)/no_ms_per_beam) ), 2 ) == 1
                l(1,isc).rx_array(1,irx) = ueant1;
            else
                l(1,isc).rx_array(1,irx) = ueant2;
            end
        end
    end

    % UE antenna orientation
    if sc{isc,3} == 1 || sc{isc,3} == 2                     % Reflector antenna in KA band
        tx_pos = l(1,isc).tx_track(1,1).initial_position;  	% Get satellite position
        orientation = zeros(3,1);                          	% Init. oientation vector
        for ir = 1 : l(1,isc).no_rx
            rx_pos = l(1,isc).rx_track(1,ir).initial_position; % Get UE position
            rt = tx_pos - rx_pos;                           % Calculate pointing vector
            rt = rt / norm(rt);                            	% Normalize to unit length
            orientation(2) = asin( rt(3) );               	% Calculate UE tilt angle
            orientation(3) = atan2( rt(2),rt(1) );        	% Calculate UE heading angle
            l(1,isc).rx_track(1,ir).orientation = orientation; % Set antenna orientation
        end
    end
end


%% Show layout plots
% Here, we visualize the layout for the GSO and the LEO600 cases. The method "qd_layout.power_map"
% projects the beams onto a 2D map, including the antenna gains and the path gain. Hence, the
% variable "map" in the following code block contains the received power for the three tiers of
% beams relative to 0 dBm transmit power. We then calculate the geometry factor (GF), i.e. the ratio
% of the strongest (serving) beam to the interfering beams plus noise. The generated plots show
% the positions of the terminals and the GF.

close all                                                   % Close all open figures
set(0,'defaultTextFontSize', 18)                            % Default font size
set(0,'defaultAxesFontSize', 18)                            % Default font Size
set(0,'defaultAxesFontName','Times')                        % Default font type
set(0,'defaultTextFontName','Times')                        % Default font type
set(0,'defaultFigurePaperPositionMode','auto')              % Default plot position
set(0,'DefaultFigurePaperType','<custom>')                  % Default paper type
set(0,'DefaultFigurePaperSize',[14.5 7.7])                  % Default paper size

show_id = [1,6];                                            % Select cases to plot
for iid = 1 : numel(show_id)
    isc = show_id(iid);
    
    if sc{isc,4} == 1; FRF = FRF1;                          % Select FRF1
    elseif sc{isc,4} == 2; FRF = FRF3;                      % Select FRF3
    else; FRF = FRF4;                                       % Select FRF4
    end
    
    ls = copy( l(1,isc) );                                  % Copy layout to local variable
    ls.no_tx = 37;                                          % Only show 3 tiers of beams

    dst = beam_separation(isc)/18;                          % Map resolution (distance btw. pixels)
    cov = beam_separation(isc)*5;                           % Area of interest
    [ map,x_coords,y_coords ] = ls.power_map( '5G-ALLSTAR_Rural_LOS',...
        'quick',dst,-cov,cov,-cov,cov,1.5 );                % Generate coverage maps for each beam
    
    Ns   = noise_thermal + 10*log10(sc{isc,8}*1e6) + sc{isc,7} - sc{isc,6}; % Noise
    P    = cat(3,map{FRF(1:ls.no_tx)==1});                  % Path gain map for each beam
    Pmax = max(P,[],3);                                     % Serving beam power for each map pixel
    GF   = Pmax ./ ( sum(P,3) - Pmax + 10.^(Ns/10) );       % Geometry factor for each pixel
    GF   = 10*log10(GF);                                    % Logarithmic scale
    
    ls.visualize([],[],0);                                  % Show Sat and MT positions on the map
    hold on
    imagesc( x_coords, y_coords, GF );                      % Plot the Geometry Factor
    hold off
    axis([-1 1 -1 1]*cov)                                   % Adjust map size to coverage area
    caxis( max(ceil(GF(:)))+[-25 0])                        % Adjust colormap range
    colmap = colormap;
    colormap( colmap*0.7 + 0.3 );                           % Adjust colors to be "lighter"
    set(gca,'layer','top')                                  % Show grid on top of the map
    colorbar('south')                                       % Position of the colormap
    title(['Beam Geometry - ',sc{isc,1}])                   % Title
end

%%
% Note that the power_map function uses the antenna of the first UE as receive antenna, including
% its orientation, but it does not update the antenna orientation. In the simulation setup, the
% first UE is at a random location within the coverage area of the first beam. The power_map
% function moves this antenna to each pixel in the map to obtain a receive power value at this
% location. Hence, only at the true position of the first UE within the first beams is the terminal
% antenna pointed directly towards the satellite. All other map positions show less power due to the
% misalignment of the receive antenna. This is not a problem for the GSO satellite since virtually
% all UEs point their antenna southwards at 45 deg elevation. The misalignment at other map points
% is well within the main lobe of the receiver dish antenna. However, for the LEO satellite, there
% is a significant misalignment error towards the edges of the map. This effect does not have an
% influence on the results since each UE has its antenna perfectly aligned.


%% Run simulations and return results
% Based on the system-level simulation assumptions as described in 3GPP TR 38.821 Table 6.1.1.1-5,
% the results of the coupling loss, geometry SIR and geometry SINR are calculated. The 3GPP TR
% 38.821 results are reported in Table 6.1.1.2-1 for the DL and in Table 6.1.1.2-1 for the UL. These
% results are representative of the average performance reported by the different companies. Here,
% we obtain the results from the QuaDRiGa simulation setup. This is done by obtaining the channel
% coefficients for each beam to UE link and then calculating the received power values for each
% beam. The table is plotted at the end of this section. The first 30 results are for the downlink,
% the remainder is for the uplink.
 
clc                                                         % Clear Command Window
disp('       |        Coupling Loss        |        Geometry SIR         |        Geometry SINR');
fprintf('       '); for n=1:3; fprintf('|  @ 0.05 |  Median |  @ 0.95 '); end; fprintf('\n');

results = zeros(size(sc,1),9);                              % Init. results variable
for isc = 1 : size(sc,1)
    if isc == 1 || isc == 31
        for n=1:97; fprintf('-'); end; fprintf('\n');      	% Bar
    end
    
    c = l(1,isc).get_channels;                              % Obtain channels for all UE positions
    
    if sc{isc,4} == 1; FRF = FRF1;                          % Select FRF1
    elseif sc{isc,4} == 2; FRF = FRF3;                      % Select FRF3
    else; FRF = FRF4;                                       % Select FRF4
    end
    
    Ns = noise_thermal + 10*log10(sc{isc,8}*1e6) + sc{isc,7} - sc{isc,6}; % Noise
    
    pow = zeros( l(1,isc).no_rx, l(1,isc).no_tx );          % Power values of all beams
    pow_srv = zeros( l(1,isc).no_rx, 1 );                   % Serving beam power
    b_srv = zeros( l(1,isc).no_rx, 1 );                     % Serving beam number
    for ir = 1 : l(1,isc).no_rx
        for it = 1 : l(1,isc).no_tx
            pow(ir,it) = sum( abs(c(ir,it).coeff(:)).^2 );  % Calculate power for all beams
        end
        b_srv(ir,1)   = ceil((ir-0.5)./no_ms_per_beam);     % Calculate serving beam number 
        pow_srv(ir,1) = pow(ir,b_srv(ir,1));                % Calculate serving beam power
    end
    
    CPL = -10*log10(pow_srv);                               % Coupling loss
    [Sh,bins] = qf.acdf( CPL );                             % Calculate coupling loss CDF
    results(isc,1) = bins(find(Sh>=0.05,1));                % 5th percentile
    results(isc,2) = median(CPL);                           % Median
    results(isc,3) = bins(find(Sh>=0.95,1));                % 95th percentile
    
    SIR = []; SINR = [];                                    % Init. SIR and SINR
    for n = 1:4
        ii_mt  = FRF(b_srv) == n;                           % Find all UEs that use the same freq.
        ii_sat = FRF(1:l(1,isc).no_tx) == n;                % Find all beams that use the same freq.
        SIR    = [SIR;  10*log10( pow_srv(ii_mt)./(sum(pow(ii_mt,ii_sat),2)-pow_srv(ii_mt))) ];
        SINR   = [SINR; 10*log10( pow_srv(ii_mt)./...
            (sum(pow(ii_mt,ii_sat),2)-pow_srv(ii_mt)+10.^(Ns/10)))];
    end
    
    [Sh,bins] = qf.acdf( SIR );                             % Calculate SIR CDF
    results(isc,4) = bins(find(Sh>=0.05,1));                % 5th percentile
    results(isc,5) = median(SIR);                           % Median
    results(isc,6) = bins(find(Sh>=0.95,1));
    
    [Sh,bins] = qf.acdf( SINR );
    results(isc,7) = bins(find(Sh>=0.05,1));                % 5th percentile
    results(isc,8) = median(SINR);                          % Median
    results(isc,9) = bins(find(Sh>=0.95,1));                % 95th percentile
    
    disp(['SC',num2str(isc,'%02d'),'  ',sprintf(' |% 8.1f', results(isc,:)) ]); % Plot table line
end
