% Author: Boxun Liu
% Demo CSI generation code
clc;clear;close all;
s = qd_simulation_parameters;
s.center_frequency = 2.4e9; %center frequency
%% Base station antenna configuration
M_BS = 4; 
N_BS = 4; 
Mg_BS = 1; % Number of nested panels in a column
Ng_BS = 1; % Number of nested panels in a row
ElcTltAgl_BS = 7; 
Hspc_Tx_BS = 0.5*s.wavelength; % Horizontal array element spacing
Vspc_Tx_BS = 0.5*s.wavelength; % Vertical array element spacing

BSAntArray = qd_arrayant.generate('3gpp-mmw',M_BS,N_BS,...
    s.center_frequency,2,ElcTltAgl_BS,...
    Vspc_Tx_BS/s.wavelength,Mg_BS,Ng_BS,...
    Vspc_Tx_BS/s.wavelength*M_BS,Hspc_Tx_BS/s.wavelength*N_BS); 
     
%% UE antenna configuration

UEAntArray = qd_arrayant.generate('3gpp-mmw',1,1,...
    s.center_frequency,1,ElcTltAgl_BS,...
    Vspc_Tx_BS/s.wavelength,Mg_BS,Ng_BS,...
    Vspc_Tx_BS/s.wavelength*M_BS,Hspc_Tx_BS/s.wavelength*N_BS);

Speed=10.1:0.1:100; %[km/h]
H_U_his = zeros(900,10,16,48,4,4,2);
H_U_pre = zeros(900,10,4,48,4,4,2);
H_D_pre = zeros(900,10,4,48,4,4,2);
for iter_Speed=1:length(Speed)
    disp(iter_Speed)

    UENum = 10;
    UESpeed = Speed(iter_Speed); %[km/h]
    Timelength = 19*0.5e-3; %Sample Period Length,[s]
    UETrackLength = UESpeed/3.6*Timelength; %Calculate UE motion path length
    IndoorRatio = 0; % Indoor user ratio
    TimeInterval = 0.5e-3; %Time sampling interval,[s]
    SnapNum = 1+floor(Timelength/TimeInterval);
    %% Configure BS-UE channel parameters
    s1 = qd_simulation_parameters;
    s1.center_frequency = 2.4e9;
    s1.set_speed(UESpeed,TimeInterval);
    s1.use_random_initial_phase = true;
    s1.use_3GPP_baseline = 1;


    %BS location
    BSlocation = [0;0;30];
    rho_min = 20;
    rho_max = 50;
    rho = rho_min+(rho_max-rho_min)*rand(1,UENum);
    phi =  120*rand(1,UENum)-60;
    UEcenter = [200;0;1.5];
    UElocation = zeros(3,UENum);
    d = zeros(1,UENum);
    for ind_UE = 1:UENum
        rho_n = rho(ind_UE);
        phi_n = phi(ind_UE);
        UElocation(:,ind_UE) = [-rho_n*cosd(phi_n);rho_n*sind(phi_n);0]+UEcenter;
        d(ind_UE) = norm(UElocation(:,ind_UE)-BSlocation);
    end
    
 
    %UE track
    for ind_UE = 1:UENum
        UEtrack(1,ind_UE) = qd_track.generate('linear',UETrackLength);
        UEtrack(1,ind_UE).name = num2str(ind_UE);
        UEtrack(1,ind_UE).interpolate('distance',1/s1.samples_per_meter,[],[],1);
    end

    % BS-UE
    l1 = qd_layout(s1); %Layout initialization
    l1.no_tx = 1; %Configure the number of base stations
    l1.tx_array = BSAntArray; %Configuring the antenna on the base station side
    l1.tx_position = BSlocation; %Configure the base station spatial location
    l1.no_rx = UENum; %Configure the number of users
    l1.rx_array = UEAntArray; %Configuring the user-side antenna
    l1.rx_track = UEtrack;
    l1.rx_position = UElocation; %Configuring user space locations
    l1.set_scenario('3GPP_38.901_UMa_NLOS');
%     l1.set_scenario('Freespace');

    [BS2UE_channel,BS2UE_builder] = l1.get_channels();
    for ii=1:UENum
        h=BS2UE_channel(ii).fr(17280e3,96);  % OFDM channel generation
        h=reshape(h,2,4,4,96,20);
        h=permute(h,[5,4,3,2,1]);
        H_U_his(iter_Speed,ii,:,:,:,:,:)=h(1:16,1:48,:,:,:);  % historical uplink CSI
        H_U_pre(iter_Speed,ii,:,:,:,:,:)=h(17:20,1:48,:,:,:); % future uplink CSI
        H_D_pre(iter_Speed,ii,:,:,:,:,:)=h(17:20,49:96,:,:,:); % future downlink CSI
    end
end