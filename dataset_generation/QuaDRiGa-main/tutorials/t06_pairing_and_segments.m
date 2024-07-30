%% Pairing and segments
%
% This tutorial shows how to set up scenarios with several transmitters and receivers and the use of
% scenarios. First, we set up a basic simulation with two transmitters. One of them is outdoors, the
% other is indoors.

%%
clear all

s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 0;                               % Disable progress bars
s.center_frequency = 2.53e9;                            % Set center frequency
l = qd_layout(s);                                       % Create new QuaDRiGa layout

l.no_tx = 2;                                            % Two BSs
l.tx_position(:,1) = [ -142 ; 355 ; 64 ];               % Outdoor BS
l.tx_position(:,2) = [ 5 ; 0; 10 ];                     % Indoor BS

%%
% We create two different MTs. MT1 is indoors. The link to BS1 is in scenario "WINNER_UMa_C2_NLOS".
% The link to BS2 is in "WINNER_Indoor_A1_LOS". MT1 has no segments. The rows in "qd_track.scenario"
% indicate the scenario for each BS. If there is only one row, then all BSs get the same scenario.
% The second MT is outdoors, far away from the indoor BS. The first part of the MT2 track is in LOS,
% the second is in NLOS. The columns of track.scenario indicate the segments. Here, all BSs get the
% same scenarios.

l.no_rx = 2;                                            % Two MTs
l.rx_track(1,1) = qd_track('linear', 0.2 );             % Linear track with 20 cm length
l.rx_track(1,1).name = 'Rx1';                           % Set the MT1 name
l.rx_track(1,1).scenario = {'WINNER_UMa_C2_NLOS';'WINNER_Indoor_A1_LOS'};  % Two Scenarios

l.rx_track(1,2) = qd_track('linear', 0.2 );             % Linear track with 20 cm length
l.rx_track(1,2).name = 'Rx2';                           % Set the MT2 name

l.rx_position(:,2) = [ 100;50;0 ];                      % Start position of the MT2 track
interpolate_positions( l.rx_track, s.samples_per_meter );  % Interpolate positions

l.rx_track(1,2).segment_index = [1 3];                  % Set segments
l.rx_track(1,2).scenario = {'WINNER_UMa_C2_LOS','WINNER_UMa_C2_NLOS'};

%%
% We calculate the channel coefficients and plot the list of created segments.

cb = l.init_builder;                                    % Initialize builder
gen_parameters( cb );                                   % Generate small-scale-fading
c = get_channels( cb );                                 % Get channel coefficients
disp( strvcat( c.name ) )                               % Show the names if the channels

%%
% As we can see, 6 segments were generated. However, the channel Tx2_Rx2 will most likely not be
% needed because of the large distance.  We thus remove the link from the pairing matrix and
% recompute the channels.

l.pairing = [1 2 1 ; 1 1 2 ];                           % Change the pairing matrix

cb = l.init_builder;                                    % Initialize channel builder object
gen_parameters( cb );                                   % Generate small-scale-fading parameters
c = get_channels( cb );                                 % Get channel coefficients
disp( strvcat( c.name ) )                               % Show the names of the channels


%%
% At last, we can combine the segments and generate the final channels.

cn = merge( c );                                        % Combine the channel coefficients
disp( strvcat( cn.name ) )                              % Show the names if the channels

