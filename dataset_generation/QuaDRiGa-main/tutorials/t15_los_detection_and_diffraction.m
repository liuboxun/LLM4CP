%% Site Specific Simulations
%
% This tutorial provides a comprehensive guide on utilizing the site-specific extensions introduced
% in QuaDRiGa 2.8. These extensions enable the import of a 3D model into QuaDRiGa, facilitating the
% identification of areas with line-of-sight (LOS) and non-LOS coverage. Consequently, it becomes
% feasible to automatically assign propagation scenarios to tracks or generate coverage maps for
% specific locations. These maps can offer rough estimates of the received power at the user
% equipment (UE) location, incorporate antenna models, and more. Key topics covered in this tutorial
% include:
%
% * Importing of 3D models
% * Positioning transmitters and receivers
% * Generating coverage maps
% * Incorporating antenna models
% * Assigning LOS and NLOS sections to track sections
%
% The 3D model extension is part of the "qd_mesh" class. However, it has two versions: a MATLAB
% version and an Nvidia-CUDA version for GPU acceleration. The latter, available only on Linux, is
% located in the "+qext" module within the "quadriga_src" folder and requires compilation for your
% specific CPU/GPU architecture. Use the "qd_mesh.has_gpu" command to check if your system supports
% GPU acceleration and whether the extension is compiled correctly.

%% Setting general parameters
% We set up some basic parameters such as center frequency and sample density.

close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.7])            	% Default Paper Size

s = qd_simulation_parameters;                           % New simulation parameters
s.center_frequency = 3.7e9;                             % Center frequencies in [Hz]
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.sample_density = 1.1;                                 % Minimum possible sample density

disp( ['GPU = ', num2str(qd_mesh.has_gpu)] );           % Check if we have GPU acceleration

%% Importing 3D models and defining the simulation layout
% We begin by setting up our simulation layout. This includes a BS equipped with single-element
% patch antenna situated in an urban area. The configuration mimics a typical urban deployment
% with equipment located above rooftops. For the 3D model, we employ the Madrid grid, a compact test
% scenario developed by the METIS project (refer to: Metis ICT-317669-METIS/D1.4 METIS Channel
% Models). This model is provided in the Wavefront OBJ file format, a text-based file format
% designed for storing and exchanging 3D model data. It is compatible with and can be directly
% exported from widely-used 3D modeling software, such as Blender.

l = qd_layout( s );                                     % New layout
l.tx_position = [ 267 ; 267 ; 60 ];                     % Base station position
l.rx_position = [ 15 ; 415 ; 1.2 ];                     % MT position

l.tx_array = qd_arrayant( 'patch' );                    % Simple patch antenna @ BS
l.tx_track.orientation(3) = pi/2;                       % Facing north

obj_fn = fullfile('madrid_grid','madrid_grid.obj');     % Location and name of the OBJ file
m = qd_mesh;                                            % Class for handling 3D model data
m.read_obj( obj_fn );                                   % Load 3D model

l.visualize([],[],2);                                   % Plot layout with UE names
hold on
m.visualize([],0);                                      % Plot 3D model
hold off
axis([-50,450,-50,600]);                                % Set plot dimensions
title('Simulation Layout');
drawnow
try; alpha(0.3); end                                    % Semi-transparent walls (only MATLAB)
view(-30,45)                                            % 3D view


%% Plot LOS / NLOS and Indoor areas
% In this section, we utilize the 3D model to assess the propagation conditions for different areas
% of the map. 3GPP identifies four possible conditions: 1. Direct Line-of-Sight (LOS) to the base
% station (BS), 2. Outdoor-to-Indoor with LOS illumination of the building, 3. Outdoor Non-LOS, and
% 4. Outdoor-to-Indoor with indirect illumination of the building. The "qd_mesh" class offers a
% method to calculate whether a direct LOS exists between two points or if the path is obstructed by
% the 3D model.
%
% We implement this method in two ways: Firstly, we verify whether the BS is visible from a given
% location. Secondly, we check if the sky is visible from the same location by initiating the path
% 500 meters above the receiver (RX) location. This approach also allows us to determine the number
% of interactions with the model. Consequently, we can identify the four states, which are then
% visually represented on the map.

x_min = 0; x_max = 387; y_min = 0; y_max = 552;         % Map edges (set by model size)
pixel_size = 2;                                         % in [m]

% Generate a grid of receiver positions
x = x_min : pixel_size : x_max;
y = y_min : pixel_size : y_max;
[X,Y] = meshgrid(x,y);
rx_pos = [X(:)'; Y(:)'; ones(1, numel(X))*1.2];         % Place receivers at 1.2 m height

% Determine LOS state and number of 3D-model interactions for each grid position
[ is_los, no_trans ] = intersect_mesh( m, l.tx_position, rx_pos );

% Determine indoor locations
tx_pos = rx_pos; tx_pos(3,:) = 500;                     % Place transmitters 500 m above each RX
is_outdoor = intersect_mesh( m, tx_pos, rx_pos );       % RXs that are not under a roof

state = zeros( size(is_los) );
state( is_los ) = 1;                                    % Set LOS state
state( is_outdoor == 0 & no_trans == 1 ) = 2;           % Indoor with LOS illumination of building
state( is_los == 0 & is_outdoor == 1 ) = 3;             % Outdoor NLOS state
state( is_outdoor == 0 & no_trans > 1 ) = 4;            % Indoor state with indirect illuminaion
state = reshape( state, numel(y), numel(x) );

l.visualize([],[],2);                                   % Plot layout with UE names
hold on
m.visualize( [], 0 );                                   % Plot 3D model
try; alpha(0.3); end                                    % Semi-transparent walls (only MATLAB)
im = imagesc( x, y, state );                            % Plot propagation state
hold off
axis([x_min,x_max,y_min,y_max])
colormap jet
title('Propagation State')
drawnow

%% Path Gain Prediction
% Next, we estimate the received power across the map. This involves determining which areas receive
% line-of-sight (LOS) service and which are under non-LOS (NLOS) conditions. Subsequently, we select
% the appropriate 3GPP path loss model. It's important to note that at lower frequencies, a
% transition area exists between LOS and NLOS, characterized by diffraction. Diffraction is the
% phenomenon where waves bend or interfere around the edges of an obstacle, reaching into regions
% that would otherwise be shadowed by the obstacle.
%
% As an experimental feature, the method "diff_trans" not only provides information about which
% locations are in LOS or NLOS, but also a relative weighting (ranging from 0 for NLOS to 1 for
% LOS). This data can be used to compute a weighted average of the two path loss models for any
% given location on the map.
%
% The following code demonstrates this methodology. Initially, we compute the path loss for each
% position on the map using the "qd_builder" class, incorporating the BS antenna gain and
% orientation. The diffraction gain is then determined from the 3D model using "qd_mesh". Lastly, we
% calculate the weighted path gain and plot the results.

b = qd_builder();
b.simpar.center_frequency = s.center_frequency(1);
b.tx_array = l.tx_array;
b.tx_track = l.tx_track;
b.rx_positions = rx_pos;

b.scenario = '3GPP_38.901_UMa_LOS';
c = b.get_los_channels('single', 'coeff');
p_los = reshape( sum(abs(c).^2, 2), numel(y), numel(x) );

b.scenario = '3GPP_38.901_UMa_NLOS';
c = b.get_los_channels('single', 'coeff');
p_nlos = reshape( sum(abs(c).^2, 2), numel(y), numel(x) );

% Calculate the diffraction gain
gain_diff = diff_trans(m, l.tx_position, rx_pos, s.center_frequency, 37, 4, 1);
gain_diff = reshape( gain_diff, numel(y), numel(x) );

% Combine the power values using the propagation state
gain = p_los .* gain_diff + (1-gain_diff) .* p_nlos;

% Plot results
l.visualize([],[],2);                                   % Plot layout with UE names
hold on
m.visualize( [], 0 );                                   % Plot 3D model
try; alpha(0.3); end                                    % Semi-transparent walls (only MATLAB)
im = imagesc( x, y, 10*log10(gain) );                   % Plot path gain
hold off
colorbar('westoutside')
caxis( [-130 -70] )
axis([x_min,x_max,y_min,y_max] )
colormap jet
title('Path Gain [dB]')
drawnow

%% Assign Propagation States to Tracks
% In the final part of this tutorial, we illustrate how to utilize the 3D model for automatically
% assigning propagation states to a mobile terminal (MT). The MT follows a straight path of 350
% meters, moving from west to east across a plaza covered by the base station (BS). Using "qd_mesh,"
% we obtain the Line-of-Sight (LOS) state along the track and identify the transition points.
% Subsequently, we assign the appropriate propagation scenario to each segment of the track. In
% contrast to the previous section, state transitions are managed by the QuaDRiGa channel merger,
% eliminating the need for the diffraction model. Instead, we define a segment length and a
% transition region. The resulting plot displays the received power along the track.
%
% However, it is crucial to remember that QuaDRiGa generates random propagation parameters, such as
% shadow fading and scatterer positions. Additionally, the path loss model is quite generic.
% Therefore, significant discrepancies can arise between the predicted path loss and the actual path
% loss in a real-world deployment. This factor may limit the practicality of this extension for
% network planning purposes.

t = qd_track( 'linear', 350 , 0 );                      % MT trajectory
t.interpolate('distance',1/s.samples_per_meter,[],[],1);  % Interpolate
t.initial_position = [ 15 ; 415 ; 1.2 ];                % Start position
t.name = 'MT1';                                         % Assign name

% Get LOS / NLOS state along the track from 3D model
is_los = intersect_mesh( m, l.tx_position, t.positions_abs );
transitions = find( abs( diff( is_los ) ) > 0.5 );      % Find LOS <> NLOS transition points
t.segment_index = [1, transitions'];                    % Set transition points as segments

% Calculate the center position of the segments
center = round( [1;transitions]/2 + [transitions; t.no_snapshots]/2 );

% Set the propagation conditions
t.scenario( is_los(center)) = {'3GPP_38.901_UMa_LOS'};
t.scenario(~is_los(center)) = {'3GPP_38.901_UMa_NLOS'};

t.split_segment(5,25,18);       % Generate sub-segements between 5 and 25 m length
t.correct_overlap;              % Shift transition points to middle of segment-overlapping area

l.rx_track(1,1) = t;            % Add track to layout
c = l.get_channels;             % Calculate channel coefficients

pow = 10*log10( reshape( sum(abs(c.coeff(:,:,:,:)).^2,3), [], 1 ) );   % Calculate the power
x_position = c.rx_position(1,:);                                        % X-position
ar = zeros(1,c.no_snap); ar(~is_los) = -200;                            % Shading for NLOS

% Plot results
set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change Plot Size
figure('Position',[ 100 , 100 , 1200 , 400]);           % New figure
a = area(x_position,ar,'FaceColor',[0.7 0.9 0.7],'LineStyle','none'); % Area shading
hold on; plot(x_position,pow); hold off;
xlabel('x-coord [m]'); ylabel('Path Gain [dB]'); grid on;
axis([x_position(1),x_position(end),[-140,-70]]);
legend('NLOS','Path Gain');
set(gca,'layer','top')
