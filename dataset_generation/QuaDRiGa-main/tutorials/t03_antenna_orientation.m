%% Effects of the Antenna-Orientation
%
% This tutorial shows how to evaluate antenna effects. It creates a simple setup with a transmit and
% a receive antenna facing each other in pure LOS conditions. Then, the transmitter is rotated
% around its x-axis and the effect on the received power is studied.
%
% One feature of the model is that it allows to freely orient the antennas at the transmitter and
% receiver. In the following, two cross-polarized patch antennas are aligned on the optical axis
% facing each other. The surface normal vectors of the transmit and the receive patch are aligned
% with the LOS. The transmitter is rotated from -90° to 90° around the optical axis. The real and
% imaginary parts of the channel coefficients are then calculated for each angle. Each real and
% imaginary part is normalized by its maximum and the results are plotted. The calculation is done
% for both, linearly and crossed polarized elements. 

%% Model and Antenna Setup
% Here, we parametrize the simulation. We place the receiver 10 m away from the transmitter and
% chose the scenario "LOSonly". Thus, no NLOS components are present. The receiver is set up as a
% multi-element array antenna using both, circular and linear polarization.

clear all
close all

a = qd_arrayant('lhcp-rhcp-dipole');    % Create circular polarized antenna

a2 = qd_arrayant('custom',90,90,0);     % Create linear polarized patch antenna
a2.copy_element(1,2);                   % Copy the antenna element
a2.rotate_pattern(90,'x',2);            % Rotate second element by 90 degree

a.append_array( a2 );                   % Append the second antenna to the first

l = qd_layout;
l.simpar.show_progress_bars = 0;        % Disable progress bar indicator

l.rx_track = qd_track('linear',0,pi);   % Create new track (pi turns the rx by 180 degree)
l.rx_position = [11;0;0];               % Set the receiver position
l.tx_position = [0;0;0];

l.set_scenario( 'LOSonly' );            % Set the scenario to LOS only
l.tx_array = a;                         % Use same antenna at both sides
l.rx_array = a;


%% Iteration over all angles
% Next, we rotate the receive antenna in 10 degree steps around its x-axis and calculate the channel
% response for each angle. 

warning('off','QuaDRiGa:qd_layout:BuilderReset');       % Disable builder-reset warnings
rot = -120:10:120;                                      % Rotation angle 
h = zeros(4,4,numel(rot));
for n = 1 : numel(rot)
    cc = copy( a );                                     % Create copy of the Tx antenna ( !!! )
    cc.rotate_pattern( rot(n) , 'x');                   % Assign rotation angle

    l.tx_array = cc;                                    % Set Tx antenna
    c = l.get_channels;                                 % Update channel coefficients
    h(:,:,n) = c.coeff(:,:,1,1);
end
warning('on','all');                                    % Enable all warnings

%% Linear Polarization results
% Now we plot the results for the linear polarization. There are two linearly polarized antennas at
% the transmitter and two at the receiver. Their orientation can either be vertical (denoted as V)
% or horizontal (denoted as H). The channel matrix thus has 8 coefficients, VV, VH, HV and HH. Each
% coefficient is complex-valued. Thus, figure shows 8 curves, 4 for the real parts and 4 for the
% imaginary parts. 

set(0,'defaultTextFontSize', 18)                      	% Default Font Size
set(0,'defaultAxesFontSize', 18)                     	% Default Font Size
set(0,'defaultAxesFontName','Times')               	    % Default Font Type
set(0,'defaultTextFontName','Times')                 	% Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')       	% Default Plot position
set(0,'DefaultFigurePaperType','<custom>')             	% Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 4.7])            	% Default Paper Size

figure('Position',[ 100 , 100 , 760 , 400]);
g = h([3,4],[3,4],:);

plot(rot,squeeze(real(g(1,1,:))),'-sk','Linewidth',0.5,'MarkerfaceColor','k','Markersize',12)
hold on
plot(rot,squeeze(real(g(2,2,:))),'-db','Linewidth',0.5,'MarkerfaceColor','b','Markersize',8)
plot(rot,squeeze(real(g(2,1,:))),'-or','Linewidth',0.5,'MarkerfaceColor','r','Markersize',8)
plot(rot,squeeze(real(g(1,2,:))),'-^g','Linewidth',0.5,'MarkerfaceColor','g','Markersize',8)

plot(rot,squeeze(imag(g(1,1,:))),'--sk','Linewidth',0.5,'Markersize',12)
plot(rot,squeeze(imag(g(2,2,:))),'--db','Linewidth',0.5,'Markersize',8)
plot(rot,squeeze(imag(g(2,1,:))),'--or','Linewidth',0.5,'Markersize',12)
plot(rot,squeeze(imag(g(1,2,:))),'--^g','Linewidth',0.5,'Markersize',12)
hold off

xlabel('Rotation Angle')
ylabel('Normalized Amplitude')
legend('real V-V','real H-H','real H-V','real V-H',...
    'imag V-V','imag H-H','imag H-V','imag V-H','location','EastOutside')


%% Circular Polarization results
% The second plot shows the same for circular polarization. The first element is LHCP (denoted as L)
% and the second is RHCP (denoted as R). As expected, all cross-polarization coefficients (RL and
% LR) are zero. 

figure('Position',[ 100 , 100 , 760 , 400]);
g = h([1,2],[1,2],:);

plot(rot,squeeze(real(g(1,1,:))),'-sk','Linewidth',0.5,'MarkerfaceColor','k','Markersize',12)
hold on
plot(rot,squeeze(real(g(2,2,:))),'-db','Linewidth',0.5,'MarkerfaceColor','b','Markersize',8)
plot(rot,squeeze(real(g(2,1,:))),'-or','Linewidth',0.5,'MarkerfaceColor','r','Markersize',8)
plot(rot,squeeze(real(g(1,2,:))),'-^g','Linewidth',0.5,'MarkerfaceColor','g','Markersize',8)

plot(rot,squeeze(imag(g(1,1,:))),'--sk','Linewidth',0.5,'Markersize',12)
plot(rot,squeeze(imag(g(2,2,:))),'--db','Linewidth',0.5,'Markersize',8)
plot(rot,squeeze(imag(g(2,1,:))),'--or','Linewidth',0.5,'Markersize',12)
plot(rot,squeeze(imag(g(1,2,:))),'--^g','Linewidth',0.5,'Markersize',12)
hold off

xlabel('Rotation Angle')
ylabel('Normalized Amplitude')
legend('real L-L','real R-R','real R-L','real L-R',...
    'imag L-L','imag R-R','imag R-L','imag L-R','location','EastOutside')

