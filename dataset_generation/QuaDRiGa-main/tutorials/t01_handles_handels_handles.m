%% The Most Common Mistake: Handles
%
% This tutorial illustrates the most common mistake that new users of the QuaDRiGa channel model
% often make. QuaDRiGa is implemented in MATLAB / Octave using the object-oriented framework. All
% QuaDRiGa objects are "handles". That means that a variable created from a QuaDRiGa class can be
% regarded as a "pointer" to the associated data in the computer memory. This enables a very
% memory-efficient implementation, for example, if all mobile terminals use the same antenna. In
% this case, the antenna pattern only needs to be stored once in the memory and each MT only needs
% to store the "pointer" to the antenna and not a copy of the data. However, working with "handles"
% is something that many MATLAB users are unfamiliar with.

%%
% In the following simple example, a layout with two base stations is created. Each BS is equipped
% with a high-gain antenna which is tilted by 12 degrees. The antenna of the second BS is rotated by
% 180 degrees so that the antennas point towards each other. WARNING: The following code will not
% create the intended result. Try to find the mistake!

clear all

a = qd_arrayant('multi', 8, 0.5, 12 );                  % Generate High-Gain Antenna

l = qd_layout;                                          % New layout
l.no_tx = 2;                                            % Two BSs
l.tx_position(:,1) = [ -200 ; 0 ; 25 ];                 % Position of BS 1
l.tx_position(:,2) = [  200 ; 0 ; 25 ];                 % Position of BS 2

l.tx_array(1,1) = a;                                    % Assign antenna to BS1
l.tx_array(1,2) = a;                                    % Assign antenna to BS2
l.tx_array(1,2).rotate_pattern( 180 , 'z' );            % Rotate BS2 antenna by 180 degree

%% 
% Here we create a plot of the layout including the sum-power that would be received by a MT at each
% position of the layout. You will see that the antenna of the first BS points in the wrong
% direction. It should point towards the east (right), but it points to the west (left).

close all

set(0,'defaultTextFontSize', 18)                      	% Default Font Size
set(0,'defaultAxesFontSize', 18)                     	% Default Font Size
set(0,'defaultAxesFontName','Times')               	    % Default Font Type
set(0,'defaultTextFontName','Times')                 	% Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')       	% Default Plot position
set(0,'DefaultFigurePaperType','<custom>')             	% Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.8])            	% Default Paper Size

[ map,x_coords,y_coords] = l.power_map( '3GPP_38.901_UMa_LOS','quick',5,-500,500,-500,500,1.5 );
P = 10*log10( sum(cat(3,map{:}),3));                    % Total received power

l.visualize([],[],0);                                   % Show BS and MT positions on the map
hold on
imagesc( x_coords, y_coords, P );                       % Plot the received power
hold off
axis([-500 500 -500 500])                               % Plot size
caxis( max(P(:)) + [-20 0] )                            % Color range 
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
set(gca,'layer','top')                                  % Show grid on top of the map
title('Incorrect antenna orientation');                 % Set plot title

%% 
% The problem is the assignment of the antenna pattern. "a", "l.tx_array(1,1)" and "l.tx_array(1,2)"
% point to the same object. When the rotation operation "l.tx_array(1,2).rotate_pattern" is called,
% the data in memory is changed. However, "a" and "l.tx_array(1,1)" point to the same object and,
% therefore, their properties are now changed too. The following example shows the correct way to do
% it. In stead of assigning a "pointer", the "copy" command creates a new object with the same data.
% The rotation operation only effects the antenna of BS2.

a = qd_arrayant('multi', 8, 0.5, 12 );                  % Generate High-Gain Antenna

l.tx_array(1,1) = copy( a );                            % Assign copy of the antenna to BS1
l.tx_array(1,2) = copy( a );                            % Assign copy of the antenna to BS2
l.tx_array(1,2).rotate_pattern( 180 , 'z' );            % Rotate BS2 antenna by 180 degree

%% 
% The following plot shows the intended result.

[ map,x_coords,y_coords] = l.power_map( '3GPP_38.901_UMa_LOS','quick',5,-500,500,-500,500,1.5 );
P = 10*log10( sum(cat(3,map{:}),3));                    % Total received power

l.visualize([],[],0);                                   % Show BS and MT positions on the map
hold on
imagesc( x_coords, y_coords, P );                       % Plot the received power
hold off
axis([-500 500 -500 500])                               % Plot size
caxis( max(P(:)) + [-20 0] )                            % Color range 
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
set(gca,'layer','top')                                  % Show grid on top of the map
title('Correct antenna orientation');                   % Set plot title

