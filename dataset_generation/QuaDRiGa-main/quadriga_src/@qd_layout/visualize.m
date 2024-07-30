function han = visualize( h_layout, tx , rx, show_names , create_new_figure )
%VISUALIZE Plots the layout. 
%
% Calling object:
%   Single object
%
% Input:
%   tx
%   A vector containing the transmitter indices that should be shown. Default: All
%
%   rx
%   A vector containing the receiver indices that should be shown. Default: All
%
%   show_names
%   Options: (0) shows no Tx and Rx names; (1, default) shows the Tx name and the scenario for each
%   track segment; (2) shows the Tx and Rx name
%
%   create_new_figure
%   If set to 0, no new figure is created, but the layout is plotted in the currently active figure
%
% Output:
%   han
%   The figure handle
%
% 
% QuaDRiGa Copyright (C) 2011-2019
% Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
% Fraunhofer Heinrich Hertz Institute, Einsteinufer 37, 10587 Berlin, Germany
% All rights reserved.
%
% e-mail: quadriga@hhi.fraunhofer.de
%
% This file is part of QuaDRiGa.
%
% The Quadriga software is provided by Fraunhofer on behalf of the copyright holders and
% contributors "AS IS" and WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, including but not limited to
% the implied warranties of merchantability and fitness for a particular purpose.
%
% You can redistribute it and/or modify QuaDRiGa under the terms of the Software License for 
% The QuaDRiGa Channel Model. You should have received a copy of the Software License for The
% QuaDRiGa Channel Model along with QuaDRiGa. If not, see <http://quadriga-channel-model.de/>. 

% Parse input arguments
if exist('tx','var') && ~isempty( tx )
    if isempty(tx)
        tx = 1:h_layout.no_tx;
    elseif ~( size(tx,1) == 1 && isnumeric(tx) ...
            &&  all( mod(tx,1)==0 ) && min(tx) > 0 && max(tx)<=h_layout.no_tx )
        error('??? "tx" must be integer > 0 and can not exceed array size')
    end
else
    tx = 1:h_layout.no_tx;
end

if exist('rx','var') && ~isempty( rx )
    if isempty(rx)
        rx = 1:h_layout.no_rx;
    elseif ~( size(rx,1) == 1 && isnumeric(rx) ...
            &&  all( mod(rx,1)==0 ) && min(rx) > 0 && max(rx)<=h_layout.no_rx )
        error('??? "rx" must be integer > 0 and can not exceed array size')
    end
else
    rx = 1:h_layout.no_rx;
end

if exist('show_names','var') && ~isempty( show_names )
    if ~( all(size(show_names) == [1,1]) && isnumeric(show_names) )
        error('??? "show_names" must be scalar and numeric.')
    end
else
    show_names = 1;
end

if exist('create_new_figure','var') && ~isempty( create_new_figure )
    if ~( all(size(create_new_figure) == [1,1]) && isnumeric(create_new_figure) )
        error('??? "create_new_figure" must be scalar and numeric.')
    end
else
    create_new_figure = true;
end

% Create a new figure
if create_new_figure
    han = figure('Position',[ 100 , 100 , 1000 , 700]);
end

% Get the orientation of the Rx
for n = 1 : h_layout.no_rx
    if isempty( h_layout.rx_track(1,n).orientation )
        h_layout.rx_track(1,n).calc_orientation;
    end
end

% Get the orientation of the Tx
for n = 1 : h_layout.no_tx
    if isempty( h_layout.tx_track(1,n).orientation )
        if h_layout.tx_track(1,n).no_snapshots == 1
            h_layout.tx_track(1,n).orientation = [0;0;0];
        else
            h_layout.tx_track(1,n).calc_orientation;
        end
    end
end

% Optional visualization of LOS and NLOS
tx_has_los = false( 1,numel(tx) );
if numel(rx) == 1 && numel(tx) > 1 && h_layout.rx_track(1,1).no_segments == 1 && ...
        numel( h_layout.rx_track(1,1).scenario ) == numel(tx)
    for n = 1:numel( tx )
        tx_has_los(n) = ~isempty( regexp( h_layout.rx_track(1,rx).scenario{tx(n),1}, '_LOS', 'once' ) );
    end
end

rx_has_los = false( 1,numel(rx) );
if numel(tx) == 1 && numel(rx) > 1 && numel( h_layout.rx_track(1,1).scenario ) >= tx
    for n = 1:numel( rx )
        rx_has_los(n) = ~isempty( regexp( h_layout.rx_track(1,n).scenario{tx,1}, '_LOS', 'once' ) );
    end
end

% Plot the first Tx position
if tx_has_los(1)
    cross_color = '+g';
else
    cross_color = '+r';
end
plot3( h_layout.tx_position(1,tx(1)),h_layout.tx_position(2,tx(1)),h_layout.tx_position(3,tx(1)),...
    cross_color,'Linewidth',3,'Markersize',15 );
hold on

% Plot the Tx antenna element positions, including the orientation
O = h_layout.tx_track(1,tx(1)).orientation(:,1);
R = qf.calc_ant_rotation( O(3,:) , -O(2,:), O(1,:) );
pos_tx_ant = R * h_layout.tx_array(1,tx(1)).element_position;
for n=1:3
    pos_tx_ant(n,:) = h_layout.tx_position(n,tx(1)) + pos_tx_ant(n,:);
end
plot3( pos_tx_ant(1,:),pos_tx_ant(2,:),pos_tx_ant(3,:),'^r','Linewidth',1,'Markersize',10 );

% Plot the first Tx track
pos = h_layout.tx_track(1,tx(1)).positions;
for n=1:3
    pos(n,:) = pos(n,:) + h_layout.tx_track(1,tx(1)).initial_position(n);
end
plot3( pos(1,:),pos(2,:),pos(3,:),'r' );

% Plot the first Rx position
if rx_has_los(1)
    o_color = 'og';
else
    o_color = 'ob';
end
seg_pos = h_layout.rx_track(1,rx(1)).initial_position;
plot3( seg_pos(1,:),seg_pos(2,:),seg_pos(3,:),o_color,'Linewidth',2,'Markersize',10 );


% Plot the Rx antenna element positions, including the orientation
O = h_layout.rx_track(1,rx(1)).orientation(:,1);
R = qf.calc_ant_rotation( O(3,:) , -O(2,:), O(1,:) );
pos_rx_ant = R * h_layout.rx_array(1,rx(1)).element_position;
for n=1:3
    pos_rx_ant(n,:) = h_layout.rx_track(1,rx(1)).initial_position(n,1) + pos_rx_ant(n,:);
end
plot3( pos_rx_ant(1,:),pos_rx_ant(2,:),pos_rx_ant(3,:),'vb','Linewidth',1,'Markersize',10 );


% Plot the first Rx track
pos = h_layout.rx_track(1,rx(1)).positions;
for n=1:3
    pos(n,:) = pos(n,:) + h_layout.rx_track(1,rx(1)).initial_position(n);
end
plot3( pos(1,:),pos(2,:),pos(3,:) );


% Do for each Tx
for m = 1 : numel(tx)
    
    % Plot a line from the ground to the Tx
    plot3( [h_layout.tx_position(1,tx(m)),h_layout.tx_position(1,tx(m))] , ...
        [h_layout.tx_position(2,tx(m)),h_layout.tx_position(2,tx(m))],...
        [0,h_layout.tx_position(3,tx(m))] ,'--r' );
    
    if m > 1
        
        % Plot the m-th Tx position
        if tx_has_los(tx(m))
            cross_color = '+g';
        else
            cross_color = '+r';
        end
        plot3( h_layout.tx_position(1,tx(m)),h_layout.tx_position(2,tx(m)),h_layout.tx_position(3,tx(m)),...
            cross_color,'Linewidth',3,'Markersize',15 );
        
        % Plot the m-th Tx antenna element positions, including the orientation
        O = h_layout.tx_track(1,tx(m)).orientation(:,1);
        R = qf.calc_ant_rotation( O(3,:) , -O(2,:) , O(1,:) );
        pos_tx_ant = R * h_layout.tx_array(1,tx(m)).element_position;
        for n=1:3
            pos_tx_ant(n,:) = h_layout.tx_position(n,tx(m)) + pos_tx_ant(n,:);
        end
        plot3( pos_tx_ant(1,:),pos_tx_ant(2,:),pos_tx_ant(3,:),'^r','Linewidth',1,'Markersize',10 );
        
        % Plot the m-th Tx track
        pos = h_layout.tx_track(1,tx(m)).positions;
        for n=1:3
            pos(n,:) = pos(n,:) + h_layout.tx_track(1,tx(m)).initial_position(n);
        end
        plot3( pos(1,:),pos(2,:),pos(3,:),'r' );
        
    end
    
    if show_names
        text( double( h_layout.tx_position(1,tx(m))),...
            double(h_layout.tx_position(2,tx(m))), ...
            double(h_layout.tx_position(3,tx(m))),...
            [h_layout.tx_name{tx(m)},' '],'HorizontalAlignment','right') ;
    end
end

% Do for each Rx
for m = 1:numel(rx)
    
    % Determine the segment positions
    si = h_layout.rx_track(1,rx(m)).segment_index;
    pos = double( h_layout.rx_track(1,rx(m)).positions );
    for n=1:3
        pos(n,:) = pos(n,:) + h_layout.rx_track(1,rx(m)).initial_position(n);
    end
    sii = floor( si + 0.3*([si(2:end),h_layout.rx_track(1,rx(m)).no_snapshots ]-si ) );
    
    
    % Plot the m-th Rx position, including segments
    if rx_has_los(m)
        o_color = 'og';
    else
        o_color = 'ob';
    end
    plot3( pos(1,si),pos(2,si),pos(3,si) ,o_color,'Linewidth',2,'Markersize',10 );
    
    if m>1
        % Plot the m-th Rx antenna element positions, including the orientation
        O = h_layout.rx_track(1,rx(m)).orientation(:,1);
        R = qf.calc_ant_rotation( O(3,:) , -O(2,:) , O(1,:) );
        pos_rx_ant = R * h_layout.rx_array(1,rx(m)).element_position;
        for n=1:3
            pos_rx_ant(n,:) = h_layout.rx_track(1,rx(m)).initial_position(n,1) + pos_rx_ant(n,:);
        end
        plot3( pos_rx_ant(1,:),pos_rx_ant(2,:),pos_rx_ant(3,:),'vb','Linewidth',1,'Markersize',10 );
        
        % Plot the m-th Rx track
        plot3( pos(1,:),pos(2,:),pos(3,:) );
        
    end
    
    if show_names == 1
        for o = 1:numel(sii)
            tmp = h_layout.rx_track(1,rx(m)).scenario(:,o);
            tmp = regexprep(tmp,'_','\\_');
            text(pos(1,sii(o)),pos(2,sii(o)),pos(3,sii(o)),tmp) ;
        end
    elseif show_names == 2
        text(double(h_layout.rx_track(1,rx(m)).initial_position(1)),...
            double(h_layout.rx_track(1,rx(m)).initial_position(2)),...
            double(h_layout.rx_track(1,rx(m)).initial_position(3)),...
            [' ',h_layout.rx_track(1,rx(m)).name]) ;
    end
end

hold off

grid on
box on
view(0, 90);

xlabel('x-coord in [m]');
ylabel('y-coord in [m]');
zlabel('z-coord in [m]');

legend('Tx-Position','Tx-Antenna','Tx-Track','Rx-Position','Rx-Antenna','Rx-Track',...
    'Location','NorthEastOutside')

a = axis;
A = a(2)-a(1);
B = a(4)-a(3);
if A<B
    a(1) = a(1) - 0.5*(B-A);
    a(2) = a(2) + 0.5*(B-A);
else
    a(3) = a(3) - 0.5*(A-B);
    a(4) = a(4) + 0.5*(A-B);
end

a(1) = min( a(1) , -1 );
a(2) = max( a(2) , 1 );
a(3) = min( a(3) , -1 );
a(4) = max( a(4) , 1 );

a = a*1.1;
axis(a);

end
