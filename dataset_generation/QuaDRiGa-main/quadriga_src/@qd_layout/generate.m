function h_layout = generate( layout_type, no_sites, isd, h_arrayant, no_sectors, sec_orientation )
%GENERATE Generates predefined network layouts.
%
% Calling object:
%   None (static method)
%
% Layout types:
%   hexagonal
%   A hexagonal network layout with up to three rings of BSs. The first BS is at coordinates [0,0].
%   The first BS of each ring is placed in the (north)-east of BS1. Additional BSs are added in
%   mathematical positive sense (counter-clockwise). Default BS height is 25 m.
%
%   regular
%   Same as hexagonal. However, each BS has three sectors. The number of sites can be 1, 7, 19 or
%   37 - resulting in 3, 21, 57 or 111 sectors, respectively. Sector orientations are 30, 150 and
%   -90 degrees in mathematical sense.
%
%   regular6
%   Same as hexagonal, but with default settings of 6 sectors per site and sector orientations 0,
%   60, 120, 180, 240 and 300 degree.
%
%   indoor
%   3GPP 38.901 indoor scenario. The number of sites can be given by a two-element array [ N, M ],
%   where N denotes the number of BSs in y-direction and m in x-direction. Default BS height is 3
%   m.
%
%   random
%   Randomly places base stations. The input parameter ISD describes the radius of the layout in
%   [m].
%
% Input:
%   layout_type
%   The layout type (string)
%
%   no_sites
%   The number of BS sites in the layout.
%
%   isd
%   The inter-site distance in [m]
%
%   h_arrayant
%   The array antenna object for each sector.
%
%   no_sectors
%   The number of sectors per site (default: 1)
%
%   sec_orientation
%   The orientation offset of the first sector in [deg]. (default: 0 deg)
%
% Output:
%   h_layout
%   The generated layout
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

if ~exist( 'h_arrayant','var' ) || isempty( h_arrayant )
    h_array = qd_arrayant('omni');
    h_arrayant = h_array;
elseif ~isa( h_arrayant ,'qd_arrayant') || ~isscalar( h_arrayant )
    error('The tx-antenna must be a scalar qd_arrayant object.');
else
    h_array = h_arrayant(1,1).copy;
end
if ~exist( 'no_sectors','var' ) || isempty( no_sectors )      % Number of sectors
    no_sectors = 1;
end
if ~exist( 'sec_orientation','var' ) || isempty( sec_orientation )      % First 3ector orientation in [deg]
    sec_orientation = 0;
end

% Duplicate antenna elements for each sector
no_el = h_array.no_elements;
for n = 2 : no_sectors
    for m = 1:no_el
         h_array.copy_element( m, (n-1)*no_el+m );
    end
end

% Rotate antenna elements
for n = 1 : no_sectors
    rot = sec_orientation + (n-1) * 360 / no_sectors;
    rot( rot>180 ) = rot( rot>180 ) - 360;
    h_array.rotate_pattern( rot, 'z', (n-1)*no_el+1 : n*no_el );
end

switch layout_type
    
    case 'hexagonal'
        
        if ~exist( 'no_sites','var' ) || isempty( no_sites )
            no_sites = 7;
        elseif no_sites > 37
           error('Maximal 37 BS sites are supported.') 
        end
        if ~exist( 'isd','var' ) || isempty( isd )
            isd = 500;
        end
        
        % Set the positions of the BS sites
        tx_position = zeros( 3,37 );
        tmp1 = 30+(0:5)*60;
        tmp1 = exp(1j*tmp1*pi/180);
        tx_position(1,2:7) = real( tmp1 );
        tx_position(2,2:7) = imag( tmp1 );
        
        tmp21 = 2*tmp1;
        tx_position(1,9:2:19) = real( tmp21 );
        tx_position(2,9:2:19) = imag( tmp21 );
        dist = real(tmp21(1));
        tmp22 = (0:5)*60;
        tmp22 = exp(1j*tmp22*pi/180)*dist;
        tx_position(1,8:2:18) = real( tmp22 );
        tx_position(2,8:2:18) = imag( tmp22 );
        
        tmp31 = 3*tmp1;
        tx_position(1,22:3:37) = real( tmp31 );
        tx_position(2,22:3:37) = imag( tmp31 );
        
        tmp32 = real(tmp31(1)) - 1j*0.5;
        tmp32 = tmp32.*exp( 1j*(0:5)*60*pi/180 );
        tx_position(1,20:3:35) = real( tmp32 );
        tx_position(2,20:3:35) = imag( tmp32 );
        
        tmp33 = real(tmp31(1)) + 1j*0.5;
        tmp33 = tmp33.*exp( 1j*(0:5)*60*pi/180 );
        tx_position(1,21:3:36) = real( tmp33 );
        tx_position(2,21:3:36) = imag( tmp33 );

        tx_position = tx_position * isd;
        tx_position(3,:) = 25;
        
        h_layout = qd_layout;
        h_layout.no_tx = no_sites;
        
        h_layout.tx_position = tx_position(:,1:no_sites);
        h_layout.tx_array = h_array;
        
    case 'regular'
        h_layout = qd_layout.generate('hexagonal',no_sites, isd, h_arrayant, 3, 30);
            
    case 'regular6'
        h_layout = qd_layout.generate('hexagonal',no_sites, isd, h_arrayant, 6, 0);

    case 'indoor'
        % 3GPP 38.901 indoor scenario (See: 3GPP TR 38.901 V14.1.0, Figure 7.2-1, pp21 )
        
        if ~exist( 'no_sites','var' ) || isempty( no_sites )      % Number od BS positions
            no_sites = [1,8];
        elseif numel( no_sites ) == 1
            no_sites = [ 1 , no_sites ];
        end
        if ~exist( 'isd','var' ) || isempty( isd )      % ISD
            isd = 20;
        end

        pos = zeros( 3, no_sites(1), no_sites(2) );
        for n = 1 : no_sites(1)
            for m = 1 : no_sites(2)
                pos(2,n,m) = (n-1)*isd;
                pos(1,n,m) = (m-1)*isd;
            end
        end
        pos = reshape( pos,3,[] );
        pos(1,:) = pos(1,:) - mean( pos(1,:) );
        pos(2,:) = -(pos(2,:) - mean( pos(2,:) ));
        pos(3,:) = 3;

        % Set BS positions 
        h_layout = qd_layout;
        h_layout.no_tx = prod( no_sites );
        h_layout.tx_position = pos;
        h_layout.tx_array = h_array;

    case 'random'
        
        if ~exist( 'no_sites','var' ) || isempty( no_sites )
            no_sites = 1;
        end
        if ~exist( 'isd','var' ) || isempty( isd )
            isd = 500; % m
        elseif ~( isnumeric(isd) && isreal(isd) && all( size(isd) == 1 ) && isd > 0 )
            error('??? "isd" must be a real scalar > 0');
        end
        
        h_layout = qd_layout;
        h_layout.no_tx = no_sites;
        
        tx_position_new = h_layout.tx_position;
        for n = 1:h_layout.no_tx
            a = rand*isd * exp( 2*pi*1j * rand );
            b = rand * (40 - 20) + 20;
            
            tx_position_new(1,n) = real(a);
            tx_position_new(2,n) = imag(a);
            tx_position_new(3,n) = b;
        end
        
        h_layout.tx_position = tx_position_new;
        h_layout.tx_array = h_array;
    
    otherwise
        error('Layout type is not supported.')
end

h_layout.name = layout_type;
end
