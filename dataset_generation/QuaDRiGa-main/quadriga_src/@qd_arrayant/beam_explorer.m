function beam_explorer( h_arrayant, Jp )
%BEAM_EXPLORER Creates an interactive plot of the beam-forming capabilities of an array antenna
%
% Calling object:
%   Single object
%
% Description:
%   When applying maximum-ratio transmission (MRT) to calculate the coupling weights of an array
%   antenna, it is possible to direct a beam towards a given direction. However, the antenna
%   geometry (i.e. the positions and the orientations of the individual elements) and the shape of
%   the individual element-patterns will determine the overall shape of the beam and the existence
%   and magnitude of so-called sidelobes.  This method creates an interactive plot that uses the
%   mouse pointer position to determine the target direction. The y-axis corresponds to the
%   elevation direction and the x-axis corresponds to the azimuth direction relative to the local
%   antenna coordinate system. Then, the method calculates the MRT-weights that direct the beam
%   towards this position and applies it to the antenna pattern. The plot is then updated in real-
%   time to visualize the radiated power using MRT beamforming. The maximum is normalized to 1, the
%   minimum is normalized to 0. Ideally, the array antenna creates a single narrow beam that
%   coincides exactly with the target direction, i.e. the maximum is always under the mouse
%   pointer. However, design limitations (i.e., using planar or circular arrays, number of
%   elements, etc.) will either lead to unwanted side-lobes or a widening of the main lobe. By
%   clicking the left mouse button, the animation is paused and it is possible to use the data-
%   pointer to read the values from the plot.  
%
%   Note: It is important to set the correct center frequency in the array object.
%
% Input:
%   Jp
%   The polarization (Jones-vector) of the probe antenna, Default [ 1 ; 0 ]
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

global b Wa Wb im

if ~exist('Jp','var') || isempty( Jp )
   Jp = [1;0];
end

if numel( h_arrayant ) > 1
   error('QuaDRiGa:qd_arrayant:beam_explorer','beam_explorer is not definded for object arrays.');
else
    h_arrayant = h_arrayant(1,1); % workaround for octave
end

tmp = h_arrayant.copy;
tmp.combine_pattern;
tmp.PFa = single( tmp.PFa );
tmp.PFb = single( tmp.PFb );
tmp.Pcoupling = single( tmp.Pcoupling );
tmp.Pelement_position = single( tmp.Pelement_position );
tmp.elevation_grid = single( tmp.elevation_grid );
tmp.azimuth_grid = single( tmp.azimuth_grid );

Wa = reshape( tmp.Fa, [], h_arrayant.no_elements );
Wb = reshape( tmp.Fb, [], h_arrayant.no_elements );

b = qd_builder('LOSonly');
b.simpar(1,1).center_frequency = h_arrayant.center_frequency;
b.simpar(1,1).show_progress_bars = 0;
b.tx_array = tmp;
b.rx_array = qd_arrayant('omni');
b.rx_array(1,1).Fa(:) = Jp(1);
b.rx_array(1,1).Fb(:) = Jp(2);
b.tx_position = [0;0;0];
b.rx_positions = [100;0;0];
b.gen_parameters;

if b.NumClusters == 1
    c = b.get_los_channels('single','coeff');
else
    c = b.get_channels;
    c = single( sum(c.coeff,3) );
end
cf = permute( c, [2,3,1] ) ;

A = abs( Wa * conj(cf) ).^2 + abs( Wb * conj(cf) ).^2;
A = A - min(A);
A = A./max(A);
A = reshape( A , h_arrayant.no_el , h_arrayant.no_az , [] );

im = imagesc(h_arrayant.azimuth_grid*180/pi,h_arrayant.elevation_grid*180/pi,A);
set(gca,'Ydir','Normal');
set(gca,'clim',[0 1]);
set(gca,'xlim',[-180 180]);
set(gca,'ylim',[-90 90]);
set(gcf,'Pointer','crosshair')
set(gcf,'WindowButtonMotionFcn', @mouseMove);
set(gcf,'WindowButtonDownFcn', @mouseClick);

end

function mouseMove(object, eventdata)

global b Wa Wb im cf az el

C = get(gca, 'CurrentPoint');
az = C(1,1);
el = C(1,2);

b.rx_positions = [ cosd(az)*cosd(el) ; sind(az)*cosd(el) ; sind(el) ]*100;
c = b.get_los_channels('single','coeff');
cf = permute( c, [2,3,1] ) ;

A = abs( Wa * conj(cf) ).^2 + abs( Wb * conj(cf) ).^2;
A = A - min(A);
A = A./max(A);
A = reshape( A , b.tx_array.no_el , b.tx_array.no_az , [] );

set( im, 'CData', A);

title(gca, ['(az,el) = (', num2str(az), ', ',num2str(el), ')']);

end

function mouseClick(object, eventdata)

global b cf az el

persistent stp
if isempty( stp )
   stp = false;
end

if stp
    set(gcf,'WindowButtonMotionFcn', @mouseMove);
    stp = ~stp;
else
    set(gcf,'WindowButtonMotionFcn', '');
    stp = ~stp;
    
    tmp = copy( b.tx_array );
    tmp.coupling = conj(cf);
    tmp.combine_pattern;
    [ gain_dBi, nomalied_power ] = calc_gain( tmp );
    [ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = calc_beamwidth( tmp );
        
    disp(['Azimuth target angle:      ',num2str(az),' deg']);
    disp(['Azimuth beam direction:    ',num2str(az_point_ang),' deg']);
    disp(['Azimuth FWHM beam width:   ',num2str(beamwidth_az),' deg']);
    disp(' ');
    disp(['Elevation target angle:    ',num2str(el),' deg']);
    disp(['Elevation beam direction:  ',num2str(el_point_ang),' deg']);
    disp(['Elevation FWHM beam width: ',num2str(beamwidth_el),' deg']);
    disp(' ');
    disp(['Main beam power:           ',num2str(nomalied_power),' dBi']);
    disp(['Normalized gain:           ',num2str(gain_dBi),' dBi']);
    disp(' ');
end

end
