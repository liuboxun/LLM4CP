function [ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] =...
    calc_beamwidth( h_qd_arrayant, i_element, thres_dB )
%CALC_BEAMWIDTH Calculates the beam width for each antenna element in [deg]
%
% Calling object:
%   Single object
%
% Description:
%   This method calculates the beamwidth in azimuth and elevation direction as well as the pointing
%   angles of each element of the array antenna. Interpolation is used to achieve a higher
%   precision as provided by the sampling angle grid.
%
% Input:
%   i_element
%   A list of element indices. Default: 1 ... no_elements
%
%   thres_dB
%   The threshold in dB (Default: 3 dB, equivalent to FWHM)
%
% Output:
%   beamwidth_az
%   The azimuth beamwidth for each element in [deg]
%
%   beamwidth_el
%   The elevation beamwidth for each element in [deg]
%
%   az_point_ang
%   The azimuth pointing angle for the main beam in [deg]
%
%   el_point_ang
%   The elevation pointing angle for the main beam in [deg]
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

if numel( h_qd_arrayant ) > 1 
   error('QuaDRiGa:qd_arrayant:calc_gain','calc_gain not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

if ~exist('i_element','var') || isempty(i_element)
    i_element = 1:h_qd_arrayant.no_elements;
elseif ~(any(size(i_element) == 1) && isnumeric(i_element) ...
        && isreal(i_element) && all(mod(i_element, 1) == 0) && all(i_element > 0))
    error('??? "i_element" must be integer and > 0')
elseif any(i_element > h_qd_arrayant.no_elements)
    error('??? "i_element" exceeds "no_elements"')
end

if ~exist('thres_dB','var') || isempty(thres_dB)
    thres_dB = 3;
end

el = h_qd_arrayant.elevation_grid;
nEl = numel(el);
tmp = (el(end)-el(1))/(100*nEl);
eli = [ el(1) : tmp : el(end)-tmp/2, el(end) ];

az = h_qd_arrayant.azimuth_grid;
nAz = numel(az);
tmp = (az(end)-az(1))/(100*nAz);
azi = [ az(1) : tmp : az(end)-tmp/2, az(end) ];

beamwidth_az = zeros(numel(i_element),1);
beamwidth_el = zeros(numel(i_element),1);
az_point_ang = zeros(numel(i_element),1);
el_point_ang = zeros(numel(i_element),1);

for n = 1 : numel(i_element)
    
    % Read the qd_arrayant elements
    Fa = h_qd_arrayant.Fa(:, :, i_element(n));
    Fb = h_qd_arrayant.Fb(:, :, i_element(n));
    
    % calculate radiation power pattern and normalize it
    P = abs(Fa).^2 + abs(Fb).^2;
    P_max = max( P(:) );
    P = P ./ P_max;
    
    % Find the elevation angle ant the azimuth angle with the maximum values
    [~,ii] = max(P(:));
    [ iEl, iAz ] = qf.qind2sub( [nEl,nAz],ii );
    
    % Calculate azimuth beamwidth
    Pi = abs(qf.interp( az, 0, Fa(iEl,:), azi )).^2 +...
        abs(qf.interp( az, 0, Fb(iEl,:), azi )).^2;
    [P_max,iM] = max(Pi);
    Pi = Pi ./ P_max;
    
    iS = find(Pi > 10^(-0.1*thres_dB),1);
    iL = find(Pi > 10^(-0.1*thres_dB),1,'last');
    beamwidth_az(n,1) = (azi(iL)-azi(iS))*180/pi;
    az_point_ang(n,1) = azi(iM)*180/pi;
    
    % Calculate elevation beamwidth
    Pi = abs(qf.interp( el, 0, Fa(:,iAz).', eli )).^2 +...
        abs(qf.interp( el, 0, Fb(:,iAz).', eli )).^2;
    [P_max,iM] = max(Pi);
    Pi = Pi ./ P_max;
    
    iS = find(Pi > 10^(-0.1*thres_dB),1);
    iL = find(Pi > 10^(-0.1*thres_dB),1,'last');
    beamwidth_el(n,1) = (eli(iL)-eli(iS))*180/pi;
    el_point_ang(n,1) = eli(iM)*180/pi;
end

end
