function [ directivity_dBi, gain_dBi ] = calc_gain( h_qd_arrayant, i_element )
%CALC_GAIN Calculates the gain in dBi of the array antenna
%
% Calling object:
%   Single object
%
% Input:
%   i_element
%   A list of element indices.
%
% Output:
%   directivity_dBi
%   Normalized gain of the antenna in dBi.
%
%   gain_dBi
%   Maximum gain of the antenna in dBi (gain = directivity - losses)
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

if nargin == 1
    i_element = 1:h_qd_arrayant.no_elements;
end

if ~(any(size(i_element) == 1) && isnumeric(i_element) ...
        && isreal(i_element) && all(mod(i_element, 1) == 0) && all(i_element > 0))
    error('??? "i_element" must be integer and > 0')
elseif any(i_element > h_qd_arrayant.no_elements)
    error('??? "i_element" exceeds "no_elements"')
end

if isa( h_qd_arrayant.PFa, 'single' )
    use_single = true;
else % double
    use_single = false;
end

% Calculate azimuth weights
% These weights are needed when there is an non-uniform grid in the azimuth sample points, e.g. for
% parabolic antennas
az = h_qd_arrayant.azimuth_grid;
az = angle( exp(1j*az));                % map to [-pi ... 0 ... pi ]
x = [az(end)-2*pi,az];
y = [az,az(1)+2*pi];
waz = diff(x)+diff(y);
waz = waz./sum(waz);

% Calculate elevation weights
% These weights are needed when there is an non-uniform grid in the angle sample points, e.g. for
% parabolic antennas
el = h_qd_arrayant.elevation_grid;
nel = numel( el );
if use_single
    wel = zeros( nel,1,'single' );
else
    wel = zeros( nel,1 );
end
for n = 1 : nel
    if n == 1
        st = -pi/2; 
        en = 0.5*( el(1) + el(2) );
    elseif n == nel
        st = 0.5*( el(n-1) + el(n) );
        en = pi/2;
    else
        st = 0.5*( el(n-1) + el(n) ); 
        en = 0.5*( el(n) + el(n+1) ); 
    end
    % The average cosine of the covered angle range
    avg_cos = sum(cos(st:(en-st)*0.04999999:en))/21;
    wel(n) = avg_cos * (en-st)/pi;
end

% Combined azimuth and elevation weights
w = wel * waz ;
w = w./sum(w(:));

if use_single
    directivity_dBi = zeros(numel(i_element),1,'single');
    gain_dBi = zeros(numel(i_element),1,'single');
else
    directivity_dBi = zeros(numel(i_element),1);
    gain_dBi = zeros(numel(i_element),1);
end

for n = 1 : numel(i_element)
    
    % Read the qd_arrayant elements
    Fa = h_qd_arrayant.PFa(:, :, i_element(n));
    Fb = h_qd_arrayant.PFb(:, :, i_element(n));
    
    % calculate radiation power pattern
    P = abs(Fa).^2 + abs(Fb).^2;
    
    % Normalize by max value
    P_max = max( P(:) );
    P = P ./ P_max;
    
    % Calculate Gain
    gain_lin    = 1 ./ sum(P(:).*w(:));
    
    directivity_dBi(n) = 10*log10(gain_lin);
    gain_dBi(n) = 10*log10(P_max);
end

end

