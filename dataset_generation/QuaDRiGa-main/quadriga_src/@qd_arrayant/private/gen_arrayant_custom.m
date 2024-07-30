function [ h_qd_arrayant, par ] = gen_arrayant_custom( phi_3dB, theta_3dB, rear_gain )
%GEN_ARRAYANT_CUSTOM
%
%   An antenna with a custom gain in elevation and azimuth. The values A,B,C and D for the
%   parametric antenna are returned.
%      * phi_3dB - 3dB beam width in azimuth direction
%      * theta_3dB - 3dB beam width in elevation direction
%      * rear_gain - Isotropic gain (linear scale) at the back of the antenna
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

% Set default variables
if ~exist('phi_3dB','var') || isempty( phi_3dB )
    phi_3dB = 120;
end

if ~exist('theta_3dB','var') || isempty( theta_3dB )
    theta_3dB = 120;
end

if ~exist('rear_gain','var') || isempty( rear_gain )
    rear_gain = 0.1;
end

if ~( size(phi_3dB,1) == 1 && isnumeric(phi_3dB) && isreal(phi_3dB) &&...
        all(size(phi_3dB) == [1 1]) )
    error('Azimuth HPBW (phi_3dB) has invalid value.')
end

if ~( size(theta_3dB,1) == 1 && isnumeric(theta_3dB) && isreal(theta_3dB) &&...
        all(size(theta_3dB) == [1 1]) )
    error('Elevation HPBW (theta_3dB) has invalid value.')
end

if ~( size(rear_gain,1) == 1 && isnumeric(rear_gain) && isreal(rear_gain) && ...
        all(size(rear_gain) == [1 1]) )
    error('Front-to-back ratio (rear_gain) has invalid value.')
end

% Generate omni antenna as default
h_qd_arrayant = qd_arrayant.generate('omni');

par.A = 0;
par.B = 0;
par.C = 0;
par.D = 0;

% Calculate the azimuth response
phi = h_qd_arrayant.azimuth_grid;
%ind = find(phi/pi*180 >= phi_3dB/2, 1);

a   = 1;        % Initial angle
dm  = 0.5;      % Step size
x   = inf;
delta = Inf;
ddir = +1;
lp = 1;
while lp < 5000 && delta > 1e-7
    if lp > 1
        an = a + ddir * dm;
        delta = abs(a-an);
    else
        an = a;
    end
    
    C = rear_gain + (1 - rear_gain) * exp(-an * phi.^2);
    xn = abs( qf.interp( phi, [], C, phi_3dB*pi/360, [], 1 ) - 0.5 );
    %xn = abs(C(ind) - 0.5);
    
    if xn < x
        a = an;
        x = xn;
    else
        ddir = -ddir;
        dm = 0.382 * dm;
    end
    lp = lp + 1;
end
C = exp(-an * phi.^2);
par.D = an;

% Calculate the elevation response
theta = h_qd_arrayant.elevation_grid;
%ind = find(theta/pi*180 >= theta_3dB/2, 1);

a   = 1;        % Initial angle
dm  = 0.5;      % Step size
x   = inf;
delta = Inf;
ddir = +1;
lp = 1;
while lp < 5000 && delta > 1e-7
    if lp > 1
        an = a + ddir * dm;
        delta = abs(a-an);
    else
        an = a;
    end
    
    D = cos(theta).^an;
    xn = abs( qf.interp( theta, [], D, theta_3dB*pi/360, [], 1 ) - 0.5 );
    %xn = abs(D(ind) - 0.5);
    
    if xn < x
        a = an;
        x = xn;
    else
        ddir = -ddir;
        dm = 0.382 * dm;
    end
    lp = lp + 1;
end
D = cos(theta).^an;
par.C = an;

par.B = rear_gain;

P = zeros(181,361);
for a = 1:181
    for b = 1:361
        P(a, b) = D(a) * C(b);
    end
end
P = rear_gain + (1-rear_gain)*P;

E_theta =  sqrt(P);

% Assign antena pattern and normalize the gain
h_qd_arrayant.Fa = E_theta;
h_qd_arrayant.Fb = zeros(h_qd_arrayant.no_el, h_qd_arrayant.no_az);
h_qd_arrayant.normalize_gain;

end
