function h_qd_arrayant = gen_arrayant_parabolic( r, f, p_min, pol, no_beams, beam_separation, gain )
%GEN_ARRAYANT_PARABOLIC
%
%   An ideal parabolic reflector antenna with input parameters:
%
%      * r - Radius of the antenna aperture in [meters]
%      * f - Center frequency in [Hz]
%      * p_min - Min. sidelobe power relative to directivity in [dB] (default: -40 dB)
%      * pol - Polarization indicator
%           1. vertical (E-theta) polarization (default)
%           2. horizontal (E-phi) polarization
%           3. LHCP
%           4. RHCP
%           5. dual-polarized two-port antenna (LHCP,RHCP)
%      * no_beams - number of beams for a multibeam antenna (hexagonal layout)
%      * beam_separation - beam separation in [deg], default is the FWHM
%      * gain - Satellite Tx max Gain in (dBi)
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

% Set inputs
if ~exist('r','var') || isempty( r )
    r = 1;
end

if ~exist('f','var') || isempty( f )
    f = 4e9;
end

if ~exist('p_min','var') || isempty( p_min )
    p_min = -40;
end

if ~exist('pol','var') || isempty( pol )
    pol = 1;
end

if ~exist('no_beams','var') || isempty( no_beams )
    no_beams = 1;
end

if ~exist('gain','var') || isempty( gain )
    gain = [];
end

if no_beams == 1
    % Create single parabpolic antenna
    
    % Wave number
    k = 2 * r * pi * f / qd_simulation_parameters.speed_of_light;
    
    % Calc pattern vs az. angle at low resolution
    azimuth_grid = 0:0.001:90;                  % AZ-Scan
    z = k .* sin(azimuth_grid*pi/180);
    e = 2 * besselj(1, z ) ./ z;                % Electric field
    e(1) = 1;
    p = 10*log10(abs(e).^2);                    % Power
    
    dp = abs(diff(p));
    
    ind_e = find( dp > 5, 1,'last' ) + 1;       % The last Null
    ind_m = find( p > p_min , 1, 'last' );      % The last time where the pattern is > p_min
    if ind_m < ind_e
        dp(1:ind_m) = 0;
        ind_m = find( dp > 5, 1,'first' ) + 1;  % The first Null after -p_min
    else
        ind_m = ind_e;
    end
    az_max = azimuth_grid(ind_m);
    
    % Use the SAGE algorithm to find the true az_max
    a   = az_max;       % Initial value
    x   = p(ind_m);
    dm  = 0.001;        % Step size
    delta = Inf; ddir = +1; lp = 2;
    while lp<100 && delta > 1e-9
        if lp>1; an = a + ddir * dm; delta = abs(a-an); else an = a; end
        
        z = k .* sin(an*pi/180);
        e = 2 * besselj(1, z ) ./ z;    % Electric field
        xn = 10*log10(abs(e).^2);       % Power
        
        if xn < x; a = an; x = xn; else ddir = -ddir; dm = 0.2 * dm; end
        lp = lp + 1;
    end
    az_max = an;
    
    % Calculate 2D pattern
    no_steps = round( az_max*k/10 );            % Number of steps
    step_size = az_max/no_steps;                % Step size
    
    % Determine the reduced grid
    gridA = (-az_max : step_size : az_max)*pi/180;
    
    % Calculate pattern in 2D
    cosG = cos(gridA);
    omega = acos( cosG' * cosG );               % Angles between the x-axis and the point on the sphere
    
    z = k .* sin(omega);
    e = 2 * besselj(1, z ) ./ z;                % Electric field
    
    ii = omega < 0.5*step_size*pi/180;          % Value at the x-axis
    e(ii) = 1;
    
    e( omega>az_max*pi/180 ) = 0;               % Cutoff
    
    % Assemble Quadriga array antenna object
    h_qd_arrayant = qd_arrayant([]);
    h_qd_arrayant.name = 'parabolic';
    h_qd_arrayant.center_frequency = f;
    h_qd_arrayant.set_grid( gridA, gridA, 0 );
    h_qd_arrayant.Fa = e;
    
    switch pol
        case 1
            % Nothing to be done.
        case 2
            h_qd_arrayant.rotate_pattern(90,'x',1,2);
        case 3
            h_qd_arrayant.copy_element(1,2);
            h_qd_arrayant.rotate_pattern(90,'x',2,2);
            h_qd_arrayant.coupling = 1/sqrt(2) * [1;1j];
            h_qd_arrayant.combine_pattern;
        case 4
            h_qd_arrayant.copy_element(1,2);
            h_qd_arrayant.rotate_pattern(90,'x',2,2);
            h_qd_arrayant.coupling = 1/sqrt(2) * [1 ; -1j];
            h_qd_arrayant.combine_pattern;
        case 5
            h_qd_arrayant.copy_element(1,2);
            h_qd_arrayant.rotate_pattern(90,'x',2,2);
            h_qd_arrayant.coupling = 1/sqrt(2) * [1 1 ; 1j -1j];
            h_qd_arrayant.combine_pattern;
        otherwise
            error('Polarization indicator must be a scalar number from 1 to 5');
    end

    if ~isempty( gain )
        h_qd_arrayant.normalize_gain([],gain);
    else
        h_qd_arrayant.normalize_gain;
    end
    
elseif no_beams > 1
    % Create multibeam antenna
      
    % Recursive call to generate a single beam antenna
    h_qd_arrayant = gen_arrayant_parabolic( r, f, p_min, pol, 1, [], gain );
    single( h_qd_arrayant );    % 30% faster
    
    if ~exist('beam_separation','var') || isempty( beam_separation )
        beam_separation = calc_beamwidth(h_qd_arrayant,1);
    end
    beam_separation = beam_separation*pi/180;
    
    % Enlarge the angle sampling grid to make space for the additional beams
    az_grid = h_qd_arrayant.azimuth_grid;
    el_grid = h_qd_arrayant.elevation_grid;
    dAz = mean(diff(az_grid));
    dEl = mean(diff(el_grid));
    if no_beams == 2
        el_grid = [el_grid, el_grid(end) + (1:ceil(beam_separation/dEl))*dEl ];
        
    elseif no_beams == 3
        el_grid = [el_grid, el_grid(end) + (1:ceil(beam_separation/dEl))*dEl ];
        az_grid = [az_grid, az_grid(end) + (1:ceil(beam_separation/dAz))*dAz ];
        
    elseif no_beams < 5.5
        el_grid = [ el_grid(1) + (-ceil(beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(beam_separation/dEl))*dEl  ];
        az_grid = [az_grid, az_grid(end) + (1:ceil(beam_separation/dAz))*dAz ];
        
    elseif no_beams < 7.5
        el_grid = [ el_grid(1) + (-ceil(beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(beam_separation/dAz))*dAz ];
        
    elseif no_beams < 10.5
        el_grid = [ el_grid(1) + (-ceil(beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(2*beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(beam_separation/dAz))*dAz ];
        
    elseif no_beams < 13.5
        el_grid = [ el_grid(1) + (-ceil(beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(2*beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(2*beam_separation/dAz))*dAz ];    

    elseif no_beams < 16.5
        el_grid = [ el_grid(1) + (-ceil(2*beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(2*beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(2*beam_separation/dAz))*dAz ];
        
    elseif no_beams < 19.5
        el_grid = [ el_grid(1) + (-ceil(2*beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(2*beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(2*beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(2*beam_separation/dAz))*dAz ];
        
    elseif no_beams < 21.5
        el_grid = [ el_grid(1) + (-ceil(2*beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(2*beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(3*beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(2*beam_separation/dAz))*dAz ];
        
    elseif no_beams < 23.5
        el_grid = [ el_grid(1) + (-ceil(2*beam_separation/dEl):-1)*dEl,...
            el_grid, el_grid(end) + (1:ceil(2*beam_separation/dEl))*dEl  ];
        az_grid = [ az_grid(1) + (-ceil(3*beam_separation/dAz):-1)*dAz,...
            az_grid, az_grid(end) + (1:ceil(3*beam_separation/dAz))*dAz ];
        
    else
        error('Number of beams must be <= 23');
    end
    
    % Boundary conditions
    iEl = el_grid > pi/2 | el_grid < -pi/2;
    iAz = az_grid > pi | az_grid < -pi;
    if any( iEl ) || any( iAz )
        tmp = pi/ceil(pi/dAz);
        az_grid = [ -pi : tmp : pi-tmp/2, pi ];
        el_grid = el_grid(~iEl);
    end
    
    % Update the grid in "h_qd_arrayant"
    set_grid( h_qd_arrayant, az_grid, el_grid );
    
    % Make a copy of the original parabolic antenna
    if pol == 5 % Dual-polarization
        a  = sub_array( h_qd_arrayant,2 );  % RHCP
        aL = sub_array( h_qd_arrayant,1 );  % LHCP
        h_qd_arrayant = copy(a);            % Use RHCP for beam 1
    else
        a = copy( h_qd_arrayant );
    end
        
    % Generate the beam offset angles
    beam_offset = zeros( 2,23 );
    tmp1 = 90-(0:5)*60;
    tmp1 = exp(1j*tmp1*pi/180);
    beam_offset(1,2:7) = real( tmp1 );
    beam_offset(2,2:7) = imag( tmp1 );
    tmp21 = 2*tmp1;
    beam_offset(1,9:2:19) = real( tmp21 );
    beam_offset(2,9:2:19) = imag( tmp21 );
    dist = real(tmp21(2));
    tmp22 = 120-(0:5)*60;
    tmp22 = exp(1j*tmp22*pi/180)*dist;
    beam_offset(1,8:2:18) = real( tmp22 );
    beam_offset(2,8:2:18) = imag( tmp22 );
    beam_offset(:,20) = [ -1.5*sqrt(3) ; 0.5  ];
    beam_offset(:,21) = [ -1.5*sqrt(3) ; -0.5  ];
    beam_offset(:,22) = [ 1.5*sqrt(3) ; 0.5  ];
    beam_offset(:,23) = [ 1.5*sqrt(3) ; -0.5  ];
    beam_offset = beam_offset * beam_separation; % [rad]
    
    warning('off','QuaDRiGa:qd_arrayant:append_array');
    % Construct multi-beam antenna
    for n = 2:no_beams
        if pol == 5 && any( n == [2,3,5,6,8,12,14,18,20,23] )
            b = copy(aL);   % Use LHCP pattern in dual-pol mode
        else
            b = copy(a);
        end
        if abs(beam_offset(2,n)) > 1e-8 && abs(beam_offset(1,n)) > 1e-8
            b.rotate_pattern( -[0;beam_offset([2,1],n)*180/pi],'xyz',[],3);
        elseif abs(beam_offset(2,n)) > 1e-8 
            b.rotate_pattern( -beam_offset(2,n)*180/pi,'y',[],3);
        elseif abs(beam_offset(1,n)) > 1e-8
            b.rotate_pattern( -beam_offset(1,n)*180/pi,'z',[],3);
        end
        append_array(h_qd_arrayant, b);
        
     end
    warning('on','QuaDRiGa:qd_arrayant:append_array');
    double( h_qd_arrayant );
else
    error('Number of beams must be >= 1');
end

end
