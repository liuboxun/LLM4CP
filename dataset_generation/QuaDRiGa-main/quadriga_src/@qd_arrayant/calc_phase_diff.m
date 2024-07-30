function phase_diff = calc_phase_diff( h_arrayant, threshold )
%CALC_PHASE_DIFF Calculates the maximum phase difference of neighboring entries in the radiation pattern
%
% Calling object:
%   Single object
%
% Description:
%   The antenna radiation pattern is described by a field pattern containing the directive gain and
%   phase. This pattern is sampled at a fixed angular grid in polar-spheric coordinates. An
%   important step for the channel generation is the interpolation of the field patterns, which is
%   performed by qd_qrrayant.interpolate. However, since the entries in the field pattern are
%   generally complex-valued, different interpolation methods can be used: linear interpolation
%   interpolates the real and imaginary parts independently. Spherical linear interpolation
%   interpolates the phase using constant-speed motion along a unit-radius circle arc. The latter
%   is more accurate - especially when the phase differs significantly between the two entries. On
%   the other side, linear interpolation is more efficient and reasonably accurate for small
%   differences in phase (up to 15 degrees). In order to assess, which interpolation should be
%   used, the method calc_phase_diff calculate the maximum phase difference within the field
%   pattern. The interpolate function then applies the appropriate method.
%
% Input:
%   threshold
%   The maximum power difference in dB relative to the power in the direction of maximum radiation
%   for which the phase difference is calculated. This prevents entries in the field patter that
%   have very little directive power from being considered for the phase difference calculation.
%   Default value: 20 dB
%
% Output:
%   phase_diff
%   A vector containing the maximum phase difference in degree for each element of the array
%   antenna
%
%
% QuaDRiGa Copyright (C) 2011-2021
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

if numel( h_arrayant ) > 1
   error('QuaDRiGa:qd_arrayant:calc_phase_diff','"calc_phase_diff" is not definded for object arrays.');
else
    h_arrayant = h_arrayant(1,1); % workaround for octave
end

if ~exist( 'threshold','var' ) || isempty( threshold )
    threshold = 0.1;     % -20 dB
else
    threshold = sqrt(10^(-threshold/10));
end

no_elements = h_arrayant.no_elements;

if isa( h_arrayant.PFa, 'single' )
    R0 = single( 3.7433921e-23 );
    R1 = single(-0.9999);
    use_single = true;
else % double
    R0 = 2.222758749485077e-162;
    R1 = -0.99999999;
    use_single = false;
end

% Calculate the maximum amplitude per element over Fa and Fb
% Equal to: sqrt(max(max(abs(h_arrayant.PFa(:,:,1)).^2)))
[ ~,max_amplitude ] = h_arrayant.calc_gain;
max_amplitude = sqrt(10.^(max_amplitude/10));

if use_single
    phase_diff = zeros( no_elements,1,'single' );
else
    phase_diff = zeros( no_elements,1 );
end

for iF = 1:2
    if iF == 1
        gF = h_arrayant.PFa;
    else
        gF = h_arrayant.PFb;
    end
    
    if ~isreal( gF )
        
        gFr = real( gF ) + R0;
        gFi = imag( gF ) + R0;
        gP  = sqrt( gFr.^2 + gFi.^2 );
        gFr = gFr ./ gP + R0;
        gFi = gFi ./ gP + R0;
        
        % Azimuth direction
        gA  = gFr(:,1:end-1,:).*gFr(:,2:end,:) + gFi(:,1:end-1,:).*gFi(:,2:end,:);  % cos(phase)
        gB  = (gA<=1).*gA + (gA>1);                     % fix numeric precision problem when points have same phase
        gC  = (gB>R1).*gB + (gB<=R1).*R1;               % fix precision problem when points have 180 deg pahse diffrence
        gD  = acos( gC ) + R0;                          % phase between a and b (can be 0) --> fix by adding R0
        for n = 1 : no_elements
            mask = 0.5*(gP(:,1:end-1,n) + gP(:,2:end,n)) > max_amplitude(n)*threshold;
            if any( mask(:))
                omega_n = gD( :,:,n );
                omega_n( ~mask ) = 0;
                omega_n( omega_n > pi*0.995 ) = 0;
                phase_diff(n) = max( phase_diff(n), max(omega_n(:))*180/pi );
            end
        end
        
        % Elevation direction
        gA  = gFr(1:end-1,:,:).*gFr(2:end,:,:) + gFi(1:end-1,:,:).*gFi(2:end,:,:);  % cos(phase)
        gB  = (gA<=1).*gA + (gA>1);                     % fix numeric precision problem when points have same phase
        gC  = (gB>R1).*gB + (gB<=R1).*R1;               % fix precision problem when points have 180 deg pahse diffrence
        gD  = acos( gC ) + R0;                          % phase between a and b (can be 0) --> fix by adding R0
        for n = 1 : no_elements
            mask = 0.5*(gP(1:end-1,:,n) + gP(2:end,:,n)) > max_amplitude(n)*threshold;
            if any( mask(:))
                omega_n = gD( :,:,n );
                omega_n( ~mask ) = 0;
                omega_n( omega_n > pi*0.995 ) = 0;
                phase_diff(n) = max( phase_diff(n), max(omega_n(:))*180/pi );
            end
        end
    end
    
end

% Temporary storage
h_arrayant.Pphase_diff = phase_diff;

end
