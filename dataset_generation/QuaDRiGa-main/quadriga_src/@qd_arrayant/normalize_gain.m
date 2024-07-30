function gain_dBi = normalize_gain( h_qd_arrayant, i_element, gain )
%NORMALIZE_GAIN Normalizes all patterns to their gain
%
% Calling object:
%   Single object
%
%
% Input:
%   i_element
%   A list of elements for which the normalization is done. Default: All elements
%
%   gain
%   The gain that should be set in the pattern. If this variable is not given, the gain is
%   calculated from the pattern
%
% Output:
%   gain_dBi
%   Normalized gain of the antenna
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

if ~exist('i_element','var') || isempty( i_element )
    i_element = 1:h_qd_arrayant.no_elements;
end

if ~exist('gain','var') || isempty( gain )
    gain = NaN(1,numel(i_element));
elseif numel(gain) == 1
    gain = ones(1,numel(i_element))*gain;
elseif numel(gain) ~= numel(i_element)
    error('The number of gain values must either be 1 or match the numebr of elements.')
end

if ~(any(size(i_element) == 1) && isnumeric(i_element) ...
        && isreal(i_element) && all(mod(i_element, 1) == 0) && all(i_element > 0))
    error('??? "i_element" must be integer and > 0')
elseif any(i_element > h_qd_arrayant.no_elements)
    error('??? "i_element" exceed "no_elements"')
end

gain_dBi = zeros(numel(i_element),1);

for n = 1 : numel(i_element)
    
    % Calculate the gain using the calc-gain method
    [gain_dBi(n),P_max] = calc_gain( h_qd_arrayant, i_element(n) );
    
    gain_lin = 10.^(0.1*gain_dBi(n));
    P_max = 10.^(0.1*P_max);
    
    % Read the element patterns 
    Fa = h_qd_arrayant.Fa(:, :, i_element(n)); 
    Fb = h_qd_arrayant.Fb(:, :, i_element(n));
    
    if ~isnan( gain(n) )
        gain_lin = 10.^(0.1*gain(n));
    end
    
    % Normalize Patterns by their gain
    tmp = sqrt(gain_lin./P_max);
    h_qd_arrayant.Fa(:, :, i_element(n)) = Fa .* tmp;
    h_qd_arrayant.Fb(:, :, i_element(n)) = Fb .* tmp;
end

end
