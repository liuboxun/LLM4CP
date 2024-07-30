function copy_element( h_qd_arrayant, i_source, i_target )
%COPY_ELEMENT Creates a copy of an antenna element
%
% Calling object:
%   Single object
%
%
% Input:
%   i_source
%   Index of the array object that should be copied. The value must be scalar, integer and greater
%   than 0 and it can not exceed the array size.
%
%   i_target
%   Target can be a scalar or vector with elements > 0.
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
   error('QuaDRiGa:qd_arrayant:copy_element','copy_element not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

% Parse input arguments
if ~( size(i_source,1) == 1 && isnumeric(i_source) && all(size(i_source) == [1 1]) ...
        &&  all( mod(i_source,1)==0 ) && min(i_source) > 0 && max(i_source) <= h_qd_arrayant.no_elements )
    error('??? "i_source" must be scalar, integer > 0 and can not exceed the number of antenna elements')
end

if ~( size(i_target,1) == 1 && isnumeric(i_target) ...
        &&  all( mod(i_target,1)==0 ) && min(i_target) > 0)
    error('??? "i_target" must be integer > 0')
end

if i_target > h_qd_arrayant.no_elements
    h_qd_arrayant.no_elements = max(i_target);
end

% Copy the data from i_source to i_target
i_target = setdiff(i_target,i_source);
for n = 1 : numel(i_target)
    h_qd_arrayant.element_position(:,i_target(n)) = h_qd_arrayant.element_position(:,i_source);
    h_qd_arrayant.Fa(:,:,i_target(n)) = h_qd_arrayant.Fa(:,:,i_source);
    h_qd_arrayant.Fb(:,:,i_target(n)) = h_qd_arrayant.Fb(:,:,i_source);
end

end
