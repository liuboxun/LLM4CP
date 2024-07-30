function a = sub_array( h_qd_arrayant, i_element )
%SUB_ARRAY Generates a sub-array with the given array indices
%
% Calling object:
%   Single object
%
% Description:
%   This function creates a copy of the given array with only the selected elements specified in
%   i_element.
%
% Input:
%   i_element
%   A list of element indices
%
% Output:
%   a
%   An arrayant object with the desired elements
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

if ~isempty( setdiff( i_element , 1:h_qd_arrayant.no_elements ) )
    error('The indices specified in i_element do not exist in the array.')
end

a = qd_arrayant( [] );

tmp = sprintf( '%d,',i_element );
a.name =  [ h_qd_arrayant.name,'; El. ',tmp(1:end-1) ];

a.center_frequency          = h_qd_arrayant.center_frequency;
a.elevation_grid            = h_qd_arrayant.elevation_grid;
a.azimuth_grid              = h_qd_arrayant.azimuth_grid;
a.no_elements               = numel( i_element );
a.element_position          = h_qd_arrayant.element_position( :,i_element );
a.Fa                        = h_qd_arrayant.Fa( :,:,i_element );
a.Fb                        = h_qd_arrayant.Fb( :,:,i_element );

if all( size( h_qd_arrayant.coupling ) == h_qd_arrayant.no_elements([1,1]) )
    a.coupling = h_qd_arrayant.coupling( i_element,i_element );
end

end
