function [ xprL, xprC, cprL, cprC ] = calc_xpr( xprmat )
%CALC_XPR Calculates the cross and co-polarization metrics from a given Jones matrix
%
% Calling object:
%   None (static method)
%
% Description:
%   QuaDRiGa uses Jones calculus to describe and model polarization effects. A polarized
%   electromagnetic wave is represented by a Jones vector, and interactions of that wave with the
%   environment (transmission through a medium, reflection, scattering, etc.) are represented by
%   Jones matrices. Suppose that a electromagnetic wave is traveling in the positive x-direction,
%   then the electric and magnetic fields E and H both lie in the plane "transverse" to the
%   direction of motion. Furthermore, H is determined from E by 90-degree rotation, so the
%   polarization of the wave can be determined by studying E. Thus, the Jones vector represents the
%   amplitude and phase of the electric field in the z and y directions. The Jones matrices are
%   operators that act on these Jones vectors.
%
% Input:
%   xprmat
%   An array of Jones matrices. Dimensions: [ 2, 2, n_path, n_snapshots ]
%
% Output:
%   xprL
%   The cross-polarization ratio (linear scale) for linear polarized waves Dimensions: [ n_path,
%   n_snapshots ] Suppose that the incoming wave is linearly polarized (either vertically with a
%   field component only in z-direction or horizontally with a field component only in y
%   direction), then xprL describes how much power is transferred from one polarization axis to the
%   other by the interaction, e.g., from V (z-axis) to H (y-axis) or vice-versa. Values can range
%   from 0 to infinity, where infinity means that the polarization is perfectly maintained and 0
%   means that all powers is transfered to the other polarization.
%
%   xprC
%   The cross-polarization ratio (linear scale) for circular polarized waves Dimensions: [ n_path,
%   n_snapshots ] Suppose that the incoming wave is circularly polarized (either LHCP or RHCP),
%   then xprC describes how much power is transferred from one polarization to the other.
%
%   cprL
%   The co-polarization ratio (linear scale) for linear polarized waves Dimensions: [ n_path,
%   n_snapshots ] If a linearly polarized wave interacts with a medium, the two polarization
%   components generally experience different reflection coefficients, depending on the material
%   properties and the angle of incidence. The co-polarization ratio describes the power-difference
%   between the two components after the scattering event.
%
%   cprC
%   The co-polarization ratio (linear scale) for circularly polarized waves Dimensions: [ n_path,
%   n_snapshots ] If a circularly polarized wave interacts with a medium, the two polarization
%   components generally experience different phase-shifts coefficients, depending on the material
%   properties and the angle of incidence. The co-polarization ratio describes the power-difference
%   between the two components after the scattering event.
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


if size(xprmat,1) == 2 && size(xprmat,2) == 2
    xprmat = reshape( xprmat, 4, size(xprmat,3), [] );
elseif size(xprmat,1) ~= 4
    error('QuaDRiGa:calc_xpr','Wrong format of the polarization transfer matrix.');
else
    xprmat = reshape( xprmat, 4, size(xprmat,2), [] );
end


% Linear cross-polarization ratio
xprL = 0.5 * abs( xprmat(1,:,:) ).^2 ./ abs( xprmat(2,:,:) ).^2 +...
    0.5 * abs( xprmat(4,:,:) ).^2 ./ abs( xprmat(3,:,:) ).^2;
xprL = permute( xprL, [3,2,1] );
xprL(isnan(xprL)) = 1;

% Circular cross-polarization ratio
xLL = xprmat(1,:,:) + 1j*xprmat(2,:,:) + 1j*xprmat(3,:,:) - xprmat(4,:,:);
xLR = xprmat(1,:,:) - 1j*xprmat(2,:,:) + 1j*xprmat(3,:,:) + xprmat(4,:,:);
xRL = xprmat(1,:,:) - 1j*xprmat(2,:,:) - 1j*xprmat(3,:,:) + xprmat(4,:,:);
xRR = xprmat(1,:,:) + 1j*xprmat(2,:,:) - 1j*xprmat(3,:,:) - xprmat(4,:,:);
xprC = 0.5 * abs(xLL).^2 ./ abs(xLR).^2 + 0.5 * abs(xRR).^2 ./ abs(xRL).^2;
xprC = permute( xprC, [3,2,1] );
xprC(isnan(xprC)) = 1;

% Linear co-polarization ratio
cprL = ( abs( xprmat(1,:,:) ).^2 + abs( xprmat(2,:,:) ).^2 ) ./...
    ( abs( xprmat(3,:,:) ).^2 + abs( xprmat(4,:,:) ).^2 );
cprL = permute( cprL, [3,2,1] );
cprL(isnan(cprL)) = 1;

% Circular co-polarization ratio
cprC = ( abs(xLL).^2 + abs(xLR).^2 ) ./ ( abs(xRR).^2 + abs(xRL).^2 );
cprC = permute( cprC, [3,2,1] );
cprC(isnan(cprC)) = 1;

end
