function combine_pattern( h_qd_arrayant, center_frequency )
%COMBINE_PATTERN Calculates a virtual pattern of the given array
%
% Calling object:
%   Single object
%
% Description:
%   When the inputs of an array antenna are coupled (i.e. fed with the same signal), then it is
%   possible to combine the elements of the array. This function calculates the virtual pattern by
%   using the QuaDRiGa simulator. Individual coupling weights can be set in the "coupling" property
%   of the qd_arrayant object. Phase offsets of the individual antenna elements due to their
%   positions in the array ("element_position" property of the calling qd_arrayant object) are
%   calculated for the phase center of the array.
%
% Input:
%   center_frequency
%   The center frequency in [Hz]. If this input variable is not given, it is assumed that the
%   element spacings in the "element_position" property of the calling arrayant object are given in
%   multiples of the carrier wavelength.
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
   error('QuaDRiGa:qd_arrayant:combine_pattern','combine_pattern not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

% We assume that spacings can be given in units "lambda".
% The base-frequency is therefor ~ 300 MHz , so the wavelength is 1 m

if ~exist('center_frequency','var')
    center_frequency = h_qd_arrayant.center_frequency;           
end

% The receiver positions are placed in 1000 lambda distance in the same grid
% given by the elevation and azimuth angles in the original qd_arrayant.

phi   = h_qd_arrayant.azimuth_grid;
theta = h_qd_arrayant.elevation_grid';
no_az = h_qd_arrayant.no_az;
no_el = h_qd_arrayant.no_el;
no_tx = h_qd_arrayant.no_elements; 
no_positions = no_az * no_el;

precision = 'double';
if isa(phi,'single') && isa(theta,'single')
    precision = 'single';
    lambda = 299792458 / single( center_frequency );
else
    lambda = 299792458 / center_frequency;
end
wave_no = 2*pi/lambda;

B = zeros( 3,no_el,no_az,precision);
B(1,:,:) = cos(theta)*cos(phi);
B(2,:,:) = cos(theta)*sin(phi);
B(3,:,:) = sin(theta)*ones(1,no_az);
B = 1000*lambda*reshape(B, 3, no_positions);

% Calculate the angles
angles = zeros( 4,no_positions,precision);
angles(1,:) = atan2( B(2,:),  B(1,:) );     % ThetaBs 
angles(2,:) = pi + angles(1,:);             % ThetaMs 
angles(3,:) = atan( B(3,:) ./ sqrt( B(1,:).^2 + B(2,:).^2 ) );   % EaBs

% When Rx and Tx are at the same position, the angle is NaN
% We set it to 0 instead.
angles(3, isnan( angles(3,:) ) ) = 0;
angles(4,:) = -angles(3,:);     % EaMs

% Interpolate the patterns
[ Vt,Ht,Pt ] = h_qd_arrayant.interpolate( angles(1,:) , angles(3,:) );
Ct = h_qd_arrayant.coupling;
Pt = exp( -1j*(  wave_no*( Pt )));

% Calculate the coefficients
c = zeros( 2*no_tx , no_positions, precision);
for i_tx = 1 : no_tx
    PatTx = [ Vt(i_tx,:) ; Ht(i_tx,:) ];
    
    % First component
    ind = (i_tx-1)*2 + 1;
    c(ind,:) = PatTx(1,:) .* Pt(i_tx,:);
    
    % Second component
    ind = ind + 1;
    c(ind,:) = PatTx(2,:) .* Pt(i_tx,:);
end

% Apply antenna coupling
c = reshape( c , 2 , no_tx , no_positions );

n_tx = size(Ct,2);
coeff = zeros( 2 , n_tx , no_positions, precision);
if all(size(Ct) == [ n_tx , n_tx ]) && ...
        all(all( abs( Ct - eye(n_tx)) < 1e-10 ))
    
    % Identity matrix, no oupling
    coeff = c;
    
elseif all(size(Ct) == [ n_tx , n_tx ]) && ...
        all(all( abs( Ct - diag(diag(Ct)) ) < 1e-10 ))
    
    % The tx has a diagonal matrix
    for i_tx = 1 : n_tx
        coeff( :,i_tx,: ) = c(:,i_tx,:) .* Ct( i_tx,i_tx );
    end
    
else
    % The tx has no diagonal matrix
    for i_tx_in = 1 : no_tx
        for i_tx_out = 1 : size(Ct,2)
            coeff( :,i_tx_out,: ) = coeff( :,i_tx_out,: ) + c(:,i_tx_in,:) .* Ct(i_tx_in,i_tx_out );
        end
    end
end

pat = permute( coeff , [3,2,1] ); % Map, Tx, Rx

% Write the output pattern
h_qd_arrayant.no_elements = size( pat,2 );
h_qd_arrayant.PFa = reshape( pat(:,:,1), no_el ,no_az , [] );
h_qd_arrayant.PFb = reshape( pat(:,:,2), no_el ,no_az , [] );
h_qd_arrayant.Pelement_position = zeros(3,size( pat,2 ),precision);
h_qd_arrayant.Pcoupling = eye( h_qd_arrayant.no_elements,precision);
h_qd_arrayant.Pphase_diff = [];

end
