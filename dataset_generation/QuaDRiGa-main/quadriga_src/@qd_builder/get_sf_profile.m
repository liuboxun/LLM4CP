function [ sf,kf ] = get_sf_profile( h_builder, evaltrack, txpos )
%GET_SF_PROFILE Returns the shadow fading and the K-factor along a track
%
% Calling object:
%   Single object
%
% Description:
%   This function returns the shadow fading and the K-factor along the given track. This function
%   is mainly used by the channel builder class to scale the output channel coefficients. The
%   profile is calculated by using the data in the LSF autocorrelation model and interpolating it
%   to the positions in the given track.
%
% Input:
%   evaltrack
%   Handle to a 'qd_track' object for which the SF and KF should be interpolated.
%
%   txpos
%   The corresponding tx positions. This variable can be provided by: 
%   (1) A 'qd_track' object having the same number of snapshots as the 'evaltrack'; 
%   (2) A 3-element vector containing the fixed x,y, and z coordinate of the tx; 
%   (3) A [3 x N] matrix containing one tx position for each position on the 'evaltrack'; 
%   (4) The variable is unprovided or empty. In this case, the tx-positions are obtained from the
%       'qd_builder' object. 
%
% Output:
%   sf
%   The shadow fading [linear scale] along the track
%
%   kf
%   The  K-factor [linear scale] along the track
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

if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "get_sf_profile" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

% Parse input variable
if ~( isa(evaltrack, 'qd_track') )
    error('QuaDRiGa:qd_builder:get_sf_profile','??? "evaltrack" must be of class "track".')
elseif ~any( size(evaltrack) == 1  )
    error('QuaDRiGa:qd_builder:get_sf_profile','??? "evaltrack" must be vector.')
end

% Get the positions along the track
x = evaltrack.positions(1,:) + evaltrack.initial_position(1);
y = evaltrack.positions(2,:) + evaltrack.initial_position(2);
z = evaltrack.positions(3,:) + evaltrack.initial_position(3);

nP = evaltrack.no_snapshots;
oP = ones( 1,nP );

% Get the tx_position(s)
% The readout of the tx-positions must match the readout in "get_lsp_val". Otherwise, results may
% differ.
if h_builder.dual_mobility == -1
    h_builder.check_dual_mobility;
end

if h_builder.dual_mobility == 0 % Single Mobility
    % Do not use tx-positions in the SF interpolation
    txpos = h_builder.tx_position(:,1);
    
elseif ~exist('txpos','var') || isempty( txpos )                % No tx-position given
    error('QuaDRiGa:qd_builder:get_sf_profile','Tx-position is not provided');
    
elseif isa( txpos, 'qd_track')                              % Input is provided as a "qd_track" object
    if txpos.no_snapshots == 1 && nP > 1
        txpos = txpos.initial_position(:,oP);
        
    elseif txpos.no_snapshots == nP
        initial_position = txpos.initial_position;
        txpos = txpos.positions;
        txpos = txpos + initial_position * oP;
        
    else
        error('QuaDRiGa:qd_builder:get_sf_profile',...
            'Number of positions in tx track does not match number of positions in evaltrack.');
    end
    
elseif nP > 1 && size( txpos,2 ) == 1
    txpos = txpos(:,oP);
    
elseif all( size( txpos ) ~= [3,nP] )
    error('QuaDRiGa:qd_builder:get_sf_profile','Tx-position is ambiguous.');
end

if isempty( h_builder.sos )
    % We don't have a SOS generator for the LSPs. We want to enforce the values from the builder.
    error('QuaDRiGa:qd_builder:get_sf_profile','No qd_sos object found.');  
else
    % Calculate the KF and the SF from the SOS generators
    [ ~, kf, sf ] = gen_lsf_parameters( h_builder, 3, 0, txpos, [x;y;z] );
end

end
