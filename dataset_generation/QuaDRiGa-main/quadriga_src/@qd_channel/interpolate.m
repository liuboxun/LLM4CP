function [ c, dist_snap ] = interpolate( h_channel, dist, algorithm )
%INTERPOLATE Interpolates the channel coefficients and delays.
%
% Calling object:
%   Single object
%
% Description:
%   The channel builder creates one snapshot for each position that is listed in the track object.
%   When the channel sampling theorem is not violated (i.e. the sample density is ≥ 2), then the
%   channel can be interpolated to any other position on the track. This can be used e.g. to
%   emulate arbitrary movements along the track. For more information see
%   'qd_track.movement_profile', 'qd_track.interpolate_movement', or the tutorial "Applying Varying
%   Speeds (Channel Interpolation)".
%
% Input:
%   dist
%   A vector containing distance values on the track. The distance is measured in [m] relative to
%   the beginning of the track.  Alternatively, "dist" can be given as a 3-D tensor with dimensions
%   [ Rx-Antenna , Tx-Antenna , Snapshot ].  In this case, interpolation is done for each antenna
%   element separately.
%
%   algorithm
%   Selects the interpolation algorithm. The default is linear interpolation. Optional are:
%      * linear - Linear interpolation (optimized for speed)
%      * cubic - Cubic spline interpolation of the channel coefficients and piecewise cubic
%        hermite polynomial interpolation for the delays
%
% Output:
%   c
%   A 'qd_channel' object containing the  interpolated coefficients and delays for each entry in
%   'dist'.
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

if numel( h_channel ) > 1
    error('QuaDRiGa:qd_channel:interpolate','??? "interpolate" is only defined for scalar objects.')
else
    h_channel = h_channel(1,1); % workaround for octave
end

if ~exist( 'dist' , 'var' ) || isempty( dist )
    error('QuaDRiGa:qd_channel:interpolate','??? wrong number of input arguments. You mus specify "dist".');
end

if exist( 'algorithm' , 'var' ) && ~isempty( algorithm )
    switch algorithm
        case 'linear'
            use_linear_interpolation = true;
        case 'cubic'
            use_linear_interpolation = false;
        otherwise
            error('Channel interpolation method type not found; supported are: linear, cubic');
    end
else
    use_linear_interpolation = true;
end

% Get the snapshot positions from the channel data
if isempty( h_channel.rx_position )
    error('QuaDRiGa:qd_channel:interpolate','??? channel object has no positioning information.')
end

% Get the dimension of the channel tensor
nrx = h_channel.no_rxant;
ntx = h_channel.no_txant;
nsnap = h_channel.no_snap;
L = h_channel.no_path;
individual_delays = h_channel.individual_delays;

% Calculate the distance of the snapshot relative to the beginning of the track
rx_position = h_channel.rx_position;
dist_snap = sqrt( sum( abs( rx_position(:,2:end) - rx_position(:,1:end-1) ).^2 ,1 ) );
dist_snap = [0, cumsum(dist_snap) ]';
track_length = dist_snap(end);

% If the track length is 0, we assume that the the positions are given in units of snapshots
if numel(dist_snap) > 1 && track_length < 1e-8 && max(dist) > 0
    if min(dist) > 1-1e-8 && max(dist) < nsnap+1e-8
        dist_snap(:) = 1:nsnap;
        track_length = nsnap;
    end
end

% Parse the distance values
if ~( min(dist(:)) >= min(dist_snap)-1e-8 && max(dist(:)) - track_length < 1e-8 )
    error('QuaDRiGa:qd_channel:interpolate','??? "dist" must be numeric and can not exceed track length');
elseif numel( size(dist) ) == 2 && any( size(dist) ) == 1
    interpolate_per_antenna = false;
elseif size(dist,1) == nrx && size(dist,2) == ntx
    interpolate_per_antenna = true;
else
    error('QuaDRiGa:qd_channel:interpolate','??? "dist" has wrong format');
end

if interpolate_per_antenna
    dist = reshape( permute( dist,[3,1,2] ), [], nrx*ntx );
else
    dist = reshape( dist,[],1 );
end

% Reorder coefficient matrix to [ path , snap , ant ]
cf = h_channel.coeff;                                       % Copy the coefficients
cf = permute( cf, [3,4,1,2] );                              % Reorder dimensions
cf = reshape( cf, L, nsnap, [] );

% Get the average power over all antennas with dimensions [ path, snap ]
if size( cf,3 ) ~= 1                                        % Needed for Octave
    PC = mean( abs( cf ).^2, 3 );                           % Get the combined power
else
    PC = abs( cf ).^2;
end

% Get the delays with dimension [ path , snap , ant ]
% Get the average delays with dimensions [ path, snap ]
tmp_channel = qd_channel;                                   % Copy the delays
copy( tmp_channel, h_channel );
if individual_delays
    dl = h_channel.delay;
    tmp_channel.individual_delays = false;
    dlC = tmp_channel.delay;
else
    dlC = h_channel.delay;
    tmp_channel.individual_delays = true;
    dl = tmp_channel.delay;
end
dl = permute( dl, [3,4,1,2] );                              % Reorder dimensions
dl = reshape( dl, L, nsnap, [] );

% Calculate the average distance between sample points
rx_position = h_channel.rx_position;
tx_position = h_channel.tx_position;
if isempty( tx_position )
    tx_position = [0;0;0];
end
if size( tx_position,2 ) == 1 && size( rx_position,2 ) > 1
    tx_position = tx_position(:,ones(1,nsnap));
end
tx_rx_dist = rx_position - tx_position;
dist_rx = mean( sqrt(sum(( tx_rx_dist(:,2:end) - tx_rx_dist(:,1:end-1) ).^2)) );

% The effective sample density in the channel object (sample per half-wavelength)
if h_channel.center_frequency ~= 0
    sample_density = qd_simulation_parameters.speed_of_light / h_channel.center_frequency / dist_rx / 2;
else
    sample_density = 2.5; % Default value
end

% The threshold for the delay difference between two snapshots. If the MT is moving towards the BS at
% maximum velocity, the delay shortens. The additional factor of 2 takes into account that the LOS
% delay might be normalized out. 
if sample_density < 100
    thres = 4 * sample_density * dist_rx / qd_simulation_parameters.speed_of_light;
else
    thres = 0;
end

ddl = abs( diff( dlC,[],2 ) );                              % Delay differences

% For debugging
% plot(ddl'*1e9); hold on; plot( [1,size(ddl,2)],[thres,thres]*1e9,'--k' ); hold off; title( sample_density );

% Get the list of all MPCs
mpc = zeros( nsnap, 3 );                                    % List of MPCs
mpc( 1,: ) = [ 1 , 1 , nsnap ];                             % LOS path
mpc_ind = 2;                                                % Index of the next MPC

for l = 2 : L
    paths = find( ddl(l,:)>thres );                         % Find segment ends
    paths( 1,end+1 ) = nsnap;                               % Last path
    
    ls = 1;                                                 % Counter
    for m = 1 : numel(paths)
        if any( PC(l,ls:paths(m)) > 1e-35 )                 % Check path power
            mpc( mpc_ind,: ) = [ l, ls, paths(m) ];
            mpc_ind = mpc_ind + 1;
        end
        ls = paths(m) + 1;
    end
end
mpc = mpc( 1:mpc_ind-1 ,: );                                % Remove zeros

coeff = zeros( nrx, ntx, L, size( dist,1 ) );               % Create output data structure
delay = coeff;

for n = 1 : size( mpc,1 )                                   % Interpolate channel for each MPC
    pin = dist_snap( mpc(n,2) : mpc(n,3) ).';               % Range of the input coefficients
    
    if numel( pin ) ~= 1
        if interpolate_per_antenna
            no_ant = nrx * ntx;
        else
            no_ant = 1;
        end
        
        for ant = 1 : no_ant
            io = dist( :,ant ) > pin(1)-1e-5 & dist( :,ant ) < pin(end)+1e-5;
            po = dist( io,ant ).';
            
            if interpolate_per_antenna
                cfI = cf( mpc(n,1), mpc(n,2):mpc(n,3),ant );
                dlI = dl( mpc(n,1), mpc(n,2):mpc(n,3),ant );
            else
                cfI = cf( mpc(n,1), mpc(n,2):mpc(n,3),: );
                dlI = dl( mpc(n,1), mpc(n,2):mpc(n,3),: );
            end
            cfIa = abs( cfI );                                  % Amplitudes
            
            if mpc(n,2) ~= 1                                    % Create a smooth ramp up process
                cfIa(1,1,:) = 0;
            end
            if mpc(n,3) ~= nsnap                                % Smooth ramp-down process
                cfIa(1,end,:) = 0;
            end
            
            if use_linear_interpolation
                cfA  = qf.interp( pin, 0, cfIa, po, [], true );
                cfIp = permute( angle( cfI ), [2,3,1] );
                cfP  = permute( qf.slerp( pin, cfIp, 0, po, true ), [3,1,2] );
                dlO  = qf.interp( pin, 0, dlI, po, [], true );
            else
                cfA  = permute( pchip( pin, permute( cfIa,[3,2,1] ), po ) ,[3,2,1] );
                cfIp = permute( angle( cfI ), [2,3,1] );
                cfP  = permute( qf.slerp( pin, cfIp, 0, po, true ), [3,1,2] );
                dlO  = permute( pchip( pin, permute( dlI,[3,2,1] ), po ) ,[3,2,1] );
            end
            
            if interpolate_per_antenna
                [irx,itx] = qf.qind2sub( [nrx,ntx],ant );
                coeff( irx,itx,mpc(n,1),io ) = reshape( cfA.*exp(1j*cfP) , 1,1,1,[] );
                delay( irx,itx,mpc(n,1),io ) = reshape( dlO              , 1,1,1,[] );
            else
                coeff( :,:,mpc(n,1),io ) = permute( reshape( cfA.*exp(1j*cfP), 1,[],nrx,ntx ), [3,4,1,2] );
                delay( :,:,mpc(n,1),io ) = permute( reshape( dlO             , 1,[],nrx,ntx ), [3,4,1,2] );
            end
        end
    end
end

c = qd_channel( coeff,delay );                              % Write data to output channel object
c.name = h_channel.name;
c.center_frequency = h_channel.center_frequency;
c.version = h_channel.version;
c.individual_delays = individual_delays;

% Interpolate the positions
po = mean(dist,2);
if use_linear_interpolation
    tmp = permute(h_channel.rx_position,[3,2,1]);
    tmp = qf.interp( dist_snap, 0, tmp , po, [], true );        % Use double
    c.rx_position = permute( tmp,[3,2,1]);
else
    c.rx_position = pchip( dist_snap, h_channel.rx_position, po );
end
if size(h_channel.tx_position,2) > 1                            % Dual Mobility
    if use_linear_interpolation
        tmp = permute(h_channel.tx_position,[3,2,1]);
        tmp = qf.interp( dist_snap, 0, tmp , po, [], true );    % Use double
        c.tx_position = permute( tmp,[3,2,1]);
    else
        c.tx_position = pchip( dist_snap, h_channel.tx_position, po );
    end
else
    c.tx_position = h_channel.tx_position;
end

% Process the par-struct
par = h_channel.par;
if ~isempty('par')
    if isfield(par,'pg') && numel(par.pg) > 1.5
        if use_linear_interpolation
            par.pg = qf.interp( dist_snap, 0, par.pg , po );
        else
            par.pg = pchip( dist_snap, par.pg, po );
        end
    end
end
c.par = par;

if ~individual_delays
    c.individual_delays = false;
end

end
