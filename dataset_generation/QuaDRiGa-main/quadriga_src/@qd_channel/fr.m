function freq_response = fr( h_channel, bandwidth, carriers, i_snapshot, use_gpu )
%FR Transforms the channel into frequency domain and returns the frequency response
%
% Calling object:
%   Single object
%
% Input:
%   bandwidth
%   The baseband bandwidth in [Hz]
%
%   carriers
%   The carrier positions. There are two options:
%      * Specify the total number of sub-carriers. In this case, 'carriers' a scalar natural number
%        > 0. The sub-carriers are then equally spaced over the bandwidth. The first entry of the
%        generated spectrum is equal to the center frequency f0. The spectrum is generated from f0
%        to f0+bandwidth.
%
%      * Specify the sub-carrier positions. In this case, 'carriers' is a vector of sub-carrier
%        positions relative to the bandwidth. The carrier positions are given relative to the
%        bandwidth where '0' is the begin of the spectrum (i.e., the center frequency f0) and '1' is
%        equal to f0+bandwidth. To obtain the channel frequency response centered around f0, the
%        input variable 'carriers' must be set to '(-N/2:N/2)/N', where N is the number of sub-
%        carriers.
%
%   i_snapshot
%   The snapshot numbers for which the frequency response should be calculated. By default, i.e. if
%   'i_snapshot' is not given, all snapshots are processed.
%
%   use_gpu
%   Enables or disables (default) GPU acceleration. This requires a compatible GPU and the
%   "Parallel Computing Toolbox" for MATLAB. In Octave, GPU acceleration is available through the
%   "ocl"-package (https://octave.sourceforge.io/ocl).
%
% Output:
%   freq_response
%   The complex-valued channel coefficients for each carrier in frequency domain. The indices of
%   the 4-D tensor are: [ Rx-Antenna , Tx-Antenna , Carrier-Index , Snapshot ]
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

if numel( h_channel ) > 1
    error('QuaDRiGa:qd_channel:fr','??? "fr" is only defined for scalar objects.')
else
    h_channel = h_channel(1,1); % workaround for octave
end

if nargin < 3
    error('QuaDRiGa:qd_channel:fr','??? You must specify the bandwidth and the number of carriers.')
end

if ~exist( 'i_snapshot','var' ) || isempty( i_snapshot )
    i_snapshot = 1 : h_channel.no_snap;
    check = false;
else
    check = true;
end
    
if ~exist( 'use_gpu','var' )
    use_gpu = qd_simulation_parameters.has_gpu;
elseif logical( use_gpu ) &&  ~qd_simulation_parameters.has_gpu
    use_gpu = 0;
end

if ~( size(bandwidth,1) == 1 && isnumeric(bandwidth) && all(size(bandwidth) ==...
        [1 1]) && min(bandwidth) > 0 )
    error('??? The bandwidth "bandwidth" must be scalar and > 0')
end

if isnumeric(carriers) && isreal(carriers)
    
    if all(size(carriers) == [1 1]) && mod(carriers,1)==0 && carriers>0
        pilot_grid = ( 0:carriers-1 )/carriers;
        
    elseif numel( size(carriers) ) == 2 && any(size(carriers)==1)
        if size(carriers,2) == 1
            pilot_grid = carriers.';
        else
            pilot_grid = carriers;
        end
        carriers = numel(pilot_grid);
        
    else
        error('??? Invalid input for "carriers".')
    end
else
    error('??? The no. of carriers must be numeric.')
end

if check
    if ~( any( size(i_snapshot)==1 ) && isnumeric(i_snapshot) &&...
            all( mod(i_snapshot,1)==0 ) ...
            && min(i_snapshot) > 0 && max(i_snapshot)<=h_channel.no_snap )
        error(['??? The snapshot range must be numeric,',...
            ' integer and can not exceed the numbers of snapshots']);
    end
end

% Get the dimension of the channel tensor
n_rx = h_channel.no_rxant;
n_tx = h_channel.no_txant;
n_i_snapshots = numel(i_snapshot);
n_taps = h_channel.no_path;

% Preallocate some memory and rearrange coefficients
freq_response = zeros(n_i_snapshots * n_rx * n_tx, carriers); 
if h_channel.individual_delays
    m = reshape(permute(h_channel.delay(:, :, :, i_snapshot)*bandwidth,...
        [ 4 1 2 3 ]), n_i_snapshots*n_rx*n_tx, n_taps);
else
    m = repmat(h_channel.delay(:, i_snapshot)'*bandwidth, n_rx*n_tx, 1);
end
c = reshape(permute(h_channel.coeff(:, :, :, i_snapshot), [ 4 1 2 3 ]),...
    n_i_snapshots*n_rx*n_tx, n_taps);

% The arguments of the exponential function
v  = -2 * pi * 1j * pilot_grid;

% Load all variables to GPU memory
if use_gpu == 1
    freq_response(1) = freq_response(1) + 1j*4e-324;
    m(1) = m(1) + 1j*4e-324;
    c(1) = c(1) + 1j*4e-324;
    freq_response = gpuArray( freq_response );
    m = gpuArray( m );
    c = gpuArray( c );
    v = gpuArray( v );
elseif use_gpu == 2
    freq_response = single( freq_response );
    m = single( m );
    c = single( c );
    v = single( v );
    freq_response(1) = freq_response(1) + 1j*1e-45;
    m(1) = m(1) + 1j*1e-45;
    c(1) = c(1) + 1j*1e-45;
    freq_response = gpuArray( freq_response );
    m = gpuArray( m );
    c = gpuArray( c );
    v = gpuArray( v );
end

% The main calculation
for i_tap = 1 : n_taps
    if use_gpu
        freq_response = freq_response + repmat( c(:,i_tap), 1, carriers ) .* exp( m(:,i_tap) * v);
    else
        freq_response = freq_response + bsxfun(@times, exp( m(:,i_tap) * v), c(:,i_tap) );
    end
end

% Retrieve data from GPU memory
if use_gpu
    freq_response = double( gather( freq_response ) );
end

% Reorder the output dimensions
freq_response = reshape(freq_response, n_i_snapshots, n_rx, n_tx, carriers);
freq_response = permute(freq_response, [ 2 3 4 1 ]);

end
