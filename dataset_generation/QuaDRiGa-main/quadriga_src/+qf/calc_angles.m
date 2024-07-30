function [ az, el, J, pow, resolved, RX_beam, RX_coeff, azimuth_grid, elevation_grid  ] = ...
    calc_angles( cf, h_qd_arrayant, no_subpath, range, use_gpu, no_chunk, verbose  )
%CALC_ANGLES Estimates the arrival angles from the channel coefficients
%
% Calling object:
%   None (static method)
%
% Description:
%   This function estimates the arrival angles from a given coefficient matrix cf. This is done by
%   comparing the coefficients with the antenna response of the given antenna object h_qd_arrayant
%   and determining the most likely arrival direction in azimuth and elevation relative to the
%   local antenna coordinate system. Note that the direction finding results depend on the spatial
%   resolution capabilities of the antenna. If the array antenna has no spatial resolution (e.g. a
%   single dipole), then the returned values will be incorrect.
%
% Input:
%   cf
%   Channel coefficients; Size [ no_rx, no_tx, no_path, no_snap ]
%
%   h_qd_arrayant
%   Receive antenna object; The number of elements must match  [ no_rx ]
%
%   no_subpath
%   Maximum number of sub-paths per path; Default: 1
%
%   range
%   Allowed angle range in degree [ -el +el -az +az ]; Default: [-90 90 -180 180]
%
%   use_gpu
%   Enables or disables GPU acceleration
%
%   no_chunk
%   Number of snapshots that are processed in parallel; Default: 1
%
%   verbose
%   Show the estimation progress; 0 = Disable, 1 = Show, 2 = Details
%
% Output:
%   az
%   Azimuth angles in [rad]; Size [ no_tx, no_path, no_snap, no_subpath ]
%
%   el
%   Elevation angles in [rad], Size [ no_tx, no_path, no_snap, no_subpath ]
%
%   J
%   Jones vectors, Size [ 2, no_tx, no_path, no_snap, no_subpath ]
%
%   pow
%   Power of the path without antenna pattern; Size [ no_tx, no_path, no_snap, no_subpath ]
%
%   resolved
%   Resolved spatial power; Size [ no_tx, no_path, no_snap ]
%
%   RX_beam
%   MRC beam for all paths and sub-paths; Size [ no_el, no_az, no_snap ]
%
%   RX_coeff
%   Reconstructed channel coefficients; Size [ no_rx, no_tx, no_path, no_snap, no_subpath ]
%
%   azimuth_grid
%   Azimuth sample angles of the MRC beam in [rad]; Size [ 1, no_az ]
%
%   elevation_grid
%   Elevation sample angles of the MRC beam in [rad]; Size [ 1, no_el ]
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

% Regularization parameter R0 is set such that R0^2 is 1.4013e-45 (smallest possible single)
R0 = single( 3.7433921e-23 );
R1 = single( 1.4012985e-45 );     % Smallest possible single

no_tx = size( cf,2 );       % Number of transmit antennas
no_path = size(cf,3);       % Number of paths
no_snap = size(cf,4);       % Number of snapshots

if ~exist('no_subpath','var') || isempty( no_subpath )
    no_subpath = 1;
end

if ~exist( 'range','var' ) || isempty( range )
    range = [-90 90 -180 180];
end

if ~exist( 'use_gpu','var' ) || isempty( use_gpu )
    use_gpu = qd_simulation_parameters.has_gpu;
elseif logical( use_gpu ) && ~qd_simulation_parameters.has_gpu
    use_gpu = false;
end

if ~exist('no_chunk','var') || isempty( no_chunk )
    no_chunk = 1;
else
    no_chunk = min( no_chunk, no_snap );
end

if ~exist('verbose','var') || isempty( verbose )
    verbose = 1;
end

% Check h_qd_arrayant
if exist('h_qd_arrayant','var') && ~isempty( h_qd_arrayant ) ...
        && isa(h_qd_arrayant,'qd_arrayant') && numel(h_qd_arrayant) == 1
    if isa( h_qd_arrayant.azimuth_grid,'double' ) || any( abs(h_qd_arrayant.element_position(:) ) > 1e-6)
        h_qd_arrayant = copy( h_qd_arrayant );
        if isa( h_qd_arrayant.azimuth_grid,'double' )
            single( h_qd_arrayant );
        end
        if any( abs(h_qd_arrayant.element_position(:) ) > 1e-6)
            combine_pattern( h_qd_arrayant );
        end
    end
else
    error('QuaDRiGa:QF:calc_angles:h_qd_arrayant','??? "h_qd_arrayant" must be a qd_arrayant object.')
end

% Range mask
el_mask = h_qd_arrayant.elevation_grid*180/pi + 1e-4 > range(1) & ...
    h_qd_arrayant.elevation_grid*180/pi - 1e-4 < range(2);
if ~any( el_mask )
    [~,tmp] = min( abs(h_qd_arrayant.elevation_grid*180/pi - mean(range(1:2))) );
    el_mask(tmp) = true;
end
az_mask = h_qd_arrayant.azimuth_grid*180/pi + 1e-4 > range(3) & ...
    h_qd_arrayant.azimuth_grid*180/pi - 1e-4 < range(4);
if ~any( az_mask )
    [~,tmp] = min( abs(h_qd_arrayant.azimuth_grid*180/pi - mean(range(3:4))) );
    az_mask(tmp) = true;
end

% Warining if there is a non-zero element position
if any( h_qd_arrayant.element_position(:) ~= 0 )
    warning('QuaDRiGa:QF:calc_angles:h_qd_arrayant','??? Array antenna element patterns are not relative to the array phase center!')
end

elevation_grid = single( h_qd_arrayant.elevation_grid( el_mask ) );
azimuth_grid   = single( h_qd_arrayant.azimuth_grid(az_mask) );
azimuth_grid_diff = mean(diff( azimuth_grid ));
elevation_grid_diff = mean(diff( elevation_grid ));

if abs(azimuth_grid(1)+pi) < 1e-6 && abs(azimuth_grid(end)-pi) < 1e-6
    wrap_around = true;
else
    wrap_around = false;
end

no_az = sum( az_mask );
no_el = sum( el_mask );
no_rx = h_qd_arrayant.no_elements;

% Numer of data sets processed in parallel
no_parallel = no_tx * no_path * no_chunk;
no_ang = no_az*no_el;

% Generate offset vector for linear indexing on GPU
cIndOffsetP = int64( 0:no_parallel-1 ) * int64( no_ang );
gIndOffsetR = int64( 0:no_rx-1 ) * int64( no_ang );

% Format antenna patterns
gTheta = reshape( single( h_qd_arrayant.Fa(el_mask,az_mask,:) ), [], no_rx );
gPhi   = reshape( single( h_qd_arrayant.Fb(el_mask,az_mask,:) ), [], no_rx );
gTheta = gTheta + 1j*R0;    % Enforce complex-valued single
gPhi   = gPhi   + 1j*R0;    % Enforce complex-valued single

% The reconstructed pattern response after angle estimation
gGhat = zeros( no_rx, no_parallel, no_subpath ,'single');
gGhat(1,1) = 1j*R1;

% Transfer variables to GPU
if use_gpu
    try
        gTheta = gpuArray( gTheta );
        gPhi = gpuArray( gPhi );
        gGhat = gpuArray( gGhat );
        gIndOffsetR = gpuArray( gIndOffsetR );
    catch
        use_gpu = false;
    end
end

% Allocate GPU memory for multiple spatial sub-paths
if no_subpath > 1
    gJv = zeros( 1, no_parallel, no_subpath ,'single');         % Jones vector, vertical
    gJh = zeros( 1, no_parallel, no_subpath ,'single');         % Jones vector, horizontal
    gVp = zeros( no_rx, no_parallel, no_subpath ,'single');     % Interpolated antenna pattern, vertical
    gHp = zeros( no_rx, no_parallel, no_subpath ,'single');     % Interpolated antenna pattern, horizontal
    gJv(1,1) = 1j*R1;                                        % Complex valued
    gJh(1,1) = 1j*R1;
    gVp(1,1) = 1j*R1;
    gHp(1,1) = 1j*R1;
    if use_gpu % Transfer variables to GPU
        gJv = gpuArray( gJv );
        gJh = gpuArray( gJh );
        gVp = gpuArray( gVp );
        gHp = gpuArray( gHp );
    end
end

% Kernel for the spline interpolation function
K = single( 0.5 .* ...
    [0,  2,  0,  0; ...
    -1,  0,  1,  0; ...
    2, -5,  4, -1; ...
    -1,  3, -3,  1].' );

% Create spline interpolation matrix
u  = single( 0:0.01:1 );
gSpline = K * [ ones(1,numel(u),'single') ; u ; u.^2 ; u.^3];
gSpline = [ [ gSpline ; zeros(1,numel(u),'single') ] , [ zeros(1,numel(u)-1,'single') ; gSpline(:,2:end)] ];
no_spline = size( gSpline,2 );
gSpline = repmat( repmat(gSpline',1,5) , no_spline,1 ) .* gSpline( reshape(repmat(1:5,5,1),[],1),...
    reshape(repmat(1:no_spline,no_spline,1),[],1) )';
if use_gpu
    gSpline = gpuArray( gSpline );
end

% Initialize output variables
az       = zeros( no_tx, no_path, no_snap, no_subpath, 'single' );
el       = zeros( no_tx, no_path, no_snap, no_subpath, 'single' );
J        = zeros( 2, no_tx, no_path, no_snap, no_subpath, 'single' );
J(1,1) = J(1,1) + 1j*R1;
pow      = zeros( no_tx, no_path, no_snap, no_subpath, 'single' );
resolved = zeros( no_tx, no_path, no_snap, 'single' );
RX_beam  = zeros( no_el, no_az, no_snap, 'single' );
RX_coeff = zeros( no_rx, no_tx, no_path, no_snap, no_subpath, 'single' );
RX_coeff(1,1) = RX_coeff(1,1) + 1j*R1;

% Process data chunk-wise
tStartAll = clock;
for i_chnk = 1 : ceil( no_snap/no_chunk )
    i_snap = (i_chnk-1)*no_chunk+1 : min( i_chnk*no_chunk, no_snap );
    n_snap = numel( i_snap );
    if verbose && n_snap == 1
        fprintf(['Snapshot ',num2str(i_snap),' / ',num2str(no_snap),' ... ']);
    elseif verbose
        fprintf(['Snapshots ',num2str(i_snap(1)),'-',num2str(i_snap(end)),' / ',num2str(no_snap),' ... ']);
    end
    tStart = clock;
    
    % Extract coefficients for the current chunk
    if n_snap < no_chunk
        gG = cat(2, reshape(single(cf(:,:,:,i_snap)),no_rx,[]), zeros(no_rx,(no_chunk-n_snap)*no_tx*no_path,'single'));
    else
        gG = reshape(single(cf(:,:,:,i_snap)),no_rx,[]);
    end
    gG(gG==0) = R0;
    
    % Transfer G to GPU
    if use_gpu
        gG(1) = gG(1) + 1j*R1;
        gG = gpuArray( gG );
    end
    
    % Find spatial subpaths
    tS = clock;
    for m1 = 1 : no_subpath
        for m2 = 0 : m1
            if m2 < no_subpath
                m = m2+1;
                
                % Remove previously detected spatial sub-paths from  coefficients matrix
                iM = find( (1:m1) ~= m );
                gGminusGhat = gG;
                for m3 = 1 : numel(iM)
                    gGminusGhat = gGminusGhat - gGhat(:,:,iM(m3));
                end
                gGconj = conj( gGminusGhat );
                
                % Calculate the MRC beam for all directions in the antenna patterns
                gMRC = abs( gTheta * gGconj ).^2;
                gMRC = gMRC + abs( gPhi * gGconj ).^2;
                
                % Find the maximum
                [~,ndx] = max( gMRC );
                if ~use_gpu
                    ndx = int64( ndx );
                end
                
                if verbose == 2; fprintf('\n  Subpath %02d : MRC       : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Read the antenna response at the found angles for both polarizations
                gV = reshape( gTheta( reshape(repmat(ndx,no_rx,1)+repmat(gIndOffsetR',1,no_parallel),[],1) ),no_rx, no_parallel);
                gH = reshape(   gPhi( reshape(repmat(ndx,no_rx,1)+repmat(gIndOffsetR',1,no_parallel),[],1) ),no_rx, no_parallel);
                
                % Calculate pseudoinverse of the antenna response for the calculation of the Jones Vector
                gA  = sum( conj( gV ) .* gV ,1 );
                gB  = sum( conj( gV ) .* gH ,1 );
                gC  = sum( conj( gH ) .* gV ,1 );
                gD  = sum( conj( gH ) .* gH ,1 );
                gP  = 1./(gA.*gD - gB.*gC + R0);
                gVi =  repmat( gD.*gP, no_rx,1 ) .* conj( gV ) - repmat( gB.*gP, no_rx,1 ) .* conj(gH);
                gHi = -repmat( gC.*gP, no_rx,1 ) .* conj( gV ) + repmat( gA.*gP, no_rx,1 ) .* conj(gH);
                
                % The antenna night have limited polarization resolution. This needs to be considered.
                gP  = abs(gD)./abs(gA);         % Calculate the condition number
                
                % gP < 0.1                      % Use only V-Polarization
                % gP > 10                       % Use only H-Polarization
                gA = gP >= 0.1;
                gB = gP <= 10;
                gVi = ( gVi .* repmat(gA,no_rx,1) + repmat(1-gA,no_rx,1) ./ (no_rx*gV) ) .* repmat(gB,no_rx,1);
                gHi = ( gHi .* repmat(gB,no_rx,1) + repmat(1-gB,no_rx,1) ./ (no_rx*gH) ) .* repmat(gA,no_rx,1);
                
                % Calculate the Jones Vectors
                gA = sum( gVi .* gGminusGhat , 1);
                gB = sum( gHi .* gGminusGhat , 1);

                gP = 1./sqrt( abs(gA).^2 + abs(gB).^2 );  % 1/sqrt(x) is computationally expensive!
                gA = abs(gA .* gP).^2;          % Vertical component
                gB = abs(gB .* gP).^2 + R0;     % Horizontal component
                
                if verbose == 2; fprintf('\n  Subpath %02d : Jones     : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Update the MRC beam using the polarization information
                gA = gA ./ gB;
                gMRC = abs( gTheta * gGconj ).^2;       % Conversion from complex to real --> requires memory in GPU
                gMRC = gMRC .* repmat(gA,no_ang,1);
                gMRC = gMRC + abs( gPhi * gGconj ).^2;
                gMRC = gMRC .* repmat(gB,no_ang,1);
                
                if verbose == 2; fprintf('\n  Subpath %02d : MRC 2     : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Calculate RX MRC beam for all TX antennas, paths and supaths
                if m1 == no_subpath && m2 == 0
                    gRX_beam = reshape( sum(reshape( gMRC, no_ang, [], no_chunk ),2),no_ang,no_chunk );
                elseif m1 == no_subpath
                    gRX_beam = gRX_beam + reshape( sum(reshape( gMRC, no_ang, [], no_chunk ),2),no_ang,no_chunk );
                end
                
                if verbose == 2; fprintf('\n  Subpath %02d : RX Beam   : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Find the maximum
                [~,ndx] = max( gMRC );
                if use_gpu
                    cNdx = gather( ndx );
                else
                    cNdx = int64( ndx );
                end
                iEl = rem(cNdx-1, no_el) + 1;
                iAz = (cNdx - iEl)/no_el + 1;
                
                % Get the neighboring elevation positions
                pEl = [ iEl-2 ; iEl-1 ; iEl ; iEl+1  ; iEl + 2 ];
                ind = pEl(1,:) < 1;
                if any(ind)
                    pEl(:,ind) = pEl(:,ind) - repmat( pEl(1,ind)-1, 5,1);
                end
                ind = pEl(5,:) > no_el;
                if any(ind)
                    pEl(:,ind) = pEl(:,ind) - repmat( pEl(5,ind)-no_el, 5,1);
                end
                
                % Get the neighboring azimuth positions
                pAz = [ iAz-2 ; iAz-1 ; iAz ; iAz+1 ; iAz+2 ];
                ind = pAz(1,:) < 1;
                if any(ind)
                    if wrap_around
                        pAz(pAz==0)  = no_az - 1;
                        pAz(pAz==-1) = no_az - 2;
                    else
                        pAz(:,ind) = pAz(:,ind) - repmat( pAz(1,ind)-1, 5,1);
                    end
                end
                ind = pAz(5,:) > no_az;
                if any(ind)
                    if wrap_around
                        pAz(pAz==no_az+1) = 2;
                        pAz(pAz==no_az+2) = 3;
                    else
                        pAz(:,ind) = pAz(:,ind) - repmat( pAz(5,ind)-no_az, 5,1);
                    end
                end
                
                % Generate linear index for reading the MRC data from GPU memory
                cNdx = (pAz( reshape(repmat(1:5,5,1),[],1),:)-1)*no_el + repmat(pEl,5,1) + repmat( cIndOffsetP,25,1 );
                if use_gpu
                    gNdx = gpuArray( reshape(cNdx,1,[]) );
                else
                    gNdx = reshape(cNdx,1,[]);
                end
                
                % Read the data from GPU memory and perform spline interpolation
                gMRC  = gMRC( gNdx );
                gMRC  = gSpline * reshape( gMRC, 25, no_parallel );
                
                % Find the maximum
                [~,ndx] = max( gMRC );
                if use_gpu
                    cNdx = gather( ndx );
                else
                    cNdx = int64( ndx );
                end
                iEls = rem(cNdx-1, no_spline) + 1;
                iAzs = (cNdx - iEls)/no_spline + 1;
                
                % Calculate the interpolated angles
                cAz =   azimuth_grid(iAz) + (2*single(iAzs-1)/(no_spline-1)-1)*azimuth_grid_diff;
                cAz = mod(cAz+pi,2*pi)-pi;
                cEl = elevation_grid(iEl) + (2*single(iEls-1)/(no_spline-1)-1)*elevation_grid_diff;
                
                if verbose == 2; fprintf('\n  Subpath %02d : Spline    : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Interpolate the antenna pattern on the CPU (GPU is buggy)
                if no_subpath == 1
                    [ gVp, gHp ] = interpolate( h_qd_arrayant, cAz, cEl, [],[],[], 4 );
                else
                    [ gVp(:,:,m), gHp(:,:,m) ] = interpolate( h_qd_arrayant, cAz, cEl, [],[],[], 4 );
                end
                
                if verbose == 2; fprintf('\n  Subpath %02d : Antenna   : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Calculate the pseudoinverse of the antenna response
                if m == 1 % nRX x 2 matrix for 2 polarizations --> Direct formula available
                    
                    gA  = sum( conj( gVp(:,:,1) ) .* gVp(:,:,1) ,1 );
                    gB  = sum( conj( gVp(:,:,1) ) .* gHp(:,:,1) ,1 );
                    gC  = sum( conj( gHp(:,:,1) ) .* gVp(:,:,1) ,1 );
                    gD  = sum( conj( gHp(:,:,1) ) .* gHp(:,:,1) ,1 );
                    gP  = 1./(gA.*gD - gB.*gC + R0);     % Determinant
                    gVi =  repmat( gD.*gP, no_rx,1 ) .* conj( gVp(:,:,1) ) - repmat( gB.*gP, no_rx,1 ) .* conj(gHp(:,:,1));
                    gHi = -repmat( gC.*gP, no_rx,1 ) .* conj( gVp(:,:,1) ) + repmat( gA.*gP, no_rx,1 ) .* conj(gHp(:,:,1));
                    gP  = abs(gD)./abs(gA);              % Condition number
                    gVi = ( gVi .* repmat(gP>=0.1,no_rx,1) + repmat(gP<0.1,no_rx,1) ./ (no_rx*gVp(:,:,1)+1j*R0) ) .* repmat(gP<=10, no_rx,1);
                    gHi = ( gHi .* repmat(gP<=10, no_rx,1) + repmat(gP>10, no_rx,1) ./ (no_rx*gHp(:,:,1)+1j*R0) ) .* repmat(gP>=0.1,no_rx,1);
                    
                else % nRX x 2m matrix for multiple spatial paths --> Use the Moore-Penrose Inverse (A' * A)^-1 * A'
                    
                    % Calculate (A' * A)
                    gA  = zeros( 1, no_parallel, (2*m)^2, 'single' );
                    gVi = zeros( no_rx, no_parallel, m, 'single' );
                    gHi = zeros( no_rx, no_parallel, m, 'single' );
                    gA(1,1)  = 1j*R1;
                    gVi(1,1) = 1j*R1;
                    gHi(1,1) = 1j*R1;
                    if use_gpu
                        gA  = gpuArray( gA );
                        gVi = gpuArray( gVi );
                        gHi = gpuArray( gHi );
                    end
                    for im = 1 : (2*m)^2
                        i1 = mod(ceil(im/m)-1,2)+1;         % Select V (=1) or H (=2) for the row index
                        i2 = mod(im-1,m)+1;                 % Entry within V/H for the row index
                        i3 = ceil(2*im/(2*m)^2);            % Select V (=1) or H (=2) for the column index
                        i4 = mod(ceil(im/(2*m))-1,m)+1;     % Entry within V/H for the column index
                        if i1 == 1 && i3 == 1
                            gA(:,:,im) = sum( conj( gVp(:,:,i2) ) .* gVp(:,:,i4) ,1 );
                        elseif i1 == 1 && i3 == 2
                            gA(:,:,im) = sum( conj( gVp(:,:,i2) ) .* gHp(:,:,i4) ,1 );
                        elseif i1 == 2 && i3 == 1
                            gA(:,:,im) = sum( conj( gHp(:,:,i2) ) .* gVp(:,:,i4) ,1 );
                        else % i1 == 2 && i3 == 2
                            gA(:,:,im) = sum( conj( gHp(:,:,i2) ) .* gHp(:,:,i4) ,1 );
                        end
                    end
                    
                    % Calculate (A' * A)^-1
                    % We need the inverse of gA, but it is generally ill-conditioned. The only stable option
                    % is to use a truncated SVD. However, there is no GPU implementation on openCL of this.
                    if use_gpu      % Transfer back to CPU
                        cA = reshape( permute( gather( gA ), [3,2,1] ), 2*m, 2*m, no_parallel );
                    else
                        cA = reshape( permute( gA, [3,2,1] ), 2*m, 2*m, no_parallel );
                    end
                    for ip = 1 : no_parallel
                        [U,S,V] = svd(cA(:,:,ip),'econ');   % Calculate SVD
                        S(S<(0.01*S(1))) = 0;               % Truncate
                        S(S>0) = 1./S(S>0);                 % Invert
                        cA(:,:,ip) = (V*S.')*U';            % Recompose
                    end
                    if use_gpu      % Transfer back to GPU
                        gA = gpuArray( permute( reshape( cA, (2*m)^2, no_parallel ), [3,2,1] ) );
                    else
                        gA = permute( reshape( cA, (2*m)^2, no_parallel ), [3,2,1] );
                    end
                    
                    % Calculate (A' * A)^-1 * A'
                    for im = 1 : (2*m)^2
                        i1 = mod(ceil(im/m)-1,2)+1;         % Select V (=1) or H (=2) for inverted pattern
                        i2 = mod(im-1,m)+1;                 % Entry within V/H for the inverted pattern
                        i3 = ceil(2*im/(2*m)^2);            % Select V (=1) or H (=2) for the column index
                        i4 = mod(ceil(im/(2*m))-1,m)+1;     % Entry within V/H for the column index
                        if i1 == 1 && i3 == 1
                            gVi(:,:,i2) = gVi(:,:,i2) + repmat( gA(:,:,im), [no_rx,1,1] ) .*  conj( gVp(:,:,i4) );
                        elseif i1 == 1 && i3 == 2
                            gVi(:,:,i2) = gVi(:,:,i2) + repmat( gA(:,:,im), [no_rx,1,1] ) .*  conj( gHp(:,:,i4) );
                        elseif i1 == 2 && i3 == 1
                            gHi(:,:,i2) = gHi(:,:,i2) + repmat( gA(:,:,im), [no_rx,1,1] ) .*  conj( gVp(:,:,i4) );
                        else % i1 == 2 && i3 == 2
                            gHi(:,:,i2) = gHi(:,:,i2) + repmat( gA(:,:,im), [no_rx,1,1] ) .*  conj( gHp(:,:,i4) );
                        end
                    end
                end
                
                % Calculate the final Jones Vectors
                if m == 1 && no_subpath == 1
                    gJv = sum( gVi .* gG , 1);
                    gJh = sum( gHi .* gG , 1);
                elseif m == 1
                    gJv(:,:,1) = sum( gVi .* gG , 1);
                    gJh(:,:,1) = sum( gHi .* gG , 1);
                    gJv(:,:,2:no_subpath) = R0;      % For numeric stability
                    gJh(:,:,2:no_subpath) = 0;
                else % m > 1
                    gJv(:,:,1:m) = gJv(:,:,1:m) .* repmat(gR>=0.95,[1,1,m]) + sum( gVi(:,:,1:m).*repmat(gG,[1,1,m]) ,1) .* repmat(gR<0.95,[1,1,m]);
                    gJh(:,:,1:m) = gJh(:,:,1:m) .* repmat(gR>=0.95,[1,1,m]) + sum( gHi(:,:,1:m).*repmat(gG,[1,1,m]) ,1) .* repmat(gR<0.95,[1,1,m]);
                end
                gP = sqrt( abs(gJv).^2 + abs(gJh).^2 ) + R0;  % sqrt(x) is computationally expensive!
                
                if verbose == 2; fprintf('\n  Subpath %02d : Jones 2   : %1.2f',[m,etime(clock, tS)]); tS = clock; end
                
                % Reconstructed pattern response from the estimated angles and Jones vector
                gGhat(:,:,1:m) = repmat( gP(:,:,1:m), [no_rx,1,1] ) .* ( gVp(:,:,1:m) .* repmat( gJv(:,:,1:m)./gP(:,:,1:m), [no_rx,1,1] ) +...
                    gHp(:,:,1:m) .* repmat( gJh(:,:,1:m)./gP(:,:,1:m), [no_rx,1,1] ) );
                
                % Calculate the MSE of the estimate
                gR = sum( abs( sum(gGhat(:,:,1:m),3) ).^2, 1) ./ sum(abs(gG).^2,1);
                
                % Save the angles (CPU memory)
                az(:,:,i_snap,m) = reshape( cAz( 1:no_tx*no_path*n_snap ), no_tx, no_path, n_snap );
                el(:,:,i_snap,m) = reshape( cEl( 1:no_tx*no_path*n_snap ), no_tx, no_path, n_snap );
            end
        end
    end
    
    % Save the Jones Vector to CPU Memory
    if use_gpu
        tmp = gather( gJv./gP );
        J(1,:,:,i_snap,:) = reshape( tmp( 1,1:no_tx*no_path*n_snap,: ), 1, no_tx, no_path, n_snap, no_subpath );
        tmp = gather( gJh./gP );
        J(2,:,:,i_snap,:) = reshape( tmp( 1,1:no_tx*no_path*n_snap,: ), 1, no_tx, no_path, n_snap, no_subpath );
        tmp = gather( gP );
        pow(:,:,i_snap,:) = reshape( tmp( 1,1:no_tx*no_path*n_snap,: ).^2, no_tx, no_path, n_snap, no_subpath );
        resolved(:,:,i_snap) = reshape( gather(gR( 1:no_tx*no_path*n_snap )), no_tx, no_path, n_snap );
    else
        J(1,:,:,i_snap,:) = reshape( gJv( 1,1:no_tx*no_path*n_snap,: ), 1, no_tx, no_path, n_snap, no_subpath );
        J(2,:,:,i_snap,:) = reshape( gJh( 1,1:no_tx*no_path*n_snap,: ), 1, no_tx, no_path, n_snap, no_subpath );
        tmp = reshape( gP( 1,1:no_tx*no_path*n_snap,: ), 1, no_tx, no_path, n_snap, no_subpath );
        J(:,:,:,i_snap,:) = J(:,:,:,i_snap,:)./repmat(tmp,[2,1,1,1,1]);
        pow(:,:,i_snap,:) = reshape( tmp.^2, no_tx, no_path, n_snap, no_subpath );
        resolved(:,:,i_snap) = reshape( gR( 1:no_tx*no_path*n_snap ), no_tx, no_path, n_snap );
    end
    
    % Read RX beam and RC coeffs from GPU
    if use_gpu
        RX_beam(:,:,i_snap) = reshape( gather(gRX_beam(:,1:n_snap)), no_el, no_az, n_snap );
        if n_snap == no_chunk
            RX_coeff(:,:,:,i_snap,:) = reshape( gather(gGhat), no_rx, no_tx, no_path, n_snap, no_subpath );
        else
            tmp = gather(gGhat);
            RX_coeff(:,:,:,i_snap,:) = reshape( tmp(:,1:no_tx*no_path*n_snap,:), no_rx, no_tx, no_path, n_snap, no_subpath );
        end
    else
        RX_beam(:,:,i_snap) = reshape( gRX_beam(:,1:n_snap), no_el, no_az, n_snap );
        RX_coeff(:,:,:,i_snap,:) =  reshape( gGhat(:,1:no_tx*no_path*n_snap,:), no_rx, no_tx, no_path, n_snap, no_subpath );
    end
   
    % Progress report
    if no_snap > 0 && verbose
        if verbose == 2; fprintf('\n  Total Time             : '); end
        cTime = etime(clock, tStart);
        finished = i_chnk / ceil( no_snap/no_chunk );
        time_remaining = etime(clock, tStartAll) * (1-finished)/finished;
        time_hours = floor( time_remaining/3600 );
        time_minutes = round( (time_remaining/3600 - time_hours)*60 );
        fprintf([num2str(cTime,'%1.1f'),' sec, ',...
            num2str( time_hours,'%d' ),':',num2str(time_minutes,'%02d'),' remaining\n']);
    end
end
