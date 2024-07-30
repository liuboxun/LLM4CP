function [ V, H, dist, azimuth, elevation ] = interpolate( h_qd_arrayant, azimuth, elevation, i_element,...
    orientation, threshold, use_gpu )
%INTERPOLATE Interpolates the array antenna field patterns
%
% Calling object:
%   Single object
%
% Description:
%   This function interpolates the polarimetric antenna field patterns for a given set of azimuth
%   and elevation angles. Interpolation of the antenna field patterns is very computing intensive.
%   It must be performed several thousands of times during a simulation run. Therefore, the
%   function implements two interpolation methods: 2D linear interpolation and spheric
%   interpolation. Linear interpolation is used for real-valued field patterns and for complex-
%   valued field pattern that have a small phase-variation between neighboring elements. However,
%   linear interpolation performs very poorly when there is a large phase difference between two
%   neighboring samples of the pattern. In this case, spheric interpolation is used.
%
% Input:
%   azimuth
%   A vector of azimuth angles in [rad]. The default dimensions are: [ 1, no_ang ]. It is possible
%   to provide a different angle for each element of the array antenna. In this case, the
%   dimensions are [ no_elements, no_ang ], where 'no_elements' corresponds to the number of
%   entries in 'i_element' or the number of elements in the array antenna if 'i_element' is not
%   given.
%
%   elevation
%   A vector of elevation angles in [rad]. The dimensions are: [ 1, no_ang ] or in case of per-
%   element angles [ no_elements, no_ang ].
%
%   i_element
%   The element indices for which the interpolation is done. If no element index is given, the
%   interpolation is done for all elements in the array. Dimensions: [ 1, no_elements ]
%
%   orientation
%   This (optional) 3-element vector describes the orientation of the array antenna. The The first
%   value describes the ”bank angle”, the second value describes the ”tilt angle”, (positive values
%   point upwards), the third value describes the bearing or ”heading angle”, in mathematic sense.
%   East corresponds to 0, and the angles increase counter-clockwise, so north is 90 degrees, south
%   is -90 degree, and west is 180 degree. All values are given in [rad]. By default, the
%   orientation is [0;0;0], i.e. the broadside of the antenna points at the horizon towards the
%   East.
%
%   threshold
%   The maximum phase difference in [deg] between two neighboring entries in the antenna field
%   pattern for which linear interpolation is allowed. Linear interpolation is much faster, but
%   also inaccurate for large phase offsets. By default, a 0 degree threshold is used, i.e.,
%   spheric polarization is used for all complex-valued patterns.
%
%   use_gpu
%   Enables or disables (default) GPU acceleration. This requires a compatible GPU and the
%   "Parallel Computing Toolbox" for MATLAB. In Octave, GPU acceleration is available through the
%   "ocl"-package (https://octave.sourceforge.io/ocl).
%
% Output:
%   V
%   The interpolated vertical field pattern (E-θ-component). Dimensions: [ no_elements, no_ang ]
%
%   H
%   The interpolated horizontal field pattern (E-ϕ-component).  Dimensions: [ no_elements, no_ang ]
%
%   dist
%   The effective distances between the antenna elements when seen from the direction of the
%   incident path. The distance is calculated by an projection of the array positions on the normal
%   plane of the incident path. This is needed for the planar wave approximation. 
%   Dimensions: [ no_elements, no_ang ]
%
%   azimuth
%   The azimuth angles in [rad] for the local antenna coordinate system, i.e., after applying the
%   'orientation'. If no orientation vector is given, these angles are identical to the input
%   azimuth angles.
%
%   elevation
%   The elevation angles in [rad] for the local antenna coordinate system, i.e., after applying the
%   'orientation'. If no orientation vector is given, these angles are identical to the input
%   elevation angles.
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

if numel( h_qd_arrayant ) > 1
    error('QuaDRiGa:qd_arrayant:interpolate','interpolate not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

if ~exist( 'i_element','var' ) || isempty( i_element )
    i_element = 1 : h_qd_arrayant.Pno_elements;
end

i_element = uint32( i_element );

if ~exist( 'orientation','var' ) || isempty( orientation )
    orientation = [];
elseif all( abs(orientation) < 1e-6 )
    orientation = [];
end

if ~exist( 'threshold','var' ) || isempty( threshold )
    threshold = 0; % Always use circular interpolarion
end

if ~exist( 'use_gpu','var' ) || isempty( use_gpu )
    use_gpu = 0;
elseif logical( use_gpu ) && ~qd_simulation_parameters.has_gpu
    use_gpu = 0;
end

% Typecast array to match requested GPU presision
if (use_gpu == 2 || use_gpu == 4) && isa( h_qd_arrayant.PFa, 'double' )
    single( h_qd_arrayant );
elseif (use_gpu == 1 || use_gpu == 3) && isa( h_qd_arrayant.PFa, 'single' )
    double( h_qd_arrayant );
end

% Calculate the maximum phase difference for each element. Store the variable in hidden property to
% avoid multiple calculations.
if threshold > 0 && isempty( h_qd_arrayant.Pphase_diff )
    phase_diff = h_qd_arrayant.calc_phase_diff;
elseif threshold > 0
    phase_diff = h_qd_arrayant.Pphase_diff;
end

% Set some numeric boundaries for single and double precision
if isa( h_qd_arrayant.PFa, 'single' )
    % Regularization parameter R0 is set such that R0^2 is 1.40e-45 (smallest possible single)
    R0 = single( 3.7433921e-23 );  
    R1 = single(-0.9999);
    use_single = true;
else % double
    % Regularization parameter R0 is set such that R0^2 is 4.94e-324 (smallest possible double)
    R0 = 2.222758749485077e-162;
    R1 = -0.99999999;
    use_single = false;
end

% Typecast input variables, if needed
if use_single && isa( azimuth, 'double' )
    azimuth = single( azimuth );
    elevation = single( elevation );
    orientation = single( orientation );
elseif ~use_single && isa( azimuth, 'single' )
    azimuth = double( azimuth );
    elevation = double( elevation );
    orientation = double( orientation );
end

% Get initial values
no_element = numel( i_element );
no_values = numel( azimuth );
no_ang = size( azimuth,2 );
per_element_interpol = size(azimuth,1) == no_element;
azimuth = azimuth(:);
elevation = elevation(:);
azimuth_grid = h_qd_arrayant.azimuth_grid(:);
elevation_grid = h_qd_arrayant.elevation_grid(:);
element_position = h_qd_arrayant.Pelement_position(:,i_element);
no_az = uint32( numel( azimuth_grid ));
no_el = uint32( numel( elevation_grid ));

% Mapping of angles to (-pi/2,pi/2) and changing the azimuthal orientation of rays
% with original elevation angles within +-(pi/2,3pi/2)+n*2*pi. 29.9.2008 EsaK

ind = elevation > pi | elevation <= -pi;
elevation(ind) = mod( elevation(ind)+pi , 2*pi )-pi;

ind = elevation > pi/2;
elevation(ind) = pi - elevation(ind);
azimuth(ind) = azimuth(ind) + pi;

ind = elevation < -pi/2;
elevation(ind) = -pi - elevation(ind);
azimuth(ind) = azimuth(ind) + pi;

ind = azimuth > pi | azimuth <= -pi;
azimuth(ind) = mod( azimuth(ind)+pi , 2*pi )-pi;

% Include the device orientation
if ~isempty( orientation )
    if per_element_interpol && size(orientation,2) == no_ang
        [ ~, azimuth, elevation, gamma ] = qf.calc_ant_rotation( ...
            repmat(  orientation(3,:), no_element,1),...
            repmat( -orientation(2,:), no_element,1),...
            repmat(  orientation(1,:), no_element,1),...
            azimuth, elevation );
    else
        [ ~, azimuth, elevation, gamma ] = qf.calc_ant_rotation( ...
            orientation(3,:), -orientation(2,:), orientation(1,:),...
            azimuth, elevation );
    end
end

% Interpolate field patterns
tmp_1_no_vals = uint32( (1:no_values) );

% Determine the nearest location of xi in x and the difference to the next point
[a,b]   = sort( azimuth );
[~,a]   = sort( [azimuth_grid;a] );
ui      = uint32( 1:(no_az + no_values) );
ui(a)   = ui;
ui      = ui(no_az+1:end) - tmp_1_no_vals;
ui(b)   = ui;
ui( ui==no_az ) = no_az-1;
ui( ui==0 ) = 1;
if numel( azimuth_grid ) > 1
    uin = ui+1;
    u = (azimuth-azimuth_grid(ui))./( azimuth_grid(uin)-azimuth_grid(ui) );
else
    uin = 1;
    u = 0;
end

% Determine the nearest location of yi in y and the difference to the next point
[a,b]   = sort( elevation );
[~,a]   = sort( [elevation_grid;a] );
vi      = uint32( 1:(no_el + no_values) );
vi(a)   = vi;
vi      = vi(no_el+1:end) - tmp_1_no_vals;
vi(b)   = vi;
vi( vi==no_el ) = no_el-1;
vi( vi==0 ) = 1;
if numel( elevation_grid ) > 1
    vin = vi+1;
    v = (elevation-elevation_grid(vi))./( elevation_grid(vin)-elevation_grid(vi) );
else
    vin = 1;
    v = 0;
end

if ~per_element_interpol % Expand u and v for all elements
    u = repmat( u, 1, no_element );
    u = u';
    u = u(:);
    v = repmat( v, 1, no_element );
    v = v';
    v = v(:);
end

if use_gpu % Transfer to GPU
    u = gpuArray( u );
    v = gpuArray( v );
end


% Illustration of the Interpolation procedure
%
%      c -----f------------------- d
%      |      |                    |
%      |      |(1-v)               |
%      |      |                    |
%      |      |                    |
%      |  u   |         (1-u)      |
%      g----- x ------------------ h
%      |      |                    |
%      |      |v                   |
%      |      |                    |
%      a------e--------------------b


% Determine the linear indices for reading values from the antenna patterns
offset = no_az*no_el .* (i_element-1)';
if per_element_interpol
    offset = repmat( offset, 1, no_ang);
    offset = reshape( offset, 1 , no_values );
    ind = [ vi+(ui-1)*no_el + offset ; vi+(uin-1)*no_el + offset ; ...
        vin+(ui-1)*no_el + offset ; vin+(uin-1)*no_el + offset ]'; % [ a,b,c,d ]
else
    ind = cat( 3, ...
        repmat( offset,1,no_ang ) + repmat(   vi+(ui-1)*no_el,no_element,1 ), ...
        repmat( offset,1,no_ang ) + repmat(  vi+(uin-1)*no_el,no_element,1 ), ...
        repmat( offset,1,no_ang ) + repmat(  vin+(ui-1)*no_el,no_element,1 ), ...
        repmat( offset,1,no_ang ) + repmat( vin+(uin-1)*no_el,no_element,1 ) );
    ind = reshape( ind, no_element*no_ang, 4 ); % [ a,b,c,d ]
end

% Interpolate antenna patterns
isrealV = isreal( h_qd_arrayant.PFa );
isrealH = isreal( h_qd_arrayant.PFb );
for iVH = 1:2
    if iVH == 1
        gF = reshape( h_qd_arrayant.PFa( ind(:) ), no_ang*no_element, 4 );
    else
        gF = reshape( h_qd_arrayant.PFb( ind(:) ), no_ang*no_element, 4 );
    end
    
    if isreal( gF ) % Linear interpolation
        
        if use_gpu % Transfer to GPU
            gF = gpuArray( gF );
        end
        
        % Interpolate Amplitude
        if iVH == 1
            if use_gpu && ~isrealV
                V = (1-v).* ((1-u).* gF(:,1) + u.* gF(:,2)) + v.* ((1-u).* gF(:,3) + u.* gF(:,4)) + 1j*R0;
            else
                V = (1-v).* ((1-u).* gF(:,1) + u.* gF(:,2)) + v.* ((1-u).* gF(:,3) + u.* gF(:,4));
            end
        else
            if use_gpu && ~isrealH
                H = (1-v).* ((1-u).* gF(:,1) + u.* gF(:,2)) + v.* ((1-u).* gF(:,3) + u.* gF(:,4)) + 1j*R0;
            else
                H = (1-v).* ((1-u).* gF(:,1) + u.* gF(:,2)) + v.* ((1-u).* gF(:,3) + u.* gF(:,4));
            end
        end
        
    elseif threshold > 0 && all( phase_diff(i_element) < threshold )
        % Use linear interpolation of the real and imaginary part of the patterm. This is less
        % accurate but more efficient
        
        gFr = real( gF );
        gFi = imag( gF );
        
        if use_gpu % Transfer to GPU
            gFr = gpuArray( gFr );
            gFi = gpuArray( gFi );
        end
        
        % Interpolate Real and Imaginary part
        gFr = (1-v).* ((1-u).* gFr(:,1) + u.* gFr(:,2)) + v.* ((1-u).* gFr(:,3) + u.* gFr(:,4));
        gFi = (1-v).* ((1-u).* gFi(:,1) + u.* gFi(:,2)) + v.* ((1-u).* gFi(:,3) + u.* gFi(:,4));
        
        % Combine results
        if iVH == 1
            V = gFr + 1j * gFi;
        else
            H = gFr + 1j * gFi;
        end
        
    else % Spheric interpolation
        
        gFr = real( gF ) + R0;
        gFi = imag( gF ) + R0;
        
        if use_gpu % Transfer to GPU
            gFr = gpuArray( gFr );
            gFi = gpuArray( gFi );
        end
        
        % Calculate Amplitude (absolute value)
        gP = sqrt( gFr.^2 + gFi.^2 );
        
        % Normalize real and imaginary parts - get the phasees of the elements
        gFr = gFr ./ gP + R0;
        gFi = gFi ./ gP + R0;
        
        % Interpolate Amplitude
        gPi = (1-v).* ((1-u).* gP(:,1) + u.* gP(:,2)) + v.* ((1-u).* gP(:,3) + u.* gP(:,4));
        
        % Point E
        gA  = gFr(:,1).*gFr(:,2) + gFi(:,1).*gFi(:,2);  % cos(phase) between point a and b
        gB  = (gA<=1).*gA + (gA>1);                     % fix numeric precision problem when points have same phase
        gC  = (gB>R1).*gB + (gB<=R1).*R1;               % fix precision problem when points have 180 deg pahse diffrence
        gD  = acos( gC ) + R0;                          % phase between a and b (can be 0) --> fix by adding R0
        gE  = sin( gD );                                % sin( R0 ) = R0
        gWp = sin(    u  .* gD ) ./ gE;                 % this will return v when gC is set to R0 due to sin(v*R0) = v*R0
        gWn = sin( (1-u) .* gD ) ./ gE;                 % this will return 1-v when gC is set to R0        
        grE = gWn .* gFr(:,1) + gWp .* gFr(:,2);        % real part of point E
        giE = gWn .* gFi(:,1) + gWp .* gFi(:,2);        % imaginary part of point E
        
        % Point F
        gA  = gFr(:,3).*gFr(:,4) + gFi(:,3).*gFi(:,4);  % cos(phase) between point c and d
        gB  = (gA<=1).*gA + (gA>1);                     % fix numeric precision problem when points have same phase
        gC  = (gB>R1).*gB + (gB<=R1).*R1;               % fix precision problem when points have 180 deg pahse diffrence
        gD  = acos( gC ) + R0;                          % phase between c and d (can be 0) --> fix by adding R0
        gE  = sin( gD );                                % sin( R0 ) = R0
        gWp = sin(    u  .* gD ) ./ gE;                 % this will return v when gC is set to R0 due to sin(v*R0) = v*R0
        gWn = sin( (1-u) .* gD ) ./ gE;                 % this will return 1-v when gC is set to R0        
        grF = gWn .* gFr(:,3) + gWp .* gFr(:,4);        % real part of point F
        giF = gWn .* gFi(:,3) + gWp .* gFi(:,4);        % imaginary part of point F
        
        % Point X
        gA  = grE.*grF + giE.*giF;                      % cos(phase) between point e and f
        gB  = (gA<=1).*gA + (gA>1);                     % fix numeric precision problem when points have same phase
        gC  = (gB>R1).*gB + (gB<=R1).*R1;               % fix precision problem when points have 180 deg pahse diffrence
        gD  = acos( gC ) + R0;                          % phase between e and f (can be 0) --> fix by adding R0
        gE  = sin( gD );                                % sin( R0 ) = R0
        gWp = sin(    v  .* gD ) ./ gE;                 % this will return v when gC is set to R0 due to sin(v*R0) = v*R0
        gWn = sin( (1-v) .* gD ) ./ gE;                 % this will return 1-v when gC is set to R0        
        grX = gWn .* grE + gWp .* grF;                  % real part of point X
        giX = gWn .* giE + gWp .* giF;                  % imaginary part of point X
        
        if iVH == 1
            V = gPi .* grX + 1j * gPi .* giX;
        else
            H = gPi .* grX + 1j * gPi .* giX;
        end
    end
end

% Include the polarization rotation in the antenna response
if ~isempty( orientation )
    if ~per_element_interpol
        gamma = repmat(gamma',no_element,1);
        gamma = gamma(:);
    end
    if use_gpu % Transfer to GPU
        if ~isrealH || ~isrealV
            gamma(1) = gamma(1) + 1j*R0;
        end
        gamma = gpuArray( gamma );
    end
    gS = cos( gamma );
    gamma = sin( gamma );
    gT = gS .* V - gamma .* H;
    H  = gamma .* V + gS .* H;
    V  = gT;
end

% Format output
V = reshape( V, no_element, no_ang );
H = reshape( H, no_element, no_ang );

% Transfer to CPU if output is requestion in CPU space
if use_gpu == 1 || use_gpu == 2
    V = gather( V );
    H = gather( H );
elseif use_gpu > 2  % Make sure output is complex-valued
    if isrealV
        V = V + 1j*R0;
    end
    if isrealH
        H = H + 1j*R0;
    end
end

if nargout > 2 && any( element_position(:) ~= 0 )
    
    % Transform angles from geographic coordinates to Cartesian coordinates
    C = [ cos( azimuth ), cos( elevation ), sin( elevation ) ];
    C(:,1) = C(:,1) .* C(:,2);
    C(:,2) = C(:,2) .* sin( azimuth );
    
    if ~per_element_interpol
        C = reshape( repmat( reshape(C,1,no_ang,3), [no_element,1,1]) ,no_element*no_ang,3);
    end
    
    % Calculate the projection
    dist = sum( C .* repmat( element_position', no_ang, 1 ),2);
    dist = repmat(dist,1,3).*C;
    dist = -sign(sum(C.*dist,2)) .* sqrt(sum(dist.^2,2));
    dist = reshape( dist, no_element, no_ang );

elseif nargout > 2
    if use_single
        dist = zeros( no_element, no_ang,'single' );
    else
        dist = zeros( no_element, no_ang );
    end
end

if nargout > 3 % Return transformed azimuth and elevation angles
    azimuth = reshape( azimuth, [], no_ang );
    elevation = reshape( elevation, [], no_ang );
end

end
