function rotate_pattern( h_qd_arrayant, deg, rotaxis, i_element, usage )
%ROTATE_PATTERN Rotates antenna patterns
%
% Calling object:
%   Single object
%
% Description:
%   Pattern rotation provides the option to assemble array antennas out of single elements. By
%   setting the 'element_position' property of an array object, elements can be placed at different
%   coordinates. In order to freely design arbitrary array configurations, however, elements often
%   need to be rotated (e.g. to assemble a +/- 45° crosspolarized array out of single dipoles).
%   This functionality is provided here.
%
% Input:
%   deg
%   The rotation angle in [degrees] ranging from -180° to 180°
%
%   rotaxis
%   The rotation axis specified by the string 'x','y', 'z', or 'xyz'. In case of 'xyz', three
%   rotations are performed: one around the 'x' axis, followed by one around the 'y' axis and
%   around the 'z' axis. In this case, the input 'deg' must be a 3-element vector containing the 3
%   rotation angles.
%
%   i_element
%   The element indices for which the rotation is done. If no element index is given, the rotation
%   is done for all elements in the array.
%
%   usage
%   The optional parameter 'usage' can limit the rotation procedure either to the pattern or
%   polarization. Possible values are:
%      * 0: Rotate both (pattern+polarization) - default
%      * 1: Rotate only pattern
%      * 2: Rotate only polarization
%      * 3: Same as (0), but without grid interpolation
%
% QuaDRiGa Copyright (C) 2011-2020
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
   error('QuaDRiGa:qd_arrayant:rotate_pattern','calc_gain not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

% Parse input arguments
if exist('rotaxis','var')
    if ~( ischar(rotaxis) && ...
            (strcmp(rotaxis,'x') || strcmp(rotaxis,'y') || strcmp(rotaxis,'z') || strcmp(rotaxis,'xyz') ) )
        error('QuaDRiGa:qd_arrayant:rotate_pattern','??? "rotaxis" can only be x,y or z.')
    end
else
    rotaxis = 'y';
end

if exist('deg','var')
    if ~( isnumeric(deg) && isreal(deg) )
        error('QuaDRiGa:qd_arrayant:rotate_pattern','??? "deg" must be real-valued')
    end
else
    deg = 0;
end

if exist('i_element','var') && ~isempty( i_element )
    if ~( size(i_element,1) == 1 && isnumeric(i_element) ...
            &&  all( mod(i_element,1)==0 ) && min(i_element) > 0 && max(i_element)<=h_qd_arrayant.no_elements )
        error('QuaDRiGa:qd_arrayant:rotate_pattern',...
            '??? "i_element" must be integer > 0 and can not exceed the number of elements')
    end
else
    i_element = 1:h_qd_arrayant.no_elements;
end

if exist('usage','var')
    if ~( all(size(usage) == [1 1]) && isnumeric(usage) ...
            && any(usage == [0,1,2,3]) )
        error('QuaDRiGa:qd_arrayant:rotate_pattern','??? "usage" must be 0,1,2 or 3')
    end
else
    usage = 0;
end

% Get the angles
phi   = h_qd_arrayant.azimuth_grid;
theta = h_qd_arrayant.elevation_grid';
no_az = h_qd_arrayant.no_az;
no_el = h_qd_arrayant.no_el;

% Rotation vectors are given in degree, but calculations are done in rad.
deg = deg/180*pi;

zrot = [];
yrot = [];
xrot = [];
switch rotaxis
    case 'x'
        xrot = deg;
    case 'y'
        yrot = deg;
    case 'z'
        zrot = deg;
    case 'xyz'
        xrot = deg(1);
        yrot = deg(2);
        zrot = deg(3);
end

% Calculate the minimum grid resolution
dPhi = diff( [phi(end)-2*pi, phi , phi(1)+2*pi ] );      % Azimuth difference
dPhi_max = max( dPhi( dPhi>1e-7 ) );
dPhi_min = min( dPhi( dPhi>1e-7 ) );

dTheta = diff( [ -pi/2, theta' , pi/2 ] );               % Elevation difference
dTheta_max = max( dTheta( dTheta>1e-7 ) );
dTheta_min = min( dTheta( dTheta>1e-7 ) );

% Check if we need to interpolate the angular grid
if usage == 2
    interpolate_grid = false;   % Only polarization 
elseif usage == 3
    interpolate_grid = false;   % Disable grid interpolation
    usage = 0;                  % Interpolate pattern and polarization
elseif isempty( yrot ) && isempty( xrot )
    % We have only a z-rotation
    if (dPhi_max - dPhi_min) < 1e-7   % Check if the azimuth grid is equaly sampled
        interpolate_grid = false;
    else
        interpolate_grid = true;
    end
elseif abs( dPhi_max - dPhi_min ) > 1e-6 || abs( dTheta_max - dTheta_min ) > 1e-6
    interpolate_grid = true;
else
    interpolate_grid = false;
end

% Interpolate the grid, if needed
if interpolate_grid
    
    % It is not possible to rotate a subset of elements if the angular grid needs to be
    % interpolated.
    if any( ((1:h_qd_arrayant.no_elements) - (i_element)) ~= 0)
        error('QuaDRiGa:qd_arrayant:rotate_pattern',...
            'Pattern requires interpolation of the angle grid. You cannot select individual elements in this case.')
    end
    
    % Obtain the minimum power in the pattern
    P_min = sum(abs(h_qd_arrayant.Fa).^2 + abs(h_qd_arrayant.Fb).^2,3);
    P_min = min( P_min(P_min>1e-14) );
    
    % Equidistant sampled elevation grid at original resolution
    thetaN = -pi/2 : pi/round( pi/dTheta_min ) : pi/2 + dTheta_min/2;
    thetaN(1) = -pi/2; 
    thetaN(end) = pi/2; 
    
    % Equidistant sampled azimuth grid at original resolution
    phiN = -pi : 2*pi/round( 2*pi/dPhi_min ) : pi + dPhi_min/2;
    phiN(1) = -pi; 
    phiN(end) = pi;
    
    % Calculate the antenna rotation parameters
    [ ~, phiL, thetaL ] = qf.calc_ant_rotation( -zrot, -yrot, -xrot, phi(ones(1,no_el),:), theta(:,ones(1,no_az)) );
    
    % Get the elevation grid coverge
    thetaL = sort( thetaL(:) );
    ii = [1; diff( round( thetaL./(dTheta_min/3) ) ) ] > 0.5;
    thetaL = thetaL(ii);
    
    % Find the relevatnt angles and remove unused entries
    ii = false( 1,numel( thetaN ));
    for n = 2 : numel( thetaN )-1
       ii( n-1 : n+1 ) = any( thetaL > thetaN(n-1)-1e-7 & thetaL < thetaN(n+1)+1e-7 );
    end
    ii(1:end-1) = ii(1:end-1) | ii(2:end);
    ii(2:end)   = ii(2:end)   | ii(1:end-1);
    thetaN = thetaN(ii)';
    
    % Get the azimuth grid coverge
    phiL = sort( phiL(:) );
    ii = [1; diff( round( phiL./(dPhi_min/3) ) ) ] > 0.5;
    phiL = phiL(ii);
    
    % Find the relevatnt angles and remove unused entries
    ii = false( 1,numel( phiN ));
    for n = 2 : numel( phiN )-1
       ii( n-1 : n+1 ) = any( phiL > phiN(n-1)-1e-7 & phiL < phiN(n+1)+1e-7 );
    end
    ii(1:end-1) = ii(1:end-1) | ii(2:end);
    ii(2:end)   = ii(2:end)   | ii(1:end-1);
    phiN = phiN(ii);

else
    % Without grid interpolation, we keep the exisiting angular grid.
    phiN = phi;
    thetaN = theta;
end

% Calculate the antenna rotation parameters
[ R, phiL, thetaL, gamma ] = qf.calc_ant_rotation( zrot, yrot, xrot,...
    phiN(ones(1,numel(thetaN)),:), thetaN(:,ones(1,numel(phiN))) );

% Calculate the sine and cosine of gamma to reduce computational load.
cos_gamma = cos(gamma);
sin_gamma = sin(gamma);

% Placeholders for the interpolated patterns
patV = zeros( numel(thetaN), numel(phiN), numel(i_element) );
patH = patV;

% If we interpolate the grid, we need to wrap the grid
if ~h_qd_arrayant.is_wrapped && ( usage == 0 || usage == 1 ) && interpolate_grid
    wrap_grid( h_qd_arrayant );
end

for n = 1 : numel(i_element)
    % Interpolate the pattern
    if usage == 0 || usage == 1
        [ V,H ] = h_qd_arrayant.interpolate( reshape(phiL,1,[]), reshape(thetaL,1,[]), i_element(n), [], 1 );
        V = reshape(V,size(phiL));
        H = reshape(H,size(phiL));
    else
        V = h_qd_arrayant.Fa(:,:,i_element(n));
        H = h_qd_arrayant.Fb(:,:,i_element(n));
    end
    
    % Update the element position
    h_qd_arrayant.element_position(:,i_element(n)) = R*h_qd_arrayant.element_position(:,i_element(n));
    
    % Transformation of the polarization
    if usage == 0 || usage == 2
        patV(:,:,n) = cos_gamma.*V - sin_gamma.*H;
        patH(:,:,n) = sin_gamma.*V + cos_gamma.*H;
    else % mode 1 - Rotate patterns only, but not the polarization
        patV(:,:,n) = V;
        patH(:,:,n) = H;
        
    end
end

if interpolate_grid

    % Reduce grid points that have no power
    P = sum(abs(patV).^2 + abs(patH).^2,3);
    
    iTh = max(P,[],2) > P_min;
    iTh(1:end-1) = iTh(1:end-1) | iTh(2:end);
    iTh(2:end)   = iTh(2:end) | iTh(1:end-1);
    
    iPh = max(P,[],1) > P_min;   
    iPh(1:end-1) = iPh(1:end-1) | iPh(2:end);
    iPh(2:end)   = iPh(2:end) | iPh(1:end-1);
        
    % Set new angular grid
    set_grid( h_qd_arrayant, phiN(iPh) , thetaN(iTh), 0 )
    
    % Store interpolated pattern
    h_qd_arrayant.Fa(:,:,i_element) = patV(iTh,iPh,:);
    h_qd_arrayant.Fb(:,:,i_element) = patH(iTh,iPh,:);
else
    % Store interpolated pattern
    h_qd_arrayant.Fa(:,:,i_element) = patV;
    h_qd_arrayant.Fb(:,:,i_element) = patH;
end

end
