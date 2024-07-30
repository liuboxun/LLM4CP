function mse = set_grid( h_qd_arrayant , azimuth_grid , elevation_grid, use_interpolate )
%SET_GRID Sets a new grid for azimuth and elevation and interpolates the pattern
%
% Calling object:
%   Single object
%
% Description:
%   This function replaces the properties 'azimuth_grid' and 'elevation_grid' of the antenna object
%   with the given values and interpolates the antenna patterns to the new grid.
%
% Input:
%   azimuth_grid
%   Azimuth angles in [rad] were samples of the field patterns are provided The field patterns are
%   given in spherical coordinates. This variable provides the azimuth sampling angles in radians
%   ranging from -pi to piπ.
%
%   elevation_grid
%   Elevation angles in [rad] were samples of the field patterns are provided The field patterns
%   are given in spherical coordinates. This variable provides the elevation sampling angles in
%   radians ranging from -πpi/2 (downwards) to πpi/2 (upwards).
%
%   use_interpolate
%   Switch to enable (1, default) or disable (0) the antenna pattern interpolation.
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

if ~exist('azimuth_grid','var') || isempty(azimuth_grid)
    azimuth_grid = (-180 : 180)*pi/180;
else
    azimuth_grid = sort( reshape( azimuth_grid,1,[] ) );
end

if ~exist('elevation_grid','var') || isempty(elevation_grid)
    elevation_grid = (-90 : 90)*pi/180;
else
    elevation_grid = sort( reshape( elevation_grid,1,[] ) );
end

if ~exist('use_interpolate','var') || isempty(use_interpolate)
    use_interpolate = true;
end

if ~( any( size(elevation_grid) == 1 ) && isnumeric(elevation_grid) && isreal(elevation_grid) &&...
        max(elevation_grid)<=pi/2 && min(elevation_grid)>=-pi/2 )
    error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "elevation_grid" must be a vector containing values between -pi/2 and pi/2')
end

if ~( any( size(azimuth_grid) == 1 ) && isnumeric(azimuth_grid) && isreal(azimuth_grid) &&...
        max(azimuth_grid)<=pi && min(azimuth_grid)>=-pi )
    error('QuaDRiGa:qd_arrayant:wrongInputValue','??? "azimuth_grid" must be a vector containing values between -pi and pi')
end

if numel(h_qd_arrayant) > 1
    
    sic = size( h_qd_arrayant );
    prc = false( sic );
    if nargout > 0
        mse = zeros( sic );
    end
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            ind = qf.eqo( h_qd_arrayant(i1,i2,i3,i4), h_qd_arrayant );
            prc( ind ) = true;
            if nargout > 0
                mse(i1,i2,i3,i4) = set_grid( h_qd_arrayant(i1,i2,i3,i4), azimuth_grid, elevation_grid, use_interpolate );
                mse(ind) = mse(i1,i2,i3,i4); 
            else
                set_grid( h_qd_arrayant(i1,i2,i3,i4), azimuth_grid, elevation_grid, use_interpolate );
            end
        end
    end
    
else
    h_qd_arrayant = h_qd_arrayant(1,1);
    
    % Make sure we use the correct precision
    if isa( h_qd_arrayant.PFa,'single') && isa( h_qd_arrayant.PFb,'single')
        elevation_grid = single( elevation_grid );
        azimuth_grid = single( azimuth_grid );
    end
    
    if use_interpolate
        el = repmat( elevation_grid' , 1 , numel(azimuth_grid) );
        az = repmat( azimuth_grid , numel(elevation_grid) , 1 );
        [ V,H ] = h_qd_arrayant.interpolate( reshape(az,1,[]), reshape(el,1,[]) );
        V = reshape(V.',numel(elevation_grid),numel(azimuth_grid),[]);
        H = reshape(H.',numel(elevation_grid),numel(azimuth_grid),[]);
        
    else
        no_elements = h_qd_arrayant.no_elements;
        
        El_old = size(h_qd_arrayant.Fa,1);
        El_new = numel(elevation_grid);
        
        Az_old = size(h_qd_arrayant.Fa,2);
        Az_new = numel(azimuth_grid);
        
        V = zeros(El_new,Az_new,no_elements);
        H = zeros(El_new,Az_new,no_elements);
        
        if El_old <= El_new
            El_new = 1:El_old;
            El_old = 1:El_old;
        else
            El_old = 1:El_new;
            El_new = 1:El_new;
        end
        
        if Az_old <= Az_new
            Az_new = 1:Az_old;
            Az_old = 1:Az_old;
        else
            Az_old = 1:Az_new;
            Az_new = 1:Az_new;
        end
        
        V(El_new,Az_new,:) = h_qd_arrayant.Fa(El_old,Az_old,:);
        H(El_new,Az_new,:) = h_qd_arrayant.Fb(El_old,Az_old,:);
    end
    
    % Calculate the MSE of the grid reduction
    if nargout > 0 && use_interpolate
        
        % Interpolate the reduced pattern to the original grid
        a = copy( h_qd_arrayant );
        a.elevation_grid = elevation_grid;
        a.azimuth_grid = azimuth_grid;
        a.Fa = V;
        a.Fb = H;
        set_grid( a, h_qd_arrayant.azimuth_grid, h_qd_arrayant.elevation_grid, true );
        
        % Compare interpolated and original pattern
        mse = sum( abs(a.Fa - h_qd_arrayant.Fa).^2 + abs(a.Fb - h_qd_arrayant.Fb).^2,3);
        mse = -10*log10(max(mse(:)));
    end

    h_qd_arrayant.elevation_grid = elevation_grid;
    h_qd_arrayant.azimuth_grid = azimuth_grid;
    h_qd_arrayant.Fa = V;
    h_qd_arrayant.Fb = H;
    
end

end
