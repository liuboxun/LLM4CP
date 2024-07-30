function zi = interp( x, y, z, xc, yc, use_double )
%INTERP 2D linear interpolation optimized for performace
%
% Calling object:
%   None (static method)
%
% Description:
%   This function implements a 2D linear interpolation which is highly optimized for fast
%   execution. All calculations are done in single-precision floating point (30% faster than double
%   precision, but less accurate), and multiple data sets can be interpolated simultaneously. One-
%   dimensional linear interpolation can be done by usingzi = interp( x, 0, z, xc )
%
% Input:
%   x
%   Vector of x sample points; size [ 1, nx ] or [ nx, 1 ]
%
%   y
%   Vector of y sample points; size [ 1, ny ] or [ ny, 1 ]
%
%   z
%   The input data matrix; size [ ny, nx, ne ]; the 3rd dimension allows for interpolations of
%   multiple data-sets; for one-dimensional interpolation, the size must be [ 1, nx, ne ]
%
%   xc
%   Vector of x sample points after interpolation; size [ 1, nxi ] or [ nxi , 1 ]
%
%   xc
%   Vector of y sample points after interpolation; size [ 1, nyi ] or [ nyi, 1 ]
%
%   use_double
%   If set to true, double precision is used instead of single precision.
%
% Output:
%   zi
%   The interpolated data; size [ nyi, nxi, ne ]
%
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

use_single = true;
if exist( 'use_double','var' ) && use_double
   use_single = false;
end

if isempty( y )
    y = 0;
end

if use_single && isa(z,'double')
    z = single( z );
    z_is_double = true;
else
    z_is_double = false;
end

nx = numel(x);
ny = numel(y);
ne = size( z,3 );

if size( z,1 ) ~= ny || size( z,2 ) ~= nx
    error('Size of z does not match');
end

% Option for 1D linear interpolation
if ny == 1
    y  = 0;
    yc = 0;
    z  = z(:);
end

if use_single
    x  = single( x(:).' );
    y  = single( y(:).' );
    xc = single( xc );
    yc = single( yc );
else
    x = x(:).';
    y = y(:).';
end

nxc = numel(xc);
nyc = numel(yc);
oxc = ones(1,nxc,'uint8');
oyc = ones(1,nyc,'uint8');

xi = reshape( xc , 1, [] );
xi = xi( oyc,: );
xi = xi(:).';

yi = reshape( yc , [] , 1 );
yi = yi( :,oxc );
yi = yi(:).';

ni = numel(xi);
ii = uint32( 1:ni );

% Determine the nearest location of xi in x and the difference to
% the next point
[tmp,b] = sort( xi );
[~,a]   = sort( [x,tmp] );
ui      = uint32( 1:(nx + ni) );
ui(a)   = ui;
ui      = ui(nx+1:end) - ii;
ui(b)   = ui;
ui( ui==nx ) = nx-1;
ui( ui==0 ) = 1;
uin     = ui+1;
u       = (xi-x(ui))./( x(uin)-x(ui) );
u(isnan(u)) = 0;
u       = u';

% Determine the nearest location of yi in y and the difference to
% the next point
if ny > 1
    [tmp,b] = sort( yi );
    [~,a]   = sort( [y,tmp] );
    vi      = uint32( 1:(ny + ni) );
    vi(a)   = vi;
    vi      = vi(ny+1:end) - ii;
    vi(b)   = vi;
    vi( vi==ny ) = ny-1;
    vi( vi==0 ) = 1;
    vin     = vi+1;
    v       = (yi-y(vi))./( y(vin)-y(vi) );
    v(isnan(v)) = 0;
    v       = v';
else
    vi  = uint32( 1 );
    vin = uint32( 1 );
    if use_single
        v = zeros( ni,1,'single' );
    else
        v = zeros( ni,1 );
    end
end

% Calculate the scaling coefficients
c1 = (1-v).*(1-u);
c2 = (1-v).*u;
c3 = v.*(1-u);
c4 = v.*u;

% Determine the indices of the elements
pa = vi  + ( ui  -1 )*ny;
pb = vi  + ( uin -1 )*ny;
pc = vin + ( ui  -1 )*ny;
pd = vin + ( uin -1 )*ny;

pX = [pa,pb,pc,pd].';
pY = uint32( (0:ne-1)*nx*ny );

tr = true( ni,1 );
fl = false( ni,1 );
i1 = [tr;fl;fl;fl];
i2 = [fl;tr;fl;fl];
i3 = [fl;fl;tr;fl];
i4 = [fl;fl;fl;tr];

% Interpolate
if use_single
    zi = zeros( ni, ne, 'single' );
else
    zi = zeros( ni, ne );
end
for n = 1 : ne
    ndx = pY(n) + pX;
    a = z( ndx );
    zi(:,n) = c1.*a(i1) + c2.*a(i2) + c3.*a(i3) + c4.*a(i4);
end

zi = reshape(zi,nyc,nxc,ne);

if z_is_double
    zi = double( zi );
end

end
