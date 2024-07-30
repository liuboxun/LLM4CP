function [ trans_wgs84_local , trans_local_wgs84 ] = get_wgs84_trans( lon, lat )
%GET_WGS84_TRANS Calculates the transform matrices for WGS84 to local conversion
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

% Minimum distance for the reference points
min_dist = 1000; %

if numel( lon ) ~= numel( lat )
    error('Longitude and Latitude coordinate points must be of same size.')
end

lon = lon(:);
lat = lat(:);

x = 0.001;
A = [ min(lon)-x max(lat)+x ];  % Top left point
B = [ max(lon)+x max(lat)+x ];  % Top right point
C = [ min(lon)-x min(lat)-x ];  % Bottom left point
D = [ max(lon)+x min(lat)-x ];  % Bottom right point

yd = distance( A,C );  % Determine the distance between corners
xd = distance( A,B );

for n = 1:3
    if yd < min_dist
        sc = min_dist / yd;
        md = mean([A(2),C(2)]);
        df = (A(2)-C(2))*sc/2;
        A(2) = md + df;
        B(2) = A(2);
        C(2) = md - df;
        D(2) = C(2);
    end
    
    if xd < min_dist
        sc = min_dist / xd;
        md = mean([A(1),B(1)]);
        df = (B(1)-A(1))*sc/2;
        A(1) = md - df;
        C(1) = A(1);
        B(1) = md + df;
        D(1) = B(1);
    end
    
    yd = distance( A,C );     % Determine the distance between corners
    xd = distance( A,B );
end

ref_wgs84 = [ A;B;C;D ];  % Calculate transform matrix
ref_local = 0.5 * [ -xd yd ; xd yd ; -xd -yd ; xd -yd ];

[ trans_wgs84_local , trans_local_wgs84 ] = coordinate_transform( ref_wgs84,ref_local );

end


% ----- Subfunction coordinate_transform -----
function [ trans_ab , trans_ba ] = coordinate_transform(ref_a,ref_b)
%COORDINATE_TRANSFORM Calculates transformation matrices to convert coordinates
%
% The Functions calculates the transformation matrices for transforming the
% coordinates of a given coordinate system into another and back.
% You need at least 3 reference points given in ref_a and ref_b to
% perform this calculation.
%
% Stephan Jaeckel
% Fraunhofer Heinrich Hertz Institute
% Wireless Communication and Networks
% Einsteinufer 37, 10587 Berlin, Germany
% e-mail: stephan.jaeckel@hhi.fraunhofer.de

ref_a_conv = ref_a(:,1) + ref_a(:,2)*1j;    % We need complex values
ref_b_conv = ref_b(:,1) + ref_b(:,2)*1j;

% Calculation of the transform matrix
P = [real(ref_a_conv.'); imag(ref_a_conv.'); ones(1,length(ref_a)) ];
G = ref_b_conv';
trans_ab = G*pinv(P);

P = [real(ref_b_conv.'); imag(ref_b_conv.'); ones(1,length(ref_b)) ];
G = ref_a_conv';
trans_ba = G*pinv(P);
end


% ----- Subfunction distance -----
function d = distance(a,b)
% Calculates the distance in [m] between two WGS84 coordinate points

%bs1 = [13.324953942 52.516367424 89.1];
%bs2 = [13.320010821 52.512927240 108.4];
%bs3 = [13.327084958 52.512191523 74.9];

%r1 = (40000*1000 / (2 * pi));
%r2 = sin( pi * (0.5 - (a(2) + b(:,2))/360   ) )*r1;
%d1 = (tand( a(1)-b(:,1) ).*r2).^2;
%d2 = (tand( a(2)-b(:,2) )*r1).^2;

r1 = 6.366197723675814e+06;
r2 = sin( 1.570796326794897 - 0.008726646259972 * (a(2) + b(:,2)) ) *r1;

d1 = ( 0.017453292519943 * ( a(1)-b(:,1) ).*r2 ).^2;
d2 = ( 0.017453292519943 * ( a(2)-b(:,2) ) *r1 ).^2;

if size(a,2) == 2
    d = ( d1 + d2 ).^(1/2);
elseif size(a,2) == 3
    d3 = ( a(3) - b(:,3) ).^2;
    d = ( d1 + d2 + d3 ).^(1/2);
end

end
