function c = xcorrcoeff (a,b)
%XCORRCOEFF Calculates the correlation coefficient of two vectors
%
% Input:
%   a
%   Data vector of size [ 1 x N ]
%
%   b
%   Data vector of size [ T x N ]
%
% Output:
%   c
%   Correlation coefficient between the values in a and b, size [ 1 x T ]
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

if size(a,1) == 1
    a = a.';
    b = b.';
    turn = true;
else
    turn = false;
end

s = numel(a);
c =( (b'*a)/s - sum(a)*sum(b',2)/s^2 ) ./...
    sqrt( (sum(abs(a).^2)/s - abs(sum(a)/s).^2) * (sum(abs(b).^2)/s - abs(sum(b)/s).^2)' ) ;

if turn
    c = c.';
end
