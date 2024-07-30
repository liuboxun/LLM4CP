function vb_dots = init_progress_dots( val, dots )
% This function distributes the dots for the progress bar among several
% instances.
%
%   val     The number of elements per instance
%   dots    The total number of dots
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

if ~exist('dots','var') || isempty( dots )
   dots = 50; 
end

A = val/sum(val) * dots;

B = [0,cumsum(A)];
B(end) = dots;
B = round(B);

C = diff(B);
D = round(C);

if sum(D) < dots
    for n = 1 : dots-sum(D)
        [~,ii] = max(A-D);
        D(ii) = D(ii) + 1;
    end
end

if sum(D) > dots
    for n = 1 : sum(D)-dots
        ii = find( D>0 );
        [~,ij] = min( A(ii)-D(ii) );
        D(ii(ij)) = D(ii(ij)) - 1;
    end
end

vb_dots = D;

end
