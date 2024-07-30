function remove_bias( h_sos, ca, cb )
%REMOVE_BIAS Removes the bias from the SOS generator for the given set of positions and SOS phases
%
% Calling object:
%   Single object
%
% Description:
%   There is a possibility that an ill-adjusted set of SOS phases causes a bias for a given set of
%   positions. The resulting random variable is then not Normal-distributed with zero mean and unit
%   variance, but with a (small) non-zero mean. This can propagate to later processing steps that
%   depend on the random variables. The bias can be removed by adding an offset to the phases of
%   the SOS generator that forces the mean to be zero for a set of given positions. However, this
%   operation is computing-intense since it uses an iterative approach to calculate the phase
%   offset.
%
% Input:
%   ca
%   Coordinates for the first mobile device in [m] given as [3 x N] matrix. The rows correspond to
%   the x,y and z coordinate.
%
%   cb
%   Coordinates for the corresponding second mobile device in [m] given as [3 x N] matrix. The rows
%   correspond to the x,y and z coordinate. This variable must either be empty or have the same
%   size as "ca".
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

if ~exist( 'cb','var' ) || isempty( cb )
    cb = [];
elseif any( size(cb) ~= size(ca) )
    error('QuaDRiGa:qd_sos:remove_bias','Size of "cb" mut match the size of "ca".' );
end

if numel(h_sos) > 1
    
    sic = size( h_sos );
    for n = 1 : prod( sic )
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
        remove_bias( h_sos(i1,i2,i3,i4), ca, cb );
    end
    
else
    % Fix for Ocatave
    h_sos = h_sos(1,1);
    
    % Read esisting phases
    r = h_sos.sos_phase;
    
    % Check if phases are identical
    use_same = all( r(:,1) == r(:,2) );
    
    if use_same
        ref = 1;
    elseif ~use_same && isempty( cb )
        cb  = ca(:,randperm( size(ca,2) ));        % Use random permutation of ca as cb
        ref = [1,2];
    else
        ref = [1,2,1,2];
    end
    
    % Find optimal offset to remove the bias from the SOS generators
    rN = r;
    for n = 1 : numel(ref)
        cD  = 0;
        stp = pi;
        cX  = 0;
        for m = 1 : 30
            cD = cD + stp;
            if use_same
                rN = r + cD;
            else
                rN(:,ref(n)) = r(:,ref(n)) + cD;
            end
            cN = abs( mean( h_sos.val( ca, cb, rN )));
            if cN > cX
                stp = -0.4 * stp;
            end
            cX = cN;
        end
        r = rN;
    end
    
    if use_same
        h_sos.sos_phase = repmat( angle(exp(1j*r(:,1))),1,2);
    else
        h_sos.sos_phase = angle(exp(1j*r));
    end
end

end
