function [ cf, dl, ip2, ramp ] = merge_coeff( cf1, dl1, cf2, dl2, ip1, gr1, gr2, gro )
%MERGE_COEFF Merges the coefficients
%
% Input:
%   cf1         Coefficients of channel 1           [ R x T x L1 x S ]
%   dl1         Delays of channel 1                 [ R x T x L1 x S ]
%   cf2         Coefficients of channel 2           [ R x T x L2 x S ]
%   dl2         Delays of channel 2                 [ R x T x L2 x S ]
%
%   ip1         The tap positions in the output channel at the beginning [ 1 x L1 ]
%
%   gr1         Indicator (1/0) if channel 1 has a GR on tap 2
%   gr2         Indicator (1/0) if channel 2 has a GR on tap 2
%   gro         Indicator (1/0) if output channel should have a GR on tap 2
%
% Output:
%   cf          Merged coefficients                 [ R x T x L x S ]
%   dl          Merged delays                       [ R x T x L x S ]
%   ip          Tap positions for the merger        [ 1 x L ]
%   ip2         Tap positions for the next segment  [ 1 x L2 ]
%
%   L = max(L1,L2)+1;
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

% Sanity check
R  = size(cf1,1);
T  = size(cf1,2);
S  = size(cf1,4);
L1 = size(cf1,3);
L2 = size(cf2,3);
if size(dl1,1) ~= R || size(dl1,2) ~= T || size(dl1,3) ~= L1 || size(dl1,4) ~= S || ...
        size(cf2,1) ~= R || size(cf2,2) ~= T || size(cf2,3) ~= L2 || size(cf2,4) ~= S || ...
        size(dl2,1) ~= R || size(dl2,2) ~= T || size(dl2,3) ~= L2 || size(dl2,4) ~= S || ...
        numel(ip1) ~= L1
    error('QuaDRiGa:qd_channel:merge','Input sized do not match')
end

% Parse the GR switches
if ~exist('gr1','var') || isempty( gr1 )
    gr1 = false;
end
if ~exist('gr2','var') || isempty( gr2 )
    gr2 = false;
end
if ~exist('gro','var') || isempty( gro )
    gro = false;
end
gro = gr1 | gr2 | gro;

% Determine the number of NLOS paths
if gr1
    L1n = L1-2;         % Channel 1 NLOS paths
    L1s = 3;            % Channel 1 first NLOS index
else
    L1n = L1-1;         % Channel 1 NLOS paths
    L1s = 2;            % Channel 1 first NLOS index
end
if gr2
    L2n = L2-2;         % Channel 2 NLOS paths
    L2s = 3;            % Channel 2 first NLOS index
else
    L2n = L2-1;         % Channel 2 NLOS paths
    L2s = 2;            % Channel 2 first NLOS index
end

% The track is split in sub-segments. Within each sub-segment, one path of the old channel
% ramps down and a new path ramps up.
no_subseg = min([L1n,L2n,S]);

% Number of paths in the output channel
L = L1 + L2; %max( L1, L2 )+1;
if gro
    L = L+1;
end

% Initialize ip matrix
ip = zeros( 1,L );
ip( ip1 ) = 1;          % The currently used positions by the old segment
if gr2 || gro
    ip( 2 ) = 1;        % Block GR position
end

% Initialize empty tap matrix for the ramp-up taps
ip2 = zeros(1,L2);
ip2(1) = 1;             % LOS
if gr2
    ip2(2) = 2;         % GR
end

% Initialize output coefficient matrix
cf = zeros( R,T,L,S );
dl = zeros( R,T,L,S );

% Copy the old data to the output
cf(:,:,ip1,:) = cf1;
dl(:,:,ip1,:) = dl1;

% The initial order of the paths. This is later updated to find the optimal ramp up/down
% order that maintains the DS.
o1 = 1:L1;
o2 = 1:L2;

if no_subseg > 0
    % Calculate the average DS in the two segments
    [ ds1, p1, d1 ] = merging_avg_ds( cf1, dl1 );
    [ ds2, p2, d2 ] = merging_avg_ds( cf2, dl2 );
    
    % Set the target DS for each sub-segment
    weight = (1/(2*no_subseg) : 1/no_subseg : 1).';
    ds_target = single( ds1 + ( ds2-ds1 ) * weight );
    
    % The power ramp for LOS, GR, and taps without partner
    ramp_opt = single( 0.5*(1+sin((weight-0.5)*pi)) );
    
    % Find the optimal order of the ramping down/up taps that maintain the DS
    pP = single( [ p1 ; p2 ]');
    dD = single( [ d1 ; d2 ] );
    for cc = 1:2
        % Find optimal tap ordering for segment A
        for ca = L1s : L1
            tmp = o1( o1 ~= ca );        	% Remove tap from set
            cost = ones( L1,1 )*Inf;       	% Initialize costs
            for cb = L1s : L1
                % Place tap at new position and calculate costs
                o1c = [ tmp(1:cb-1) , ca , tmp(cb:end) ];
                cost(cb) = merging_cost_fcn( o1c , o2 , pP , dD , ds_target , ramp_opt, L1s, L2s );
            end
            [~,cb] = min(cost);             % Look for minimum costs
            o1 = [ tmp(1:cb-1) , ca , tmp(cb:end) ];  % Reorder
        end
        
        % Find optimal tap ordering for segment B
        for ca = L2s : L2
            tmp = o2( o2 ~= ca );        	% Remove tap from set
            cost = ones( L2,1 )*Inf;       	% Initialize costs
            for cb = L2s : L2
                % Place tap at new position and calculate costs
                o2c = [ tmp(1:cb-1) , ca , tmp(cb:end) ];
                cost(cb) = merging_cost_fcn( o1 , o2c , pP , dD , ds_target , ramp_opt, L1s, L2s );
            end
            [~,cb] = min(cost);             % Look for minimum costs
            o2 = [ tmp(1:cb-1) , ca , tmp(cb:end) ];  % Reorder
        end
    end
    
    % Calculate the ramp for each sub-interval
    ramp_length = floor( S/no_subseg );
    ramp = (1:ramp_length)/(ramp_length+1);
    ramp = 0.5*(1+sin((ramp-0.5)*pi));
    ramp = permute( ramp , [1,3,4,2] );
    ramp = ramp( ones(1,R), ones(1,T),1,: );
    ramp_down = sqrt( 1-ramp );
    ramp_up   = sqrt( ramp );
    
    % Remove deterministic components from paring vectors
    o1 = o1( L1s : end );
    o2 = o2( L2s : end );
    
    for l = 1 : no_subseg
        % Calculate the segments
        ind2 = (l-1)*ramp_length+1 : l*ramp_length;     % Current tap
        ind3 = l*ramp_length+1 : S;                     % Next taps
        
        % Ramp down old path
        i_down = ip1( o1(l) );                          % Index of the ramp-down path
        cf( :,:,i_down,ind2 ) = cf( :,:,i_down,ind2 ) .* ramp_down;
        cf( :,:,i_down,ind3 ) = 0;
        dl( :,:,i_down,ind3 ) = 0;
        
        % Ramp up new path
        [~,i_up] = min( ip );                           % Find empty slot
        cf( :,:,i_up,ind2 ) = cf2(:,:,o2(l),ind2) .* ramp_up;
        cf( :,:,i_up,ind3 ) = cf2(:,:,o2(l),ind3);
        dl( :,:,i_up,ind2 ) = dl2(:,:,o2(l),ind2);
        dl( :,:,i_up,ind3 ) = dl2(:,:,o2(l),ind3);
        
        ip2( o2(l) ) = i_up;                            % Store new tap position
        ip( i_up )   = true;                            % Lock slot for new tap
        ip( i_down ) = -1;                              % Free slot for next tap
        % -1 gives priority to previously down taps
    end
else
    % Remove deterministic components from pairing matrix
    o1 = o1( L1s : end );
    o2 = o2( L2s : end );
end

% Calculate the ramp for the entire interval
ramp = (1:S)/(S+1);
ramp = 0.5*(1+sin((ramp-0.5)*pi));
ramp = permute( ramp , [1,3,4,2] );
ramp = ramp( ones(1,R), ones(1,T),1,: );
ramp_down  = sqrt( 1-ramp );
ramp_up    = sqrt( ramp );

% Ramp down all remaining NLOS taps from the old segment
if L1n > no_subseg
    for l = no_subseg+1 : L1n
        i_down = ip1( o1(l) );                      % Index of the ramp-down path
        cf( :,:,i_down,: ) = cf( :,:,i_down,: ) .* ramp_down;
        ip( i_down ) = -1;                          % Indicate ramp-down position
    end
end

% Ramp up all remaining NLOS taps
if L2n > no_subseg
    for l = no_subseg+1 : L2n
        i_up = find( ip == 0, 1 );                  % Get empty slot
        cf( :,:,i_up,: ) = cf2(:,:,o2(l),:) .* ramp_up;
        dl( :,:,i_up,: ) = dl2(:,:,o2(l),:);
        ip2( o2(l) ) = i_up;                        % Store new tap position
        ip( i_up )   = 1;                           % Lock slot for new tap
    end
end

% Merge the LOS components
cf(:,:,1,:) = cf1(:,:,1,:) .* (1-ramp) + cf2(:,:,1,:) .* ramp;
tmp2 = dl1(:,:,1,:) - dl1(:,:,1,:);
if any( abs( tmp2(:) ) > 1e-14 )
    warning('QuaDRiGa:qd_channel:merge','LOS coefficients have different delays.')
    dl(:,:,1,:) = dl1(:,:,1,:) .* (1-ramp) + dl2(:,:,1,:) .* ramp;
end

% Merge ground reflection
if gr1 && gr2
    cf(:,:,2,:) = cf1(:,:,2,:) .* (1-ramp) + cf2(:,:,2,:) .* ramp;
    dl(:,:,2,:) = dl1(:,:,2,:) .* (1-ramp) + dl2(:,:,2,:) .* ramp;
elseif gr1 && ~gr2
    cf(:,:,2,:) = cf1(:,:,2,:) .* ramp_down;
    dl(:,:,2,:) = dl1(:,:,2,:);
elseif ~gr1 && gr2
    cf(:,:,2,:) = cf2(:,:,2,:) .* ramp_up;
    dl(:,:,2,:) = dl2(:,:,2,:);
end

% Return ramp for merging the positions and the PG
ramp = reshape( ramp(1,1,1,:) , 1,S );

% Remove zero-valued channels at the end of the merge
last_ip = find( ip ~= 0,1,'last' );
cf = cf(:,:,1:last_ip,:);
dl = dl(:,:,1:last_ip,:);


if 0 % For debugging
    %%
    ds1i = qf.calc_delay_spread_coeff(cf1,dl1);
    ds2i = qf.calc_delay_spread_coeff(cf2,dl2);
    dso = qf.calc_delay_spread_coeff(cf,dl);
       
    [ mse , weight, ds_opt ] = merging_cost_fcn( [1:L1s-1,o1] , [1:L2s-1,o2] , pP , dD , ds_target , ramp_opt, L1s, L2s );
    
    plot( 1:S, ds1i,'m','Linewidth',2 )
    hold on
    plot( 1:S, ds2i,'c','Linewidth',2 )
    
    pt = ramp_length/2 : ramp_length : S;
    pt = pt( 1:no_subseg )';
    plot( pt, [ds_target,ds_opt] ,'-o' )
    plot( 1:S, dso,'r','Linewidth',2 )
    plot( 1,ds1,'ms' )
    plot( S,ds2,'cs' )
    hold off
    legend('First segment','Second segment','Target','Optimal Ramp','Output')
    xlabel('Snapshot number')
    ylabel('Delay Spread [s]')
    grid on
    
    1
end
    

end

