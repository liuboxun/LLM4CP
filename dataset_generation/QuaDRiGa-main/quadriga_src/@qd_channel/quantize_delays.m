function h_channel_quant = quantize_delays( h_channel, tap_spacing, max_no_taps,...
    i_rxant, i_txant, fix_taps, verbose )
%QUANTIZE_DELAYS Fixes the path delays to a grid of delay bins
%
% Calling object:
%   Single object
%
% Description:
%   For some applications, e.g. channel emulation, it is not possible to achieve an infinite delay
%   accuracy. However, when the delays are rounded to a fixed grid of delay-bins (also refereed to
%   as "taps"), the time-evolving channel is no longer smooth. When a delay "jumps" from one delay-
%   bin to the next, e.g. when a MT is moving away from the BS, the phases in the frequency domain
%   representation of the channel will suddenly change as well. Multi-carrier communications
%   systems with closed-loop channel adaption (e.g. OFDM, WiFi, LTE, etc.) will exhibit poor
%   performance in this case. This method corrects this problem by approximating the "real" delay
%   value by two delays at a fixed spacing.  For example: when we assume that the required tap
%   spacing is 5 ns (which corresponds to a 200 MHz sample-rate) and the distance between BS and MT
%   is 10 m, the delay of the LOS path would be 33.4 ns. However, the fixed tap spacing only allows
%   values of 30 or 35 ns. This method approximates the LOS delay by two taps (one at 30 and one at
%   35 ns) and linear interpolation of the path power. Hence, in the frequency domain, the
%   transition from one tap to the next is smooth. But note: this only works when the bandwidth of
%   the communication system is significantly less than the sample-rate.
%
% Input:
%   tap_spacing
%   The spacing of the delay-bin in [seconds]. Default: 5 ns
%
%   max_no_taps
%   Limits the maximum number of taps. By default, this number is infinite. If the input is
%   provided, a mapping of paths to taps is done. If the maximum number of taps is too small to
%   export all paths, only the paths with the strongest power are exported. Interpolation is done
%   whenever possible, i.e., when there are sufficient taps.
%
%   i_rxant
%   A list of receive element indices. By default, all elements are exported.
%
%   i_txant
%   A list of transmit element indices. By default, all elements are exported.
%
%   fix_taps
%   An integer number from 0 to 3, indicating if same delays should be used for different antennas
%   or snapshots. The options are: 
%
%     0  Use different delays for each tx-rx pair and for each snapshot (default)
%     1  Use same delays for all antenna pairs and snapshots (least accurate)
%     2  Use same delays for all antenna pairs, but different delays for the snapshots
%     3  Use same delays for all snapshots, but different delays for each tx-rx pair
%
%   verbose
%   Enables (1, default) or disables (0) a progress bar.
%
% Output:
%   h_channel_quant
%   A qd_channel object containing the approximated delays and channel coefficients
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

% Test if we have a scalar channel object
if numel( h_channel ) > 1
    error('QuaDRiGa:qd_channel:quantize_delays','"quantize_delays" is only defined for scalar objects.')
else
    h_channel = h_channel(1,1); % workaround for octave
end

if ~exist( 'tap_spacing' , 'var' ) || isempty( tap_spacing )
    tap_spacing = 5e-9;
end
tap_spacing = single( tap_spacing );

if ~exist( 'max_no_taps' , 'var' ) || isempty( max_no_taps )
    max_no_taps = Inf;
end

if ~exist( 'i_txant' , 'var' ) || isempty( i_txant )
    i_txant = uint32( 1:h_channel.no_txant );
else
    i_txant = uint32( i_txant );
end

if ~exist( 'i_rxant' , 'var' ) || isempty( i_rxant )
    i_rxant = uint32( 1:h_channel.no_rxant );
else
    i_rxant = uint32( i_rxant );
end

if ~exist( 'fix_taps' , 'var' ) || isempty( fix_taps )
    fix_taps = 0;
end

if exist( 'verbose' , 'var' )
    if ~( isnumeric(verbose) || ~isnan(verbose) || all(size(verbose) == [1 1]) )
        error('QuaDRiGa:qd_channel:quantize_delays','"verbose" must be numeric.')
    else
        verbose = logical(verbose);
    end
else
    verbose = true;
end

% Read common variables
no_rxant = numel(i_rxant);
no_txant = numel(i_txant);
no_ant   = no_rxant*no_txant;
no_snap  = h_channel.no_snap;
no_path  = h_channel.no_path;

% We need individual delays for each antenna
if ~h_channel.individual_delays
    h_channel = copy( h_channel );
    h_channel.individual_delays = true;
end

% Each delay is approximated by two taps, one below the target delay and one above.
% The power weighting exponent must match the target bandwidth to maintain the desired channel
% power. A value of 0.5 corresponds to the full sampling bandwidth (e.g. when 200 MHZ at 5 ns tap
% spacing).

no_coeff = double(no_rxant*no_txant*no_snap);                   % Total number of coefficients

Dn = single( h_channel.Pdelay( i_rxant, i_txant, :,: ) );       % Copy delays from input
Di = floor( Dn./tap_spacing );                                  % Delay for the lower tap
Do = (Dn - Di.*tap_spacing)./tap_spacing;                       % Relative offset
Cn = single( h_channel.Pcoeff( i_rxant, i_txant, :,: ) );       % Copy coefficients

already_quantized = all( Do(:) < 0.01 | Do(:) > 0.99 );         % Test if output is already quantized

if ~already_quantized || fix_taps
    
    if verbose                                                  % Show progress bar
        fprintf('Delay Quant. [');
        vb_dots = 50;
        tStart = clock;
        m0=0;
    end
    
    if already_quantized
        Di = [];
        Ci = [];
    else
        Di = uint16( Di )+1;                                 	% Convert to uint16
        Ci = cat( 3, ((1-Do).^0.5).*Cn, (Do.^0.5).*Cn );      	% Weight coefficients
        Di = cat( 3, Di, Di+1 );                                % Lower and upper tap
    end
    clear Do                                                  	% Free memory
    
    Dn = uint16( round(Dn./tap_spacing) )+1;                    % Round and convert to uint16
    
    if fix_taps == 1                                            % Taps are fixed for all antennas and snapshots
        D = find_optimal_delays( Dn, Cn, Di, Ci, max_no_taps ); % Best matching delays
        D = D(:,ones(1,no_coeff));                              % Expand

    elseif fix_taps == 2                                        % Taps are fixed for antennas only
        D = zeros( 1, no_coeff, 'uint16' );                     % Placeholder for delays
        for s = 1 : no_snap
            if already_quantized
                Ds = find_optimal_delays( Dn(:,:,:,s), Cn(:,:,:,s),[],[], max_no_taps );
            else
                Ds = find_optimal_delays( Dn(:,:,:,s), Cn(:,:,:,s), Di(:,:,:,s), Ci(:,:,:,s), max_no_taps );
            end
            D( 1:size(Ds,1),(s-1)*no_ant+(1:no_ant) ) = Ds(:,ones(1,no_ant));
        end
        D = D( any(D~=0,2),: );                                 % Remove unneeded delays
        
    elseif fix_taps == 3                                        % Taps are fixed for snapshots only
        D = zeros( 1, no_coeff, 'uint16' );                     % Placeholder for delays
        for a = 1 : no_ant
            [r,t] = qf.qind2sub( [no_rxant,no_txant], a );
            if already_quantized
                Ds = find_optimal_delays( Dn(r,t,:,:), Cn(r,t,:,:), [],[], max_no_taps );
            else
                Ds = find_optimal_delays( Dn(r,t,:,:), Cn(r,t,:,:), Di(r,t,:,:), Ci(r,t,:,:), max_no_taps );
            end
            D( 1:size(Ds,1),(0:no_snap-1)*no_ant+a ) = Ds(:,ones(1,no_snap));
        end
        D = D( any(D~=0,2),: );                                 % Remove unneeded delays
        
    else                                                        % Variable delay per link
        no_taps = min( size(Di,3), max_no_taps );
        D = zeros( no_taps, no_coeff, 'uint16' );               % Placeholder for delays
        
    end
    no_taps = size(D,1);                                        % Save number of taps
    
    if ~already_quantized
        Di = reshape( permute( Di, [3,1,2,4] ), [], no_coeff ); % Reorder dimensions
        Ci = reshape( permute( Ci, [3,1,2,4] ), [], no_coeff ); % Reorder dimensions
    end
    Dn = reshape( permute( Dn, [3,1,2,4] ), [], no_coeff );   	% Reorder dimensions
    Cn = reshape( permute( Cn, [3,1,2,4] ), [], no_coeff );    	% Reorder dimensions
    
    iFloor = false(2*no_path,1);
    iCeil  = iFloor;
    iFloor(1:no_path) = true;
    iCeil(no_path+1:end) = true;
    
    C = complex( zeros( no_taps, no_coeff, 'single' ) );        % Placeholder for coeffs
    
    for ic = 1 : no_coeff
        if verbose; m1=ceil(ic/no_coeff*vb_dots); if m1>m0      % Update progress bar
                for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
        
        if D(1,ic) == 0                                         % Calculate optimal delays
            if already_quantized
                Ds = find_optimal_delays( Dn(:,ic), Cn(:,ic), [],[], no_taps );
            else
                Ds = find_optimal_delays( Dn(:,ic), Cn(:,ic), Di(:,ic), Ci(:,ic), no_taps );
            end
            D( 1:size(Ds,1),ic ) = Ds;                          % Save to delay variable
        end
        
        iN = ismember( Dn(:,ic), D(:,ic) );                     % Find non-interpolated taps
        
        if already_quantized
            coeff = accumarray( Dn(iN,ic), Cn(iN,ic) );         % Combine coefficients
        else
            iC = ismember( Di(:,ic), D(:,ic) );               	% Find interpolated taps
            iC = iC(iFloor) & iC(iCeil);                        % For interpolation, the upper and lower tap must be written
            iN = iN & ~iC;                                      % Do not write non-interpolated taps that are interpolated
            iC = [ iC ; iC ];                                   % Expand vector
            coeff = accumarray( [ Di(iC,ic); Dn(iN,ic) ], [ Ci(iC,ic); Cn(iN,ic) ] );
        end
        
        [ idB, ~, cfB ] = find( coeff );                        % Get mixed data
        [iX,lX] = ismember( idB, D(:,ic) );                     % Find delay positions
        C(lX(iX),ic) = cfB;                                     % Write coefficients

    end
    clear Di Dn Ci Cn                                           % Free memory
    
    iV = max(abs(C),[],2 ) > 0;                                 % Remove taps from output that are zero
    C = reshape( C(iV,:), [], no_rxant, no_txant, no_snap );    % Format output coefficients
    C = permute( C, [2,3,1,4] );
    
    D = reshape( D(iV,:), [], no_rxant, no_txant, no_snap );    % Format output delays
    D = permute( D, [2,3,1,4] );
    D = single( D-1 ) .* tap_spacing;                           % Back to [s]
    
    h_channel_quant = qd_channel( C, D );                       % Write data to output channel object
    clear C D                                                   % Free memory
    
    if verbose
        fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
    end
    
else                                                            % Already quantized channels
    no_taps = min( h_channel.no_path, max_no_taps );            % Assemble new coefficient matrix
    if h_channel.no_path > no_taps
        Dn = reshape( permute( Dn, [3,1,2,4] ), [], no_coeff ); % Reorder dimensions
        Dn = uint16( round(Dn./tap_spacing) )+1;                % Round and convert to uint16
        Cn = reshape( permute( Cn, [3,1,2,4] ), [], no_coeff ); % Reorder dimensions
        D = zeros( no_taps, no_coeff, 'uint16' );               % Placeholder for delays
        C = complex( zeros( no_taps, no_coeff, 'single' ));     % Placeholder for coeffs
        for ic = 1 : no_coeff
            Cnn = Cn(:,ic);
            [ ~, ij ] = sort( Cnn,'descend' );                  % Sort taps in descending order
            ij = sort( ij(1:no_taps) );                         % Remove weakest taps
            C(:,ic) = Cnn(ij);                                  % Save remaining delays
            D(:,ic) = Dn(ij,ic);                                % Save remaining coefficients
        end
        C = reshape( C, no_taps, no_rxant, no_txant, no_snap );	% Format output coefficients
        C = permute( C, [2,3,1,4] );
        D = reshape( D, no_taps, no_rxant, no_txant, no_snap );	% Format output delays
        D = permute( D, [2,3,1,4] );
        D = single( D-1 ) .* tap_spacing;                       % Back to [s]
        h_channel_quant = qd_channel( C, D );                   % Write data to output channel object
    else
        h_channel_quant = qd_channel( Cn, Dn );               	% Write data to output channel object
    end
end

% Copy remaining data from original channel
h_channel_quant.name = h_channel.name;
h_channel_quant.center_frequency = h_channel.center_frequency;
h_channel_quant.par = h_channel.par;
h_channel_quant.tx_position = h_channel.tx_position;
h_channel_quant.rx_position = h_channel.rx_position;

end

% ================= find_optimal_delays ======================
function D = find_optimal_delays( Dn, Cn, Di, Ci, max_no_taps )
pdp = accumarray( Dn(:), abs(Cn(:)).^2 );               % Calculate sum-PDP of non-interpolated coefficients
[ dB,~,pdp ] = find( pdp );                             % Calculate delays from sum-PDP
dB = uint16( dB );                                      % Convert to uint16
if numel( dB ) >= max_no_taps || isempty( Di )          % Not enough taps
    [ ~,ij ] = sort( pdp,'descend' );                   % Sort taps in descending order
    ij = sort( ij(1:min(max_no_taps,numel(dB))) );      % Remove weakest taps
    D = dB(ij);                                         % Best matching delays
else
    pdp = accumarray( Di(:), abs(Ci(:)).^2 );           % Calculate sum-PDP of interpolated coefficients
    D = uint16( find( pdp ) );                          % Calculate delays from sum-PDP
    if numel( D ) > max_no_taps                         % There are not enough taps for full interpolation
        dC = D( ~ismember(D,dB) );                      % Delays that are added by interpolation
        [ ~,ij ] = sort( pdp(dC),'descend' );           % Sort taps in descending order
        ij = sort( ij(1:max_no_taps-numel(dB)) );       % Remove weakest taps
        D = sort([dB;dC(ij)]);                          % Mixed taps
    end
end
end
