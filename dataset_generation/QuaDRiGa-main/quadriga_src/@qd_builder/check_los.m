function isLOS = check_los( h_builder, threshold )
%CHECK_LOS Checks for the LOS and GR paths
%
% Calling object:
%   Object array
%
% Description:
%   This method checks if the 'qd_builder' object has a line-of-sight (LOS) path and a ground-
%   reflection (GR) path. This is done by calculating the delays and the departure and arrival
%   angles of the LOS and GR path from the TX and RX positions. Then, the provided delays and
%   angles in the small-scale-fading (SSF) parameters 'qd_builder.taus', 'qd_builder.AoD',
%   'qd_builder.AoA', 'qd_builder.EoD', and 'qd_builder.EoA' are compared to the values obtained
%   from the positions to determine if the LOS or GR paths are present. Since QuaDRiGa requires
%   that the LOS path is always included in the calculations, a virtual zero-power LOS path is
%   added to the list of paths if there is no LOS path found.
%
% Input:
%   threshold
%   The minimum Ricean K-Factor in [dB] for the LOS detection. For values below this threshold, the
%   output 'isLOS' will be 0 even if there is a LOS path included in the path-list. Default
%   threshold: -30 dB.
%
% Output:
%   isLOS
%   Indicator of the LOS state of the builder with the following values:
%
%   isLOS = -2  The 'qd_builder' object has no RX positions associated to it.
%
%   isLOS = -1  The 'qd_builder' object has RX positions, but the SSF parameters were not initialized.
%
%   isLOS =  0  All TX-RX links in the builder are NLOS (i.e., the KF obtained from the path gains
%               is below the 'threshold'.
%
%   isLOS =  1  There is at least one TX-RX link in the builder that is LOS (i.e., the KF obtained
%               from is above the 'threshold'.
%
%   isLOS =  2  There is at least one TX-RX link in the builder that is LOS and the second cluster
%               of all TX-RX links is the GR path (i.e., its delays and angles match to the
%               theoretic GR path-delays and angles).
%
%   isLOS =  3  There is a GR path in the builder (i.e., delays and angles of the second cluster
%               match the theoretic GR path-delays and angles), but there is no LOS link. 
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

if ~exist( 'threshold','var' ) || isempty( threshold )
    threshold = -30;
end

if numel(h_builder) > 1
    
    % Recursive call for all objects in the array
    sic = size( h_builder );
    if numel( sic ) > 2
        error('QuaDRiGa:qd_builder:check_los','Array of qd_builder objects cannot have more that 2 dimensions.');
    end
    isLOS = zeros(sic);
    for i_cb = 1 : numel(h_builder)
        [ i1,i2 ] = qf.qind2sub( sic, i_cb );
        isLOS(i1,i2) = check_los( h_builder( i1,i2 ), threshold );
    end
    
else
    
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Convert threshold to linear scale
    threshold = 10^(0.1*threshold );
    
    % Dual Mobility indicator
    if h_builder.dual_mobility == -1 
        h_builder.check_dual_mobility( false );
    end
    
    if h_builder.no_rx_positions == 0
        isLOS = -2;
        
    elseif isempty( h_builder.NumSubPaths ) || isempty( h_builder.taus ) || isempty( h_builder.gain ) ||...
            isempty( h_builder.AoD ) || isempty( h_builder.AoA ) || isempty( h_builder.EoD ) || ...
            isempty( h_builder.EoA ) 
        isLOS = -1;
        
    else
        
        % Get required variables
        tx_pos = h_builder.tx_position;
        rx_pos = h_builder.rx_positions;
        los_angles = h_builder.get_angles*pi/180;
        n_mobiles = h_builder.no_rx_positions;
        n_freq = h_builder.no_freq;
        
        % Calculate LOS-delay
        dTR = sqrt(sum((tx_pos - rx_pos).^2));
        los_delay = dTR ./ qd_simulation_parameters.speed_of_light;
        
        % Angles must be correct within 0.1 degree
        % Delays must be correct within 0.3 ns (10 cm position accuracy)
        if h_builder.NumSubPaths(1) == 1 && ...
                all( abs( angle(exp(1j*(h_builder.AoD(:,1) - los_angles(1,:)'))) ) < 0.0017 ) && ...
                all( abs( angle(exp(1j*(h_builder.AoA(:,1) - los_angles(2,:)'))) ) < 0.0017 ) && ...
                all( abs( h_builder.EoD(:,1) - los_angles(3,:)' ) < 0.0017 ) && ...
                all( abs( h_builder.EoA(:,1) - los_angles(4,:)' ) < 0.0017 )
            
            if all( abs( max( h_builder.taus(:,1,:),[],3 ) ) < 0.3e-9 )
                isLOS = 1;
            elseif all(all( abs( h_builder.taus(:,1,:) - repmat( los_delay' ,[1,1,size(h_builder.taus,3)]) ) < 0.3e-9 ))
                h_builder.taus = h_builder.taus - repmat( los_delay' ,[1,size(h_builder.taus,2),size(h_builder.taus,3)]);
                isLOS = 1;
            else
                isLOS = 0;
            end
        else
            isLOS = 0;
        end
        
        if isLOS == 0                           % LOS component is missing
            % Add virtual zero-power LOS component
            h_builder.NumSubPaths   = [ 1,h_builder.NumSubPaths ];
            h_builder.NumClusters   = numel( h_builder.NumSubPaths );
            h_builder.taus          = cat(2, zeros( n_mobiles,1,size(h_builder.taus,3) ), h_builder.taus );
            h_builder.gain          = cat(2, zeros( n_mobiles,1,n_freq ), h_builder.gain );
            h_builder.AoD           = [ los_angles(1,:)', h_builder.AoD ];
            h_builder.AoA           = [ los_angles(2,:)', h_builder.AoA ];
            h_builder.EoD           = [ los_angles(3,:)', h_builder.EoD ];
            h_builder.EoA           = [ los_angles(4,:)', h_builder.EoA ];
            h_builder.xprmat        = cat(2, repmat( [1;0;0;-1], [1,1,n_mobiles,n_freq] ), h_builder.xprmat );
            h_builder.pin           = cat(2, zeros( n_mobiles,1,n_freq ), h_builder.pin );
            h_builder.subpath_coupling = cat(2, rand( 4,1,n_mobiles ), h_builder.subpath_coupling );
            
        elseif size(h_builder.gain,2) == 1      % There is only a LOS component, but no NLOS paths
            % Use average path gain as NLOS reference
            pg = 10.^(-0.1*h_builder.get_pl);
            if all(all( permute( h_builder.gain,[3,1,2] ) < threshold*pg ))    
                 isLOS = 0;
            end
            
        else
            % Use KF as a reference
            kf = h_builder.gain(:,1,:) ./ sum(h_builder.gain(:,2:end,:),2);
            if all( kf(:) < threshold )
                isLOS = 0;
            end
        end
        
        % Check for ground reflection component
        % Angles must be correct within 0.1 degree
        % Delays must be correct within 0.3 ns (10 cm position accuracy)
        if numel( h_builder.NumSubPaths ) > 1 && h_builder.NumSubPaths(2) == 1
            dTR_gr = sqrt(sum(abs(  [ rx_pos(1:2,:);-rx_pos(3,:) ] - tx_pos ).^2,1));
            tau_gr = (dTR_gr-dTR)./qd_simulation_parameters.speed_of_light;
            if all( abs( h_builder.taus(:,2) - tau_gr.' ) < 0.3e-9 ) && ...
                    all( abs( angle(exp(1j*(h_builder.AoD(:,2) - los_angles(1,:)'))) ) < 0.0017 ) && ...
                    all( abs( angle(exp(1j*(h_builder.AoA(:,2) - los_angles(2,:)'))) ) < 0.0017 ) && ...
                    all( h_builder.EoD(:,2) < 0 ) && ...
                    all( abs( h_builder.EoD(:,2) - h_builder.EoA(:,2) ) < 0.0017 )
                if isLOS == 1
                    isLOS = 2;
                else
                    isLOS = 3;
                end
            end
        end
    end
end

end
