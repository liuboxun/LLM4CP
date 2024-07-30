function gen_fbs_lbs( h_builder, usage, PerClusterAS )
%GEN_FBS_LBS Generates the positions of the FBS and the LBS
%
% Calling object:
%   Object array
%
% Description:
%   This method calculates the positions of the first-bounce scatterer (FBS) and the last-bounce-
%   scatterer (LBS) in 3D Cartesian coordinates. This is done by splitting each path into two
%   bounces and minimizing the distance between FBS and LBS. If resulting optimization problem has no
%   solution, a single-bounce model is used where FBS = LBS. The generation of scatterer positions
%   requires that the LSF parameters are initialized first. The output of this method is written to
%   the object properties.
%
% Scatterer pos.:
%   fbs_pos         The positions of the first-bounce scatterers in 3D Cartesian coordinates
%   lbs_pos         The positions of the last-bounce scatterers in 3D Cartesian coordinates
%
% Input:
%   usage
%   Controls the behavior of the method: If set to 0, all existing LSF parameters will be discarded
%   and the method exits. By default (1), new LSF parameters will be created, existing ones will be
%   replaced. It set to 2, existing LSF parameters will be reused and missing ones will be created.
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

% Parse input variables.
if ~exist( 'usage','var' ) || isempty( usage )
    usage = 1;
end

if ~exist( 'PerClusterAS','var' ) || isempty( PerClusterAS )
    PerClusterAS = [];
end

if numel(h_builder) > 1
    
    % Recursive call for all objects in the array
    sic = size( h_builder );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        gen_fbs_lbs( h_builder( i1,i2,i3,i4 ), usage, PerClusterAS );
    end
    
else
    
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Remove existing scatterer positions
    if usage == 0 || usage == 1
        h_builder.fbs_pos = [];
        h_builder.lbs_pos = [];
    end
    
    % Dual Mobility indicator
    if h_builder.dual_mobility == -1 
        h_builder.check_dual_mobility( false );
    end
    
    % Check if we need to do anything else
    if usage == 0 || h_builder.no_rx_positions == 0 || h_builder.simpar(1,1).use_3GPP_baseline
        return
    end
    
    % Get required variables
    tx_pos = h_builder.tx_position;
    rx_pos = h_builder.rx_positions;
    taus = h_builder.taus;
    AoD = h_builder.AoD;
    AoA = h_builder.AoA;
    EoD = h_builder.EoD;
    EoA = h_builder.EoA;
    SubPathCPL = h_builder.subpath_coupling;
    n_mobiles = size( rx_pos, 2 );
    n_clusters = size( taus, 2 );
    speed_of_light = qd_simulation_parameters.speed_of_light;
    o_mobiles = ones(1,n_mobiles);
    n_freq = size( SubPathCPL,3 );
    o_freq = ones(1,n_freq);
    NumSubPaths = h_builder.NumSubPaths;
    
    if isempty( NumSubPaths )
       NumSubPaths = ones( 1, size( taus, 2 ) );
    end
    n_paths = sum( NumSubPaths );
    o_paths = ones(1,n_paths);
    
    if isempty( h_builder.scenpar ) && isempty( PerClusterAS )
        PerClusterAS = zeros( 4, n_clusters, n_freq );
    elseif isempty( PerClusterAS )
        PerClusterAS = repmat( [h_builder.scenpar.PerClusterAS_D; h_builder.scenpar.PerClusterAS_A;...
            h_builder.scenpar.PerClusterES_D; h_builder.scenpar.PerClusterES_A ], [1,n_clusters,n_freq] );
    elseif n_freq > 1 && size(PerClusterAS,3) == 1
        PerClusterAS = repmat( PerClusterAS, [1,1,n_freq] );
    end
    
    % Distances between BS and MT
    d_2d = hypot( tx_pos(1,:) - rx_pos(1,:), tx_pos(2,:) - rx_pos(2,:) );
    d_2d( d_2d<1e-5 ) = 1e-5;
    
    % Calculate angles between BS and MT
    angles = zeros( 5,n_mobiles );
    angles(1,:) = atan2( rx_pos(2,:) - tx_pos(2,:) , rx_pos(1,:) - tx_pos(1,:) );   % Azimuth at BS
    angles(2,:) = pi + angles(1,:);                                                 % Azimuth at MT
    angles(3,:) = atan2( ( rx_pos(3,:) - tx_pos(3,:) ), d_2d );                     % Elevation at BS
    angles(4,:) = -angles(3,:);                                                     % Elevation at MT
    angles = angles.';
    
    % Deterimne the number of direct paths
    if NumSubPaths(1) == 1 && ...
            all( abs( exp(1j*AoD(:,1)) - exp(1j*angles(:,1)) ) < 0.0017 ) && ...
            all( abs( EoD(:,1) - angles(:,3) ) < 0.0017 ) && ...
            all( abs( exp(1j*AoA(:,1)) - exp(1j*angles(:,2)) ) < 0.0017 ) && ...
            all( abs( EoA(:,1) - angles(:,4) ) < 0.0017 ) && ...
            all( abs( taus(:,1) ) < 0.3e-9 )
        if n_clusters>1 && NumSubPaths(2) == 1 && ...
                all( abs( AoD(:,2) - AoD(:,1) ) < 0.0017 ) && ...    % GR has same azimuth angles as LOS path has.
                all( abs( AoA(:,2) - AoA(:,1) ) < 0.0017 ) && ...
                all( EoD(:,2) < 0 ) && ...
                all( abs( EoD(:,2) - EoA(:,2) ) < 0.0017 )  % 0.01°
            n_los_clusters = 2;     % First path is LOS path and second path is GR path
        else
            n_los_clusters = 1;     % First path is LOS path
        end
    else
        n_los_clusters = 0;         % Only NLOS paths are given
    end
    
    if n_los_clusters == 0 && any( abs(taus( :,1 )) < 0.3e-9 )
        error('QuaDRiGa:qd_builder:generate_fbs_lbs','No LOS paths are defined, but delays have a value of 0.')
    end
    
    % Scale angles for Laplacian angular power spectrum
    if ~isempty( h_builder.scenpar ) && strcmp(h_builder.scenpar.SubpathMethod,'Laplacian')
        use_laplacian_pas = true;
    else
        use_laplacian_pas = false;
    end
    
    % Combine cluster angles [ n_mobiles x n_clusters x 4 ]
    Ac = cat(3,AoD,AoA,EoD,EoA);
    
    % Placeholder for the sub-path angles  [ n_mobiles x n_paths x 4 x n_freq ]
    As = zeros( n_mobiles,n_paths,4,n_freq );
    
    % The start and end-indices for the sub-paths [ 1 x n_clusters ]
    Ms = [0,cumsum(NumSubPaths(1:end-1))]+1;
    Me = cumsum(NumSubPaths);
    
    for iF = 1 : n_freq              % Frequency
        for iA = 1 : 4          % AoD, AoA, EoD, EoA
            
            % Apply LOS angles
            As(:,1:n_los_clusters,iA,iF) = Ac(:,1:n_los_clusters,iA);
            
            % Create sub-path angles
            if any(NumSubPaths ~= 1)
                for iL = n_los_clusters+1 : n_clusters       % Cluster
                    
                    % Get the offset angles
                    if NumSubPaths(iL) == 1
                        of = 0;
                    else
                        of = get_subpath_angles( NumSubPaths(iL),[],use_laplacian_pas );
                    end
                    
                    % Add subpath-angles to cluster angles
                    ax = o_mobiles' * PerClusterAS(iA,iL,iF) * of;% + Ac(:,iL,iA) * ones(1,NumSubPaths(iL));
                    
                    % Change order to account for random sub-path coupling
                    [~,ind] = sort( SubPathCPL(iA,Ms(iL):Me(iL),iF) );
                    
                    % Store subpath angles
                    As(:,Ms(iL):Me(iL),iA,iF) = ax(:,ind);
                end
            end
        end
        
        % Apply cluster angles
        for iL = n_los_clusters+1 : n_clusters
            [As(:,Ms(iL):Me(iL),1,iF), As(:,Ms(iL):Me(iL),3,iF)] =...
                rotate_sphere( As(:,Ms(iL):Me(iL),1,iF), As(:,Ms(iL):Me(iL),3,iF), Ac(:,iL,1), Ac(:,iL,3) );
            [As(:,Ms(iL):Me(iL),2,iF), As(:,Ms(iL):Me(iL),4,iF)] =...
                rotate_sphere( As(:,Ms(iL):Me(iL),2,iF), As(:,Ms(iL):Me(iL),4,iF), Ac(:,iL,2), Ac(:,iL,4) );
        end
    end
    
    % Create clusters
    
    % The distance vector from the Tx to the initial position
    r = rx_pos - tx_pos;
    norm_r = sqrt(sum(r.^2)).';
    
    % Prepare output variables
    fbs_pos = zeros(n_mobiles,n_paths,3,n_freq);
    lbs_pos = zeros(n_mobiles,n_paths,3,n_freq);
    
    % Get the total path length of the NLOS component
    dist = clst_expand( taus(:,:,1), NumSubPaths );  % Delay for each sub-path
    dist = dist * speed_of_light + norm_r(:,o_paths);
    
    for iF = 1 : n_freq
        
        if iF > 1 && size( taus,3) == n_freq
            dist = clst_expand( taus(:,:,iF), NumSubPaths );  % Delay for each sub-path
            dist = dist * speed_of_light + norm_r(:,o_paths);
        elseif iF > 1 && size( taus,3 ) ~= 1
            error('QuaDRiGa:qd_builder:generate_fbs_lbs','Size of "taus" does not match number of frequencies.')
        end
        
        phi_d_lm    = As(:,:,1,iF); % AoD [ n_mobiles x n_paths ]
        theta_d_lm  = As(:,:,3,iF); % EoD [ n_mobiles x n_paths ]
        phi_a_lm    = As(:,:,2,iF); % AoA [ n_mobiles x n_paths ]
        theta_a_lm  = As(:,:,4,iF); % EoA [ n_mobiles x n_paths ]
        
        % Get the direction of the last bounce scatterer (LBS) seen from the receivers position.
        [ ahat_lm_x, ahat_lm_y, ahat_lm_z ] = sph2cart(phi_a_lm, theta_a_lm, 1);
        ahat_lm = cat(3, ahat_lm_x, ahat_lm_y, ahat_lm_z );
        ahat_lm = permute(ahat_lm,[3,1,2]);  % [ 3 x n_mobiles x n_paths  ]
        
        % Get the direction of the first bounce scatterer (FBS) seen from the transmitter center position.
        [ bhat_lm_x, bhat_lm_y, bhat_lm_z ] = sph2cart(phi_d_lm, theta_d_lm, 1);
        bhat_lm = cat(3, bhat_lm_x, bhat_lm_y, bhat_lm_z );
        bhat_lm = permute(bhat_lm,[3,1,2]);  % [ 3 x n_mobiles x n_paths ]
        
        % This implements the "bounce2" model
        [ norm_a_lm, norm_b_lm,~,valid ] = solve_multi_bounce_opti( ahat_lm, bhat_lm, r, dist, 2 );
        norm_a_lm = permute( norm_a_lm, [2,3,1] );
        norm_b_lm = permute( norm_b_lm, [2,3,1] );
        valid = permute( valid, [2,3,1] );
        
        if n_los_clusters > 0.5     % LOS
            norm_a_lm(:,1) = 0.5*norm_r;
            norm_b_lm(:,1) = 0.5*norm_r;
            valid(:,1) = true;
        end
        if n_los_clusters > 1.5     % GR
            valid(:,2) = false;  % Use single-bounce for ground reflection
        end
        
        % For the invalid paths, use single-bounce-model
        iv = ~valid;
        if any( iv(:) )
            ahat_iv = [ reshape( ahat_lm_x(iv),1,[]) ; reshape( ahat_lm_y(iv),1,[]) ; reshape( ahat_lm_z(iv),1,[]) ];
            rx = r(o_paths,:)'; ry = r(o_paths*2,:)'; rz = r(o_paths*3,:)';
            rx = [ reshape(rx(iv),1,[]) ; reshape(ry(iv),1,[]) ; reshape(rz(iv),1,[]) ];
            
            [ b, norm_b_lm(iv), norm_a_lm(iv) ] = solve_cos_theorem( ahat_iv , rx , reshape(dist(iv),1,[])  );
            
            tmp = reshape(norm_b_lm(iv),1,[]);
            bhat_lm_x(iv) = b(1,:)./tmp;
            bhat_lm_y(iv) = b(2,:)./tmp;
            bhat_lm_z(iv) = b(3,:)./tmp;
        end
        
        % Calculate the FBS position (relative to initial Rx-pos)
        fbs_pos(:,:,1,iF) = norm_b_lm .* bhat_lm_x - r(o_paths,:).';
        fbs_pos(:,:,2,iF) = norm_b_lm .* bhat_lm_y - r(o_paths*2,:).';
        fbs_pos(:,:,3,iF) = norm_b_lm .* bhat_lm_z - r(o_paths*3,:).';
        
        % Calculate the LBS position
        lbs_pos(:,:,1,iF) = norm_a_lm .* ahat_lm_x;
        lbs_pos(:,:,2,iF) = norm_a_lm .* ahat_lm_y;
        lbs_pos(:,:,3,iF) = norm_a_lm .* ahat_lm_z;
    end
    
    % Calculate the scatterer positions in global coordinates.
    fbs_pos = fbs_pos + permute( rx_pos(:,:,o_paths,o_freq) , [2,3,1,4] );
    lbs_pos = lbs_pos + permute( rx_pos(:,:,o_paths,o_freq) , [2,3,1,4] );
    
    % Reshape output to [ 3 x n_paths x n_mobiles x n_freq ]
    if isempty( h_builder.fbs_pos )
        h_builder.fbs_pos = permute( fbs_pos,[3,2,1,4] );
    end
    if isempty( h_builder.lbs_pos )
        h_builder.lbs_pos = permute( lbs_pos,[3,2,1,4] );
    end
    
end
end
