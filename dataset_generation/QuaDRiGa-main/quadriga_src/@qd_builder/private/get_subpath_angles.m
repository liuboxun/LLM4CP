function [ aod,eod,aoa,eoa,delay ] = get_subpath_angles( h_builder,i_mobile, use_laplacian_pas )
%GET_SUBPATH_ANGLES Generate subpaths and perform random coupling according to the 3GPP baseline model
%
%   GET_SUBPATH_ANGLES generates the subpaths around the each path and
%   randomly couples the subpaths on the Tx- and Rx side. 
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

if ~exist( 'h_builder','var' ) || isempty(h_builder)
    h_builder = 20;
end
if isa( h_builder, 'qd_builder' )
    use_qd_builder  = true;
else
    use_qd_builder  = false;
    n_clusters      = 1;
    n_subpath       = h_builder;
    n_paths         = n_subpath;
end
 
% The optimized order of the 20 sub-paths for 
offset = [-0.8844 1.1481 0.6797 -1.1481 -0.6797 1.5195 -0.0447 0.3715 -1.5195,...
    -0.2492 -0.5129 0.5129 0.8844 -2.1551 -0.1413 0.1413 0.0447 2.1551 -0.3715 0.2492 ];

if use_qd_builder
    n_clusters  = h_builder.NumClusters;                     % no. clusters
    n_subpath   = h_builder.NumSubPaths;
    n_paths     = sum( h_builder.NumSubPaths );              % no. paths
    
    % The per cluster angular spread scaling coefficients
    c_aod = h_builder.scenpar.PerClusterAS_D;    % in deg
    c_aoa = h_builder.scenpar.PerClusterAS_A;    % in deg
    c_eod = h_builder.scenpar.PerClusterES_D;    % in deg
    c_eoa = h_builder.scenpar.PerClusterES_A;    % in deg
    
    if size( h_builder.subpath_coupling, 3 ) == 1
        subpath_coupling = h_builder.subpath_coupling(:,:,1);
    else
        subpath_coupling = h_builder.subpath_coupling(:,:,i_mobile);
    end
end

% Reserve some memory for the output
aod = zeros( 1,n_paths );    
aoa = aod;
eod = aod;
eoa = aod;

ls = 1;
for l = 1 : n_clusters
    le = ls + n_subpath(l) - 1;
    
    % Get the offset angles
    if use_laplacian_pas        % Laplacian PAS - angle offset scaled by sqrt(2)
        if n_subpath(l) == 20
            of = offset .* 0.024682682989769;  % sqrt(2)*pi/180
        elseif n_subpath(l) == 1
            of = 0;
        elseif n_subpath(l) == 10
            of = [-0.02080 0.03181 0.01967 -0.02762 -0.01550 0.04141 0.00093 0.01170 -0.03724 -0.00436];
        elseif n_subpath(l) == 6
            of = [-0.02333 0.02456 0.01352 -0.02954 -0.01851 0.03330];
        elseif n_subpath(l) == 4
            of = [-0.02090 0.03009 0.01834 -0.02744];
        elseif n_subpath(l) < 20
            of = offset( 1:n_subpath(l) );
            of = (of-mean(of));
            of = of./sqrt(mean(of.^2));
            of = of .* 0.024682682989769;  % sqrt(2)*pi/180
        else
            of = rand(1,n_subpath(l));
            of = (of-mean(of));
            of = of./sqrt(mean(of.^2));
            of = of .* 0.024682682989769;  % sqrt(2)*pi/180
        end
         
    else    % Legacy PAS
        if n_subpath(l) == 20
            of = offset .* 0.017453292519943;  % in rad
        elseif n_subpath(l) == 1
            of = 0;
        elseif n_subpath(l) == 10
            of = [-0.01471 0.02249 0.01391 -0.01953 -0.01096 0.02928 0.00066 0.00827 -0.02633 -0.00308];
        elseif n_subpath(l) == 6
            of = [-0.01650 0.01737 0.00956 -0.02089 -0.01309 0.02355];
        elseif n_subpath(l) == 4
            of = [-0.01478 0.02128 0.01297 -0.0194 ];
        elseif n_subpath(l) < 20
            of = offset( 1:n_subpath(l) );
            of = (of-mean(of));
            of = of./sqrt(mean(of.^2));
            of = of .* 0.017453292519943;  % in rad
        else
            of = rand(1,n_subpath(l));
            of = (of-mean(of));
            of = of./sqrt(mean(of.^2));
            of = of .* 0.017453292519943;  % in rad
        end
    end
    
    % Change order to account for random sub-path coupling
    if use_qd_builder
        [~,ind1] = sort( subpath_coupling(1,ls:le) );
        [~,ind2] = sort( subpath_coupling(2,ls:le) );
        [~,ind3] = sort( subpath_coupling(3,ls:le) );
        [~,ind4] = sort( subpath_coupling(4,ls:le) );
        aod(ls:le) = c_aod * of( ind1 ) + h_builder.AoD( i_mobile,l );
        aoa(ls:le) = c_aoa * of( ind2 ) + h_builder.AoA( i_mobile,l );
        eod(ls:le) = c_eod * of( ind3 ) + h_builder.EoD( i_mobile,l );
        eoa(ls:le) = c_eoa * of( ind4 ) + h_builder.EoA( i_mobile,l );
    else
        aod(ls:le) = of;
        aoa(ls:le) = of;
        eod(ls:le) = of;
        eoa(ls:le) = of;
    end
    
    ls = le + 1;
end

% Calculate delays
if nargout == 5 && use_qd_builder
    dl = h_builder.taus(i_mobile,:);
    n_snapshots = h_builder.rx_track(1,i_mobile).no_snapshots;
    if h_builder.simpar(1,1).use_absolute_delays
        r_0 = h_builder.rx_positions(:,i_mobile) - h_builder.tx_position(:,i_mobile);
        D = sqrt(sum(r_0.^2)) / h_builder.simpar(1,1).speed_of_light;
        delay = repmat( dl+D , n_snapshots , 1  );
    else
        delay = repmat( dl , n_snapshots , 1  );
    end
else
    delay = [];
end

end
