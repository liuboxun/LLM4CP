function gain = diff_trans( h_mesh, orig, dest, center_frequency, ...
    no_path, no_seg, use_trans_model, obj_id, use_gpu, verbose )
%DIFF_TRANS Calculates the gain including diffraction and transmission losses
%
% Calling object:
%   Single object
%
% Description:
%   This function implements a diffraction and transmission loss model for the direct path from
%   transmitter to receiver. Free-Space path-loss is not included in the calculated gain. Upon
%   availability, this function uses GPU-acceleration to improve the computing performance.
%
% Input:
%   orig
%   Ray origins in 3D Cartesian coordinates; Dimensions: (3xN) or (3x1)
%
%   dest
%   Ray destinations in 3D Cartesian coordinates; Dimensions: (3xN) or (3x1)
%
%   center_frequency
%   Center frequency in [Hz]
%
%   no_path
%   Number of diffraction paths; Default: 37
%
%   no_seg
%   Number of segments per diffraction path; Default: 5
%
%   use_trans_model
%   Enable (1) or disable (0, default) the transmission loss model.
%
%   obj_id
%   A vector containing the object indices that should be simplified. Default: All
%
%   use_gpu
%   Enables (1) or disables (0) GPU acceleration. Default: auto-detect
%
%   verbose
%   Enables (1, default) or disables (0) progress report.
%
% Output:
%   gain
%   Gain caused by diffraction and transmission effects; linear scale; Dimensions: (1xN)
%
%
% QuaDRiGa Copyright (C) 2011-2021
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

if numel( h_mesh ) > 1
    error('QuaDRiGa:qd_mesh:diff_trans','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist( 'orig','var' ) || size(orig,1) ~= 3
    error('QuaDRiGa:qd_mesh:diff_trans','"orig" is not given or has wrong format');
end

if ~exist( 'dest','var' ) || size(dest,1) ~= 3
    error('QuaDRiGa:qd_mesh:diff_trans','"dest" is not given or has wrong format');
end

if ~exist( 'center_frequency','var' ) || size(center_frequency,1) ~= 1
    error('QuaDRiGa:qd_mesh:diff_trans','"center_frequency" is not given or has wrong format');
end

if ~exist('no_path','var') || isempty( no_path )
    no_path = 37;
end

if ~exist('no_seg','var') || isempty( no_seg )
    no_seg = 5;
end
if no_seg < 2
    no_path = 1;
    no_seg = 1;
end

if ~exist( 'use_trans_model','var' ) || isempty( use_trans_model )
    use_trans_model = false;
end

if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = true(size(h_mesh.obj_index));
    obj_id_in = [];
else
    obj_id_in = obj_id;
    obj_index = uint32( h_mesh.obj_index );
    ii = false( size( obj_index ) );
    for n = 1 : numel( obj_id )
        ii = ii | obj_index == uint32( obj_id(n) );
    end
    obj_id = ii;
end

if ~exist( 'use_gpu','var' ) || isempty( use_gpu )
    use_gpu = 1;
else
    use_gpu = use_gpu(1);
end
if use_gpu
    try
        use_gpu = use_gpu * double(qext.test_gpu_access > 0);
    catch
        use_gpu = 0;
    end
end

if ~exist( 'verbose','var' ) || isempty( verbose )
    verbose = 1;
end

% Determine number of rays
if size(orig,2) == 1 && size(dest,2) > 1
    no_ray = size(dest,2);
    orig = repmat( single(orig)',no_ray,1 );
    dest = single(dest)';
elseif size(orig,2) > 1 && size(dest,2) == 1
    no_ray = size(orig,2);
    dest = repmat( single(dest)',no_ray,1 );
    orig = single(orig)';
elseif size(orig,2) == size(dest,2)
    no_ray = size(orig,2);
    orig = single(orig)';
    dest = single(dest)';
else
    error('QuaDRiGa:qd_mesh:diff_trans','Size of "orig" and "dest" do not match.');
end

% Calculate the reducesd mesh size to be used for processing
mesh = single( h_mesh.mesh(obj_id,:) );         % Extract mesh
obj_index = h_mesh.obj_index(obj_id)';          % Extract object index
max_mesh_z = max(max(mesh(:,[3,6,9])))+0.1;     % Include ceiling

% Reduce mesh size using MEX function
if use_gpu
    [mesh,ind] = qext.mesh_reduce_size( mesh, [orig;0,0,max_mesh_z], [dest;0,0,max_mesh_z] );
    obj_index = obj_index(ind);
end

% Calculate the maximum number of interactions
if use_trans_model
    if verbose
        disp('Determine maximum number of intersections ...');
    end
    if use_gpu
        [ no_trans, ~, ~, ~, ~ ] = qext.ray_mesh_intersect( orig, dest, mesh, 1, verbose  );
    else
        [ ~, no_trans ] = intersect_mesh( h_mesh, orig', dest', obj_id_in, 1, 0, verbose );
    end
    no_hit_W = max(no_trans);
else
    no_hit_W = 1;
end

if verbose
    disp(['Approximating propagation ellipsoids with ',num2str(no_path*no_seg),' lines ...']);
end

% Calculate path length and orientation
[az,el,dist] = cart2sph(dest(:,1)-orig(:,1), dest(:,2)-orig(:,2), dest(:,3)-orig(:,3));

% Calculate the wavelength
s = qd_simulation_parameters;
s.center_frequency = center_frequency;
lambda = single(s.wavelength);

c = lambda/2 * dist;                   % Constant

% Approximate ellipse by line segments
xx = (0:100*no_seg)/(100*no_seg);
yy = sqrt(xx-xx.^2);                    % Ellipse
u = 0.5*(1-cos(pi*(0:no_seg)/no_seg));
v = sqrt(u-u.^2);                       % Start value (subsampled ellipse)
d = 0.1*ones(1,no_seg);
MSE = sum((xx-qf.interp(u,[],v,xx)).^2);
for m = 1:20
    for n = 2 : no_seg
        v(n) = v(n) + d(n);
        MSEn = sum((yy-qf.interp(u,[],v,xx)).^2);
        if MSEn > MSE
            v(n) = v(n) - d(n);
            d(n) = -d(n)/2;
        else
            MSE = MSEn;
        end
    end
end
v = c*v;

% Read Circle packling coordinates
quadriga_path = path;
if isempty( regexp( quadriga_path,';' ) ) %#ok  % Linux separates path entries by ":"
    quadriga_path = regexp(quadriga_path, ':?([^:]*quadriga_src)', 'tokens'); % Linux
else % Windows separated path entries by ";"
    quadriga_path = regexp(quadriga_path, ':?([^;]*quadriga_src)', 'tokens'); % Windows
end
data_folder = fullfile(quadriga_path{1}{1}, 'data');

fileID = fopen( fullfile(data_folder,'cci_coords.bin') ,'r');
fseek(fileID,2*sum(0:no_path-1),'bof');
crc = single( fread(fileID,[2 no_path],'int8')/128 );
fclose(fileID);

% Transform to polar coordinates
[theta,rho] = cart2pol( crc(1,:), crc(2,:) );

% Build elipsoids along the x-axis
E = zeros( no_ray, no_seg+1, no_path, 3, 'single' );
for n = 1 : no_path
    E(:,:,n,1)        = dist*u;
    E(:,2:no_seg,n,2) = cos(theta(n)) * rho(n) * v(:,2:no_seg);
    E(:,2:no_seg,n,3) = sin(theta(n)) * rho(n) * v(:,2:no_seg);
end

% Rotate ellipsoids to align with original directions
C = repmat( cos( el ), [1,no_seg+1,no_path] );
S = repmat( sin( el ), [1,no_seg+1,no_path] );
tmp = C.*E(:,:,:,1) - S.*E(:,:,:,3);
E(:,:,:,3) = S.*E(:,:,:,1) + C.*E(:,:,:,3) + repmat( orig(:,3), [1,no_seg+1,no_path] );
E(:,:,:,1) = tmp;
C = repmat( cos( az ), [1,no_seg+1,no_path] );
S = repmat( sin( az ), [1,no_seg+1,no_path] );
tmp = C.*E(:,:,:,1) - S.*E(:,:,:,2);
E(:,:,:,2) = S.*E(:,:,:,1) + C.*E(:,:,:,2) + repmat( orig(:,2), [1,no_seg+1,no_path] );
E(:,:,:,1) = tmp + repmat( orig(:,1), [1,no_seg+1,no_path] );

E = permute(E,[1,3,2,4]);
origE = E(:,:,1:end-1,:);
destE = E(:,:,2:end  ,:);

% Weight function (empirically determined to match Knife-Edge formula)
if n < 7
    S = [0.5 1];
    F = [1 1];
elseif n < 17 || n == 18
    S = [ 0.6 1.0 ];
    F = [ 1.5 0.4 ];
elseif n < 31 || n == 32 || n == 33
    S = [ 0.35 0.7 1.0 ];
    F = [ 2.5 0.3 0.4 ];
elseif n < 57
    S = [ 0.27 0.52 0.8 1.0  ];
    F = [ 3.2 0.8 0.1 0.4 ];
elseif n < 79
    S = [ 0.2 0.4 0.6 0.85 1.0  ];
    F = [ 4.1 1.6 0.4 0.0 0.6 ];
else
    S  = 0.1:0.1:1;
    F  = [ 7.6 4.4 1.5 0.3 0.2 0.5 -0.3 -0.1 0.4 0.8 ];
end
weight = F( sum(repmat(rho',[1,no_seg]) > repmat(S,[no_path,1]),2)+1 )/no_path;
weight = single( weight / sum(weight) );

% Calculate diffraction segments
if verbose
    disp('Calculating diffraction gain ...');
end

% Calculate intersections for all diffraction paths
if use_gpu
    [no_trans,~,~,~,W] = qext.ray_mesh_intersect( reshape(origE,[],3), reshape(destE,[],3), mesh, no_hit_W, verbose  );
else
    [ ~, no_trans, ~, ~, ~, W ] = intersect_mesh( h_mesh, reshape(origE,[],3)', reshape(destE,[],3)',...
        obj_id_in, no_hit_W, 0, verbose );
end
isnlos = reshape( no_trans~=0, [],no_path,no_seg );
isnlos = any( isnlos, 3 );
    
if use_trans_model
    if verbose
        disp('Calculating transmission gain ...');
    end
    
    % Adjust size of W to the actual number of transmissions
    no_hit_W = min( no_hit_W, max( no_trans ) );
    W = W(1:no_hit_W,:)';
    
    % Process each diffraction path separately
    origE   = reshape(origE,no_ray*no_path,no_seg,3);
    destE   = reshape(destE,no_ray*no_path,no_seg,3);
    W       = reshape(W,no_ray*no_path,no_seg,[]);
    isnlos  = isnlos(:);
    
    % Discard all paths that are not intersected by the mesh
    origE   = origE(isnlos,:,:);
    destE   = destE(isnlos,:,:) - origE;
    W       = W(isnlos,:,:);
    
    no_hit_W = no_hit_W+1;
    no_nlos = size( origE,1 );
    
    W(W == 0) = 1;                                  % Set maximum distance to 1
    W = sort(W,3);                                  % Sort in ascending order
    W = cat( 3, zeros(no_nlos,no_seg,'single'), W, ones(no_nlos,no_seg,'single') );   % Set first and last entry
    
    % Assemble Segment lengths and proble points
    S = {};
    E = {};
    I = false( no_nlos,no_seg,no_hit_W );               % Index list
    L = zeros( no_nlos,no_seg,no_hit_W,'single' );      % Segment length in [m]
    for n = 1 : no_seg
        for m = 1 : no_hit_W
            wx = W(:,n,m+1) - W(:,n,m);
            ii = wx ~= 0;
            if any( ii )
                I(:,n,m)  = ii;
                L(ii,n,m) = sqrt(sum(( repmat(wx(ii),[1,1,3]) .* destE(ii,n,:) ).^2,3));
                S{n,m} = reshape( origE(ii,n,:) + repmat(W(ii,n,m) + 0.5*wx(ii),[1,1,3]) .* destE(ii,n,:) ,[],3 );
                E{n,m} = S{n,m};
                E{n,m}(:,3) = max_mesh_z;
            else
                S{n,m} = single([]);
                E{n,m} = single([]);
            end
        end
    end
    
    % Test if the segments are inside an object by deermining the intersection point with the
    % "ceiling" of the object.
    if use_gpu
        [ ~,~,~,iFBS,~ ] = qext.ray_mesh_intersect( cat(1,S{:}), cat(1,E{:}), mesh, 1, verbose  );
    else
        [ ~,~,~,~,iFBS ] = intersect_mesh( h_mesh, cat(1,S{:})', cat(1,E{:})', obj_id_in, no_hit_W, 0, verbose );
    end
    
    % Obtain the object ID
    O = zeros( no_nlos,no_seg,no_hit_W,'uint32' );      % Object ID
    cnt = 0;
    for m = 1 : no_hit_W
        for n = 1 : no_seg
            if any( I(:,n,m) )
                tmp = cnt+sum(I(:,n,m));
                O(I(:,n,m),n,m) = iFBS(cnt+1:tmp);
                cnt = tmp;
            end
        end
    end
    O(O~=0) = obj_index( O(O~=0) );
    
    % Calculate transmission loss
    f_GHz = single( center_frequency/1e9 );
    gain = zeros(no_ray*no_path,1,'single');
    for n = 1 : numel( h_mesh.obj_name )
        A = single( h_mesh.obj_att_par(:,n) );
        d = sum(sum(L.*(O==uint32(n)),2),3);    % Total distance within the object
        if any( d>0 )
            L_max_dB = (A(1) + A(2)*d) .* f_GHz.^A(3);
            gamma    = min( 3.402823e+38, 10.^( A(4) * log10(f_GHz) + A(5) ) );
            gain(isnlos) = gain(isnlos) + L_max_dB.*(1-exp(-(d*gamma)./L_max_dB));                 % Transition loss in dB
        end
    end
    
    % Transform to linear value
    gain = 10.^(-gain*0.1);
    
    % Apply diffration gain
    gain = reshape( gain, no_ray, no_path );
    gain = sum( gain .* repmat(weight,[size(gain,1),1]), 2 )'.^2;
else
    gain = sum( double(~isnlos) .* weight, 2 )'.^2;
end


end

