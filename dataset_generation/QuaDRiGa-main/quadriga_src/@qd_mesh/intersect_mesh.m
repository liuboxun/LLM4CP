function [ islos, no_trans, fbs, lbs, iFBS, W ] = intersect_mesh( h_mesh, orig, dest, obj_id,...
    no_hit_W, use_gpu, verbose )
%INTERSECT_MESH Calculates ray-mesh intersections
%
% Calling object:
%   Single object
%
% Description:
%   A ray is defined by its origin 3D-position and its destination 3d-position. Along the line from
%   origin to destination, there may or may not be an object that blocks that line. This functions
%   calculates for a set of N rays if the line-of-sight (LOS) is blocked. If it is blocked, the
%   coordinates at which the interaction happened are calculates as well. Upon availability, this
%   function uses GPU-acceleration to improve the computing performance.
%
% Input:
%   orig
%   Ray origins in 3D Cartesian coordinates; Dimensions: (3xN) or (3x1)
%
%   dest
%   Ray destinations in 3D Cartesian coordinates; Dimensions: (3xN) or (3x1)
%
%   obj_id
%   A vector containing the object indices that should be simplified. Default: All
%
%   no_hit_W
%   Maximum number of interaction points to be returned in output 'W'
%
%   use_gpu
%   Enables (1) or disables (0) GPU acceleration. Default: auto-detect
%
%   verbose
%   Enables (1, default) or disables (0) progress report.
%
% Output:
%   islos
%   Logical vector indicating if an intersection happened; Dimensions: (1xN)
%
%   no_trans
%   Number of intersections for each ray; uint32 vector; Dimensions: (1xN)
%
%   fbs
%   First interaction point; single precision; Dimensions: (3xN)
%
%   lbs
%   Last interaction point; single precision; Dimensions: (3xN)
%
%   iFBS
%   Index of the first mesh element that was hit by the ray; uint32; Dimensions: (1xN)
%
%   W
%   Normalized intersection points; 0=orig, 1=dest; Dimensions: (no_hit_W x N)
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
    error('QuaDRiGa:qd_mesh:intersect_mesh','intersect not definded for object arrays.');
else
    h_mesh = h_mesh(1,1); % workaround for octave
end

if ~exist( 'orig','var' ) || size(orig,1) ~= 3
    error('QuaDRiGa:qd_mesh:intersect','"orig" is not given or has wrong format');
end
if ~exist( 'dest','var' ) || size(dest,1) ~= 3
    error('QuaDRiGa:qd_mesh:intersect','"dest" is not given or has wrong format');
end

if ~exist('obj_id','var') || isempty( obj_id )
    obj_id = true(size(h_mesh.obj_index));
else
    obj_index = uint32( h_mesh.obj_index );
    ii = false( size( obj_index ) );
    for n = 1 : numel( obj_id )
        ii = ii | obj_index == uint32( obj_id(n) );
    end
    obj_id = ii;
end

if ~exist( 'no_hit_W','var' ) || isempty( no_hit_W )
    no_hit_W = 1;
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
else
    verbose = verbose(1);
end

% Transform variables to single precision
if ~isa(orig,'single')
    orig = single( orig );
    orig_is_single = false;
else
    orig_is_single = true;
end
if ~isa(dest,'single')
    dest = single( dest );
end

% Determine number of rays
if size(orig,2) == 1 && size(dest,2) > 1
    no_ray = size(dest,2);
    orig = repmat( orig,1,no_ray );
elseif size(orig,2) > 1 && size(dest,2) == 1
    no_ray = size(orig,2);
    dest = repmat( dest,1,no_ray );
elseif size(orig,2) == size(dest,2)
    no_ray = size(orig,2);
else
    error('QuaDRiGa:qd_mesh:intersect','Size of "orig" and "dest" do not match.');
end

% Read the vertices from the mesh
mesh = single( h_mesh.mesh(obj_id,:) );

if use_gpu == 1
    orig = orig';
    dest = dest';
    [ edges, ~ ] = qext.mesh_reduce_size( mesh, orig, dest );
    [ no_trans, fbs, lbs, iFBS, W ] = qext.ray_mesh_intersect( orig, dest, edges, no_hit_W, verbose  );
    islos = no_trans < 0.5;
    fbs = fbs';
    lbs = lbs';
    
elseif use_gpu == 2
    if verbose
        disp('Computing on GPU (file interface) ...');
        verboseC = ' -v';
    else
        verboseC = '';
    end
    
    orig = orig';
    dest = dest';
    
    fileID = fopen('intersect_cuda_in.bin','w');
    fwrite(fileID,no_ray,'uint32');
    fwrite(fileID,3,'uint32');
    fwrite(fileID,orig,'single');
    fwrite(fileID,no_ray,'uint32');
    fwrite(fileID,3,'uint32');
    fwrite(fileID,dest,'single');
    fwrite(fileID,size(mesh,1),'uint32');
    fwrite(fileID,9,'uint32');
    fwrite(fileID,mesh,'single');
    fclose(fileID);
    
    quadriga_path = path;                               % Parse files in config-folder
    if isempty( regexp( quadriga_path,';' ) ) %#ok      % Linux separates path entries by ":"
        quadriga_path = regexp(quadriga_path, ':?([^:]*quadriga_src)', 'tokens'); % Linux
    else                                                % Windows separated path entries by ";"
        quadriga_path = regexp(quadriga_path, ':?([^;]*quadriga_src)', 'tokens'); % Windows
    end
    cuda_path = fullfile(quadriga_path{1}{1},filesep,'+qext',filesep,'bin');
    command = [ fullfile( cuda_path,'intersect_mesh' ),' -i ',fullfile( pwd,'intersect_cuda_in.bin' ),...
            ' -o ',fullfile( pwd,'intersect_cuda_out.bin' ),' -w ',num2str(no_hit_W,'%d'),verboseC ];
    status = system(command,'-echo');
        
    fileID   = fopen('intersect_cuda_out.bin','r');
    sz       = fread(fileID,[1 2],'*uint32');
    no_trans = fread(fileID,sz,'*uint32');
    sz       = fread(fileID,[1 2],'*uint32');
    fbs      = fread(fileID,sz,'*single')';
    sz       = fread(fileID,[1 2],'*uint32');
    lbs      = fread(fileID,sz,'*single')';
    sz       = fread(fileID,[1 2],'*uint32');
    iFBS     = fread(fileID,sz,'*uint32');
    sz       = fread(fileID,[1 2],'*uint32');
    W        = fread(fileID,sz,'*single');
    fclose(fileID);
    
    islos = no_trans < 0.5;
        
else
    if verbose
        fprintf('LOS det. CPU [');
        vb_dots = 50;
        tStart = clock;
        m0=0;
    end
    
    % Prepare output variables
    islos = false(no_ray,1);
    if nargout > 1
        no_trans = zeros(no_ray,1,'uint32');
    end
    if nargout > 2
        fbs = zeros( 3,no_ray,'single' );
    end
    if nargout > 3
        lbs = zeros( 3,no_ray,'single' );
    end
    if nargout > 4
        iFBS = zeros( no_ray,1,'uint32' );
    end
    if nargout > 5
        W = zeros( no_hit_W,no_ray,'single' );
    end
    
    cv1x = mesh(:,1);
    cv1y = mesh(:,2);
    cv1z = mesh(:,3);
    cv2x = mesh(:,4);
    cv2y = mesh(:,5);
    cv2z = mesh(:,6);
    cv3x = mesh(:,7);
    cv3y = mesh(:,8);
    cv3z = mesh(:,9);
    clear mesh
    
    % Discard the parts of the mesh that lie outside of the ray area
    cXmin = min( min( orig(1,:) ), min( dest(1,:) ) );
    cXmax = max( max( orig(1,:) ), max( dest(1,:) ) );
    cYmin = min( min( orig(2,:) ), min( dest(2,:) ) );
    cYmax = max( max( orig(2,:) ), max( dest(2,:) ) );
    cZmin = min( min( orig(3,:) ), min( dest(3,:) ) );
    cZmax = max( max( orig(3,:) ), max( dest(3,:) ) );
    
    % Select mesh parts
    iM= ( ( cv1x > cXmax ) & ( cv2x > cXmax ) & ( cv3x > cXmax ) ) |...
        ( ( cv1x < cXmin ) & ( cv2x < cXmin ) & ( cv3x < cXmin ) ) |...
        ( ( cv1y > cYmax ) & ( cv2y > cYmax ) & ( cv3y > cYmax ) ) |...
        ( ( cv1y < cYmin ) & ( cv2y < cYmin ) & ( cv3y < cYmin ) ) |...
        ( ( cv1z > cZmax ) & ( cv2z > cZmax ) & ( cv3z > cZmax ) ) |...
        ( ( cv1z < cZmin ) & ( cv2z < cZmin ) & ( cv3z < cZmin ) );
    iM = ~iM;
    
    % Discart unneded mesh parts
    cv1x = cv1x(iM);
    cv1y = cv1y(iM);
    cv1z = cv1z(iM);
    cv2x = cv2x(iM);
    cv2y = cv2y(iM);
    cv2z = cv2z(iM);
    cv3x = cv3x(iM);
    cv3y = cv3y(iM);
    cv3z = cv3z(iM);
    
    % Calculate edges from v1 to v2 and v1 to v3
    cv2x = cv2x - cv1x;
    cv2y = cv2y - cv1y;
    cv2z = cv2z - cv1z;
    cv3x = cv3x - cv1x;
    cv3y = cv3y - cv1y;
    cv3z = cv3z - cv1z;
    
    % Split Rays into x,y,z components
    % Calculate vecor from dest to orig (gD)
    cOx = orig(1,:);
    cOy = orig(2,:);
    cOz = orig(3,:);
    cDx = dest(1,:) - cOx;
    cDy = dest(2,:) - cOy;
    cDz = dest(3,:) - cOz;
    
    % Compute results ray-wise
    for i_ray = 1 : no_ray
        
        % Calculate vector pointing from "gO" to "gv1"
        gTx = cOx(i_ray) - cv1x;
        gTy = cOy(i_ray) - cv1y;
        gTz = cOz(i_ray) - cv1z;
        
        % Calculate the x component of the cross product of "gD" and "cv3"
        % First operation of calculating determinant of the matrix M = dot(cv2,pq)
        % Initialize 1st barycentric coordinate (gU)
        gPQ  = cv3z * cDy(i_ray) - cv3y * cDz(i_ray);
        gDet = single( 3.7433921e-23 ) + cv2x .* gPQ;
        gU   = gTx .* gPQ;
        
        % Calculate the y component of the cross product of "gD" and "cv3"
        % Second operation of calculating determinant of the matrix M = dot(cv2,pq)
        % Update 1st barycentric coordinate (gU)
        gPQ  = cv3x * cDz(i_ray) - cv3z * cDx(i_ray);
        gDet = gDet + cv2y .* gPQ;
        gU   = gU + gTy .* gPQ;
        
        % Calculate the z component of the cross product of "gD" and "cv3"
        % Third operation of calculating determinant of the matrix M = dot(cv2,pq)
        % Update 1st barycentric coordinate (gU)
        gPQ  = cv3y * cDx(i_ray) - cv3x * cDy(i_ray);
        gDet = gDet + cv2z .* gPQ;
        gU   = gU + gTz .* gPQ;
        
        % Calculate 1/gDet and finalize gU
        gDet = 1./gDet;
        gU = gU .* gDet;
        
        % Calculate cross-procuct between "gT" and "cv2"
        % Calculate and 2nd barycentric coordinate (gV)
        % Calculate and normalized line intersect position (gW)
        gPQ = cv2z .* gTy - cv2y .* gTz;
        gV  = gPQ .* cDx(i_ray);
        gW  = cv3x .* gPQ;
        
        gPQ = cv2x .* gTz - cv2z .* gTx;
        gV  = gV + gPQ .* cDy(i_ray);
        gW  = gW + cv3y .* gPQ;
        
        gPQ = cv2y .* gTx - cv2x .* gTy;
        gV  = gV + gPQ .* cDz(i_ray);
        gW  = gW + cv3z .* gPQ;
        
        gV  = gV .* gDet;
        gW  = gW .* gDet;
        
        % Check if the line is intersected by the Mesh
        hit = gU>=0 & gV>=0 & (gU+gV)<=1 & gW>0 & gW<1;
        
        % Write output data
        islos(i_ray) = ~any(hit,1);
        if nargout > 1
            no_trans(i_ray) = uint32( sum( hit,1 ) );
        end
        
        % Calculate FBS
        if nargout > 2
            gPQ = single( hit );
            [gX,ii] = min( gPQ .* gW - gPQ + 1, [], 1 );
            fbs(:,i_ray) = orig(:,i_ray) + gX * [cDx(i_ray);cDy(i_ray);cDz(i_ray)];
            if nargout > 4 && ~islos(i_ray)
               iFBS(i_ray) = ii;
            end
        end
        
        % Calculate LBS
        if nargout > 3
            gX  = max( gPQ .* gW );
            lbs(:,i_ray) = orig(:,i_ray) + gX * [cDx(i_ray);cDy(i_ray);cDz(i_ray)];
        end
        
        % Return all interset points
        if nargout > 5 && ~islos(i_ray)
           tmp = gW(hit);
           ii = min(no_trans(i_ray),no_hit_W);
           W(1:ii,i_ray) = tmp(1:ii);
        end
        
        % Update progress bar
        if verbose; m1=ceil(i_ray/no_ray*vb_dots); if m1>m0; for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
    end
    
    if verbose
        fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
    end
end

if nargout > 2 && ~orig_is_single
    fbs = double(fbs);
end
if nargout > 3 && ~orig_is_single
    lbs = double(lbs);
end

end
