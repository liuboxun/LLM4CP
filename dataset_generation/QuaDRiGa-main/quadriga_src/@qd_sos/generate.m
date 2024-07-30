function [ h_sos, ase, test_dir ] = generate( R, D, N, dim, uniform_smp, T, random_init, show_progress )
%GENERATE Generates SOS parameters from a given ACF
%
% This (static) method tries to approximate any given ACF by sinusoid coefficients. A sampled ACF is
% provided at the input. Then, the N sinusoid coefficients are iteratively determined until the best
% match is achieved. 
%
% Input:
%   R       The desired ACF (having values between 1 ind -1). The first value must be 1.
%   D       Vector of sample points for the ACF in [m]
%   N       Number of sinusoid coefficients used to approximate the ACF
%   dim     Number of dimensions (1, 2 or 3)
%   uniform_smp    If set to 1, sample points for 2 or more dimensions are spaced equally. If set to 0,
%                  sample pints are chosen randomly. Default: 1 (uniform)
%   T       Number of test-directions for N-dimensional approximation
%   random_init     If set to 1, the SOS frequencies are randomly initialized. Default: 0 (use
%                   initial search)
%   show_progress   If set to 1, an animation plot of the progress is shown. Default: 0
%
% Output:
%   h_sos   A qd_sos object.
%   ase     ASE for the test frequencies (vector containging the result for each iteration)
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

if ~exist( 'R','var' ) || isempty( R )
    error('You must specify an ACF.');
end

if ~exist( 'N','var' ) || isempty( N )
    error('You must specify the number of coefficients.');
end

% Read the number of sample points
S = single( numel(R) );
N = single( N );
test_dir = [];

if ~exist( 'D','var' ) || isempty( D )
    D = 0:S-1;
elseif numel(D) ~= numel( R )
    error('D and R must be of the same size');
end

if ~exist( 'dim','var' ) || isempty( dim )
    dim = 1;
end

if ~exist( 'uniform_smp','var' ) || isempty( uniform_smp )
    uniform_smp = false;
end

if ~exist( 'T','var' ) || isempty( T )
    T = 10;
end
if dim == 1
    T = 1;
end

if ~exist( 'random_init','var' ) || isempty( random_init )
    random_init = 1;
end

if ~exist( 'show_progress','var' ) || isempty( show_progress )
    show_progress = 0;
end

% Generate random angles for the multidiensional fit and test directions for the MSE evaluation
switch dim
    case 1
        theta = [];                     % Elevation
        phi = [];                       % Azimuth
        theta_t = [];                   % Elevation test direction
        phi_t = [];                     % Azimuth test direction
        
    case 2
        theta = [];                     % Elevation
        if uniform_smp                  % Azimuth
            u = 2 * pi / N;             % Circle circumference
            phi = 0 : u : 2*pi;
            phi = angle(exp(1j*phi(1:N))).';
        else
            % Random sampling of the unit circle
            phi = 2*(rand(N,1)-0.5)*pi;
        end
        u = 2 * pi / T;                 % Circle circumference
        phi_t = 0 : u : 2*pi;           % Azimuth test direction
        phi_t = angle(exp(1j*phi_t(1:T))).';
        theta_t = [];                   % Elevation test direction
        
        test_dir = phi_t;
        
    case 3
        if uniform_smp
            [ theta, phi ] = qf.pack_sphere( N );
            N = numel( phi );
        else
            % Random sampling of the unit sphere
            theta = acos(2*rand(N, 1) - 1) - pi/2;
            phi = 2*pi*(rand(N,1) - 0.5);
        end
        [ theta_t, phi_t ] = qf.pack_sphere( T );
        T = numel( phi_t );
        
        test_dir = [phi_t,theta_t];
end

% Transform input tensor into a matrix for easier processing
% Single precision provides sufficient numeric accuracy but reduces
% computing and memory requirements.
R = reshape( single(R) , S , 1 ).';
D = reshape( single(D) , S , 1 ).';

% Generate SOS object
h_sos = qd_sos([]);
h_sos.Pdist_decorr = single( D( find( R <= exp(-1) ,1 ) ) );
h_sos.dist = D;
h_sos.acf = R;

% Normalize entries in D. This saves some computing time.
dS = max(D);
Dnorm = 2*pi*D/dS;

% The frequencies along the principal axis of the coordoante system and the reconstruced ACF
fr = zeros( N, dim, 'single' );
Rr = ones( dim, S, N, 'single' );

% Reconstructed ACF at test frequencies
ft = zeros( N, T, 'single' );
Rt = ones( T, S, N, 'single' );           

oT = ones( 1,T,'uint8' );

% Predefine the weights with unit power and random phase
an = single( 1 / N );

% Single precision is faster
phi = single( phi );
theta = single( theta );
phi_t = single( phi_t ).';
theta_t = single( theta_t ).';

loop = 0;           % Iteration counter
cnt  = Inf;         % Counts the number of changed entries that improve the MSE
cst = Inf;          
ase = [];

% Randdom initialization
if random_init
    
    % Random values for the initial root frequencies
    fn = single( rand(N,1)-0.5 )*2*pi;

    % Generate directional and test frequencies
    if dim == 1 % 1D
        fr = fn;
        ft = fn;
        
    elseif dim == 2 % 2D
        fr = [fn .* cos( phi ), fn .* sin( phi ) ];
        ft = fr(:,1) * cos( phi_t ) + fr(:,2) * sin( phi_t );
        
    else % 3D
        fr = [ fn .* cos( phi ) .* cos( theta ),...
            fn .* sin( phi ) .* cos( theta ), fn .* sin( theta ) ];
        
        ft = fr(:,1) * ( cos( phi_t ) .* cos( theta_t ) ) +...
            fr(:,2) * ( sin( phi_t ) .* cos( theta_t ) ) +...
            fr(:,3) * sin( theta_t );
    end
    
    % Calculate the approximate ACF and the test ACF from the initial values
    for n = 1:N
        Rr(:,:,n) = cos( fr(n,:)' * Dnorm );
        Rt(:,:,n) = cos( ft(n,:)' * Dnorm );
    end
end

while cnt > 0
    cnt = 0;
    loop = loop + 1;

    for n = 1 : N
        
        % Updates are only applied if the ASE improves.
        frO = fr(n,:);              % Store the existing frequencies
        RrO = Rr(:,:,n);            % Store approximate ACF
        ftO = ft(n,:);              % Store the test frequencies
        RtO = Rt(:,:,n);            % Store test ACF
        
        % Select the SOS components that remain fixed
        if loop > 1 || random_init
            % Remove all detected components except the one that is refined
            ls = true(1,N);
            ls(n) = false;
        else
            ls = false(1,N);
            ls(1:n-1) = true;
        end
                
        % Select estimation dimension
        if dim == 1
            dE = 1;
        elseif dim == 2
            dx = cos( phi(n) );
            dy = sin( phi(n) );
            [~,dE] = max(abs([dx,dy]));
        elseif dim == 3
            dx = cos( phi(n) ) .* cos( theta(n) );
            dy = sin( phi(n) ) .* cos( theta(n) );
            dz = sin( theta(n) );
            [~,dE] = max(abs([dx,dy,dz]));
        end
        
        % Calculate difference between ACF and previous ACF
        if any( ls )
            RminusRr = R - sum( Rr(dE,:,ls)/sum(ls) , 3 );
        else
            RminusRr = R;
        end
        
        % Calculate the best matching SOS frequency
        f = suosi_iteration_step( RminusRr, Dnorm, an );

        % Calculate the directional components
        if dim == 1
            fr(n) = f;
        elseif dim == 2
            fr(n,:) = f .* [dx,dy];
        elseif dim == 3
            fr(n,:) = f .* [dx,dy,dz];
        end
        
        % Update approximate ACF
        Rr(:,:,n) = cos( fr(n,:)' * Dnorm );
        
        % Update test frequencies and the test ACF
        if dim == 1 % 1D
            ft = fr;
        elseif dim == 2 % 2D
            ft(n,:) = fr(n,1) .* cos( phi_t ) + fr(n,2) .* sin( phi_t );
        else % 3D
            ft(n,:) = fr(n,1) .* cos( phi_t ) .* cos( theta_t ) +...
                fr(n,2) .* sin( phi_t ) .* cos( theta_t ) + fr(n,3) .* sin( theta_t );
        end
        Rt(:,:,n) = cos( ft(n,:)' * Dnorm );
        
        % Calculate the ASE
        dbg  = sum( Rt,3 )/N;
        cstN = sum( ( R(oT,:) - dbg ).^2 ,2 ).' / S;
        cstN = sum( cstN ) / T;
  
        if cstN < cst
            % Update the list of frequencies
            cnt = cnt + 1;
            cst = cstN;
            
            if show_progress == 1
                figure;
                plot( R,'k','Linewidth',3 );
                hold on;
                py = plot( dbg.' );
                hold off;
                pt = title([loop,cnt,n]);
                ylim([-1,1])
                drawnow
                show_progress = 2;
                grid on
            elseif show_progress == 2
               set(pt, 'String', num2str([loop,cnt,n]') );
               for pn = 1 : T
                    set(py(pn), 'Ydata', dbg(pn,:) );
               end
               drawnow;
            end
        else
            % Use old variables if there is no improvement
            fr(n,:) = frO;
            ft(n,:) = ftO;
            Rr(:,:,n) = RrO;
            Rt(:,:,n) = RtO;
        end
    end
    random_init = 0;
    ase(loop) = cst;
end

% Scale the frequencies to match the distances
fr = fr(:,1:dim) / dS;

% Save output
h_sos.sos_freq = fr;
h_sos.sos_amp = sqrt( single( 2 /  N ));
h_sos.init;

end

