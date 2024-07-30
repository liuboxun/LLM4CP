classdef qd_sos < handle
%QD_SOS QuaDRiGa sum-of-sinuoids autocorrelation model
%
% DESCRIPTION
% 
% This class implements the sum-of-sinusoids method for generating spatially correlated random
% variables. An autocorrelation function (ACF), i.e. a description of the correlation vs. distance,
% needs to be provided. This function is approximated by a fourier series. The coefficients of the
% series can the be used to generate spatially correlated random variables.
%
% Calling qd_sos without options creates a new SOS object with default settings. The default ACF
% is defined by a combination of the Gauss profile (ranging from 0 to the decorrelation distance)
% and an exponential decaying profile (ranging from the decorrelation distance to a maximum
% distance of 5 time the decorrelation distance. It is approximated by 300 sinusoids in 3D
% coordinates. There are 3 optional ACF types that can be selected by acf_type. The 3D SOS
% frequencies are precomputed for 150, 300, 500, and 1000 sinusoids.
%
%    * Exponential ACF (Exp150, Exp300, Exp500, Exp1000 )
%    * Gaussian ACF (Gauss150, Gauss300, Gauss500, Gauss1000 )
%    * Combined Gaussian and Exponential ACF (Comb150, Comb300, Comb500, Comb1000)
%
%    The distributer function can be either ('normal' or 'uniform').
%
% Input (constructor):
%   acf_type
%   String describing the shape of the autocorrelation function and the number of sinusoids,
%   Default: 'Comb300'
%
%   distribution
%   Distribution of the random variables ('normal' or 'uniform'), Default: 'normal'
%
%   dist_decorr
%   Decorrelation distance in [m], Default: 10 m
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

properties
    % Name of the preset
    name = 'Custom';
    
    % Distribution of random variables (Normal or Uniform)
    distribution = 'Normal';
end

properties(Dependent)
    % Decorrelation distance in [m]
    dist_decorr
    
    % The number of dimensions (1, 2 or 3)
    dimensions
    
    % Number of sampling frequencies
    no_coefficients
end

properties
    % Vector of sample points for the ACF in [m]
    dist = [];
    
    % Desired aotocorrelation function
    acf = [];
    
    % Sinusoid coefficients (frequencies)
    sos_freq = [];
    
    % Phase offsets
    sos_phase = [];
    
    % Amplitude of the sinusoid coefficients
    sos_amp = [];
end

properties(Access=private)
    Pdist_decorr = [];
end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
end

methods
    % Constructor
    function h_sos = qd_sos( acf_type, varargin )
        if exist('acf_type','var') && ~isempty( acf_type ) && nargin > 0
            h_sos = qd_sos.set_acf( acf_type , varargin{:} );
        elseif nargin == 0
            h_sos = qd_sos.set_acf( 'Comb300' );
        end
    end
    
    % Get functions
    function out = get.dist_decorr(h_sos)
        out = h_sos.Pdist_decorr;
    end
    function out = get.dimensions(h_sos)
        out = size( h_sos.sos_freq,2 );
    end
    function out = get.no_coefficients(h_sos)
        out = size( h_sos.sos_freq,1 );
    end
    
    % Set functions
    function set.dist_decorr(h_sos,val)
        if ~( ( all(size(val) == [1 1]) || all(size(val) == [1 2]) ) && isnumeric(val) && isreal(val) && all(val > 0) )
            error('QuaDRiGa:qd_sos:wrongInputValue','??? "decorr_dist" must be scalar and > 0')
        end
        if isempty( h_sos.dist )
            error('QuaDRiGa:qd_sos:noACFgiven','??? No ACF specified')
        end
        
        if numel( val ) == 2 && val(1) == val(2)
            val = val(1);
        end
        
        % Adjust distance vector
        maxD_old = h_sos.dist(end);
        h_sos.dist = val(1) ./ h_sos.Pdist_decorr(1) .* h_sos.dist;
        
        % Adjust the frequencies
        maxD_new = max(h_sos.dist);
        h_sos.sos_freq = h_sos.sos_freq * maxD_old / maxD_new;
        
        % Save new decorr distance
        h_sos.Pdist_decorr = val;
    end
    
    function set.distribution(h_sos,value)
        supported_types = {'Normal','Uniform','UniformForced'};
        if ~( ischar(value) && any( strcmpi(value,supported_types)) )
            str = 'Distribution not supported; supported types are: ';
            no = numel(supported_types);
            for n = 1:no
                str = [str,supported_types{n}];
                if n<no
                    str = [str,', '];
                end
            end
            error('QuaDRiGa:qd_sos:wrongDistributionType',str);
        end
        h_sos.distribution = value;
    end
end

methods(Static)
    h_sos = set_acf( acf_type, distribution, dist_decorr )
    [ h_sos, ase, test_dir ] = generate( R, D, N, dim, uniform_smp, T, random_init, show_progress )
    h_sos = load( filename )
    [ s, h_sos ] = randn( dist_decorr, ca, cb, acf_type )
    [ s, h_sos ] = rand( dist_decorr, ca, cb, acf_type )
    [ s, h_sos ] = randi( dist_decorr, ca, imax, cb, acf_type )
end
end
