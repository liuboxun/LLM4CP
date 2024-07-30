classdef qd_simulation_parameters < handle
%QD_SIMULATION_PARAMETERS General configuration settings
%
% DESCRIPTION
% This class controls the simulation options and calculates constants for other classes.
%
% 
% QuaDRiGa Copyright (C) 2011-2023
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

properties(Dependent)
    
    % The number of samples per half-wave length
    % 	Sampling density describes the number of samples per half-wave length. To fulfill the
    % 	sampling theorem, the minimum sample density must be be 1 (single mobility) or 2 (dual
    % 	mobility). For  smaller values, interpolation of the channel for variable speed is not
    % 	possible. On the other hand, high values increase the computing time significantly. A good
    % 	value is around 1.2 for single-mobility and 2.5 for dual-mobility.    
    sample_density
    
    % Samples per one meter
    %   This parameter is linked to the sample density by
    %
    %       f_S = 2 * f_C * SD / c
    %
    %   where f_C is the carrier frequency in Hz, SD is the sample density and c is the speed of
    %   light.
    samples_per_meter
    
    center_frequency    % Center frequency in [Hz]
    
    % Enables or disables the 3GPP baseline model.
    % use_3GPP_baseline = 0 (default)
    %   Disables the 3GPP baseline model and uses enhanced QuaDRiGa features:
    %   * This option uses spherical waves at both ends, the transmitter and the receiver. This
    %     method uses a multi-bounce model where the departure and arrival angles are matched such
    %     that the angular spreads stay consistent.  
    %   * Uses the polarization rotation with an additional phase offset between the H and V
    %     component of the NLOS paths. The offset angle is calculated to match the XPR for circular
    %     polarization.  
    %
    % use_3GPP_baseline = 1
    %   Disables all QuaDRiGa features that are not specified by 3GPP.
    %   * Applies rotating phasors to each path which emulates time varying Doppler characteristics.
    %     Paths are not tracked and mobility is limited to maximum 10 m.  
    %   * The large-scale parameters (departure and arrival angles, shadow fading, delays, etc.) are
    %     not updated for terminal mobility. 
    %   * The phases at the array antennas are calculated by a planar wave approximation.
    %   * Spatial consistency is not available.
    %   * Multi-frequency simulations are not supported.
    %   * No polarization rotation is calculated. The polarization transfer matrix contains random
    %     phasors scaled to match the XPR. 
    use_3GPP_baseline
end

properties
    % Returns absolute delays in channel impulse response
    %   By default, delays are calculated such that the LOS delay is normalized to 0. By setting
    %   use_absolute_delays to 1 or true, the absolute path delays are included in
    %   qd_channel.delays at the output of the model.
    use_absolute_delays = false;
    
    % Initializes each path with a random initial phase
    %   By default, each path is initialized with a random phase (except the LOS path and the
    %   optional ground reflection). Setting "use_random_initial_phase" to zeros disables this
    %   function. In this case, each path gets initialized with a zero-phase.
    use_random_initial_phase = true;
    
    % Show a progress bar on the MATLAB prompt
    %   Show a progress bar on the MATLAB / Octave prompt. If this doesn't work correctly, you
    %   need to enable real-time output by calling "more off".
    show_progress_bars = true;
    
    % The autocorrelation function for generating correlated model parameters.
    %   An autocorrelation function (ACF) is a description of the correlation vs. distance.
    %   This function is approximated by a Fourier series. The coefficients of the series can be
    %   used to generate spatially correlated random variables in the qd_sos class. There are 3
    %   ACF types that can be selected by acf_type. The coefficients are precomputed for 150,
    %   300, 500, and 1000 sinusoids.
    %
    %   * Exponential ACF (Exp150, Exp300, Exp500, Exp1000 )
    %   * Gaussian ACF (Gauss150, Gauss300, Gauss500, Gauss1000 )
    %   * Combined Gaussian and Exponential ACF (Comb150, Comb300, Comb500, Comb1000)
    %   * Disable Spatial Consistency for Small-Scale Fading (Disable)
    autocorrelation_function = 'Comb300';
end

properties(Constant)
    % Version number of the current QuaDRiGa release (constant)
    version = '2.8.1-0';
end

properties(Constant)
    % Speed of light (constant)
    speed_of_light = 299792458;
end

properties(Dependent,SetAccess=protected)
    % Carrier wavelength in [m] (read only)
    wavelength
end

properties(Access=private)
    Psample_density = 2.5;
    Pcenter_frequency = 2.6e9;
    Puse_3GPP_baseline = false;
end

properties(Hidden)
    OctEq = false; % For qf.eq_octave
end

methods
    
    % Get functions
    function out = get.sample_density(obj)
        out = obj.Psample_density;
    end
    function out = get.samples_per_meter(obj)
        out = 2*max( obj.center_frequency )*obj.Psample_density ./ obj.speed_of_light;
    end
    function out = get.wavelength(obj)
        out = obj.speed_of_light ./ obj.center_frequency;
    end
    function out = get.center_frequency(obj)
        out = obj.Pcenter_frequency;
    end
    function out = get.use_3GPP_baseline(obj)
        out = obj.Puse_3GPP_baseline;
    end
    
    % Set functions
    function set.sample_density(obj,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) && isreal(value) && value > 0 )
            error('QuaDRiGa:qd_simulation_parameters:sample_density',...
                '??? Invalid sample density. The value must be real and > 0.');
        end
        obj.Psample_density = double( value );
    end
    
    function set.samples_per_meter(obj,value)
        if ~( all(size(value) == [1 1]) && isnumeric(value) && isreal(value) && value > 0 )
            error('QuaDRiGa:qd_simulation_parameters:samples_per_meter',...
                '??? Invalid samples_per_meter. The value must be real and > 0.');
        end
        obj.Psample_density = double( value )*min(obj.wavelength)/2;
    end
    
    function set.center_frequency(obj,value)
        if ~( isnumeric(value) && isreal(value) && all(value >= 0) )
            error('QuaDRiGa:qd_simulation_parameters:center_frequency',...
                '??? Invalid center frequency. The value must be real and > 0.');
        end
        if obj.Puse_3GPP_baseline && numel( value ) > 1
            warning('QuaDRiGa:qd_simulation_parameters:use_3GPP_baseline',...
                'Multi-frequency simulations are not compatible with the 3GPP baseline model. Path parameters will be uncorrellated.');
        end
        obj.Pcenter_frequency = reshape( double( value ) , 1 , [] );
    end
    
    function set.use_3GPP_baseline(obj,value)
        if ~( all(size(value) == [1 1]) && (isnumeric(value) || islogical(value)) && any( value == [0 1] ) )
           error('QuaDRiGa:qd_simulation_parameters:use_3GPP_baseline',...
               '??? "use_3GPP_baseline" must be 0 or 1')
        end
        if logical( value ) && numel( obj.Pcenter_frequency ) > 1
             warning('QuaDRiGa:qd_simulation_parameters:use_3GPP_baseline',...
                'Multi-frequency simulations are not compatible with the 3GPP baseline model. Path parameters will be uncorrellated.');
        end
        obj.Puse_3GPP_baseline = logical( value );
    end
    
    function set.use_absolute_delays(obj,value)
        if ~( all(size(value) == [1 1]) && (isnumeric(value) || islogical(value)) && any( value == [0 1] ) )
            error('QuaDRiGa:qd_simulation_parameters:use_absolute_delays',...
                '??? "use_absolute_delays" must be 0 or 1')
        end
        obj.use_absolute_delays = logical( value );
    end
    
    function set.use_random_initial_phase(obj,value)
        if ~( all(size(value) == [1 1]) && (isnumeric(value) || islogical(value)) && any( value == [0 1] ) )
           error('QuaDRiGa:qd_simulation_parameters:use_random_initial_phase',...
               '??? "use_random_initial_phase" must be 0 or 1')
        end
        obj.use_random_initial_phase = logical( value );
    end
    
    function set.show_progress_bars(obj,value)
        if ~( all(size(value) == [1 1]) && (isnumeric(value) || islogical(value)) && any( value == [0 1] ) )
            error('QuaDRiGa:qd_simulation_parameters:show_progress_bars',...
                '??? "show_progress_bars" must be 0 or 1')
        end
        obj.show_progress_bars = logical( value );
    end
end

methods(Static)
    gpu = has_gpu;
end

end
