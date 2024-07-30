function gen_parameters( h_builder, usage, vb_dots )
%GEN_PARAMETERS Generates LSF parameters, SSF parameters and scatterer positions
%
% Calling object:
%   Object array
%
% Description:
%   This function generates all parameters that are needed for the channel coefficient generation.
%   The outputs of the method are stored in the class properties. This includes the following
%   steps:
%
%   * Initialize the random generators by calling "qd_builder.init_sos". If the random generators
%     are already initialized, use the existing initialization.
%
%   * Generate correlated large scale parameters for each user position. Those parameters are
%     needed by the channel builder to calculate initial SSF parameters for each track or segment
%     which are then evolved into time varying channels.
%
%   * Generates the small-scale-fading parameters for the channel builder. Already existing
%     parameters are overwritten. However, due to the spatial consistency of the model, identical
%     values will be obtained for the same rx positions. Spatial consistency can be disabled by
%     setting 
%              "qd_builder.scenpar.SC_lambda = 0" or 
%              "qd_builder.simpar.autocorrelation_function = 'Disable'"
%
%   * Calculates the positions of the scatterers.
%
%
% Input:
%   usage
%   Controls the working mode of the method. The allowed options are:
%
%   * usage = 0
%     Clears all existing LSF and SSF parameters including the SOS random generators.
%
%   * usage = 1
%     Generates only the LSF parameters. Existing LSF parameters will be overwritten and all SSF
%     parameters will be cleared. If the SOS generators are not initialized, they are initialized
%     first. Existing SOS generators in "qd_builder.sos" are reused. This leads to identical results
%     when calling the method multiple times.
%
%   * usage = 2
%     Generates the SSF parameters. Existing SSF parameters will be overwritten. Existing SOS
%     generators and LSF parameters will be reused.
%
%   * usage = 3
%     Calculates the scattering cluster positions from the existing SSF parameters. In some cases,
%     this may lead to changes in the departure angles (AoD, EoD). If LSF or SSF parameters are not
%     initialized, they are initialized first.
%
%   * usage = 4 (default)
%     Clears existing LSF parameters, SSF parameters and cluster positions and calculates new ones.
%     Existing SOS generators are reused.
%
%   * usage = 5 
%     Keeps all existing parameters and creates new parameters only if they are missing.
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

if ~exist('usage','var') || isempty(usage)
    usage = 4;
end

verbose = h_builder(1,1).simpar(1,1).show_progress_bars;
if ( exist('vb_dots','var') && vb_dots == 0 ) || usage == 0
    verbose = 0;
end

if verbose && nargin < 3
    fprintf('Parameters   [');
    vb_dots = 50;
    tStart = clock;
end
m0=0;

if numel(h_builder) > 1
    
    % Equally distribute the dots in the progress bar
    sic = size( h_builder );
    vb_dots = zeros( 1,numel(h_builder) );
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        if h_builder(i1,i2,i3,i4).dual_mobility == -1
            check_dual_mobility( h_builder(i1,i2,i3,i4) );
        end
        if verbose
            vb_dots(i_cb) = h_builder(i1,i2,i3,i4).no_rx_positions;
        else
            % Workaround for Octave 4
            if numel( sic ) == 4
                h_builder(i1,i2,i3,i4).simpar(1,1).show_progress_bars = false;
            elseif numel( sic ) == 3
                h_builder(i1,i2,i3).simpar(1,1).show_progress_bars = false;
            else % 2 and 1
                h_builder(i1,i2).simpar(1,1).show_progress_bars = false;
            end
        end
    end
    if verbose
        vb_dots = init_progress_dots(vb_dots);
    end
    
    for i_cb = 1 : numel(h_builder)
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        if h_builder( i1,i2,i3,i4 ).no_rx_positions > 0
            gen_parameters( h_builder( i1,i2,i3,i4 ), usage, vb_dots(i_cb) );
        end
    end
    
else
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    if usage == 0
        
        % Clear all variables
        gen_fbs_lbs( h_builder, 0 );
        gen_ssf_parameters( h_builder, 0 );
        gen_lsf_parameters( h_builder, 0 );
        init_sos( h_builder, 0 );
        
    else
        
        % Initialize the random generators.
        % If they are already initialized, the existing values will not be changed.
        init_sos( h_builder,2 );
        
        % Delete existing values
        switch usage
            case 2
                gen_fbs_lbs( h_builder, 0 );
                gen_ssf_parameters( h_builder, 0 );
            case 3
                gen_fbs_lbs( h_builder, 0 );
            case {1,4}
                gen_fbs_lbs( h_builder, 0 );
                gen_ssf_parameters( h_builder, 0 );
                gen_lsf_parameters( h_builder, 0 );
        end
        
        % If there LSF parameters missing, generate them
        gen_lsf_parameters( h_builder, 2 );
        
        % Update progress bar (50% point)
        if verbose; m1=ceil(1/2*vb_dots); if m1>m0
                for m2=1:m1-m0; fprintf('o'); end; m0=m1; end
        end
                
        % If there SSF parameters missing, generate them
        if usage > 1.5
            gen_ssf_parameters( h_builder, 2 );
        end
        
        % If the scatterer positions are missing, create them
        if usage > 2.5
            gen_fbs_lbs( h_builder, 2 );
        end
        
        % Update progress bar( 100% point )
        if verbose; m1=ceil((2/2*vb_dots)); if m1>m0
                for m2=1:m1-m0; fprintf('o'); end; m0=m1; end
        end
    end
end

if verbose && nargin < 3
    fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
end

end
