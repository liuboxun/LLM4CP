function write_conf_file( h_builder, fn, write_defaults, ref, ref_sec, valid_range )
%WRITE_CONF_FILE Writes configuration files from qd_builder objects
%
% Calling object:
%   Single object
%
% Input:
%   fn
%   String containing the filename
%
%   write_defaults
%   If set to true, default values are written to the file.
%
%   ref
%   String containing the literature reference of the parameters
%
%   ref_sec
%   Reference for each section; Cell-Array containing 8 strings
%
%   valid_range
%   A 4x2 matrix containing the validity range of the parameters (to be written as text into the
%   conf file). The first row is for the frequency in GHz, the second row contains the distance in
%   m, the third row the allowed BS antenna height and the fourth row contains the allowed MT
%   height in m. The first column is for the lower range, the second for the upper range. Entering
%   NaN into a row prevents the writing of that row to the file.
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

if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "write_conf_file" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

if ~exist( 'fn','var' ) || isempty( fn )
    error('QuaDRiGa:qd_arrayant:xml_write',...
        'You did not specify a filename.');
end

if ~exist( 'write_defaults','var' ) || isempty( write_defaults )
    write_defaults = 0;
else
    write_defaults = Inf;
end

if ~exist( 'ref','var' ) || isempty( ref )
    ref = '';
end

if ~exist( 'ref_sec','var' ) || isempty( ref_sec )
    ref_sec = [];
elseif ~iscell(ref_sec) || numel( ref_sec ) ~= 8
    error('QuaDRiGa:qd_arrayant:xml_write',...
        '"ref_sec" must be a cell array with 8 variables.');
end

if ~exist( 'valid_range','var' ) || isempty( valid_range )
    valid_range = [];
elseif all( size( valid_range ) ~= [4,2] )
    error('QuaDRiGa:qd_arrayant:xml_write',...
        '"valid_range" must be a 4x2 array.');
end

% Open file
fid = fopen( fn, 'w');

fprintf(fid,['%% Config file for scenario "',h_builder.scenario,'"\n']);

if ~isempty( ref )
    fprintf(fid,['%% See: ',ref,'\n']);
end

if ~isempty(valid_range)
    fprintf(fid,'\n');
    if ~any(isnan(valid_range(1,:)))
        fprintf(fid,'%% Valid frequency range:        ');
        if valid_range(1,1) == valid_range(1,2)
            fprintf(fid,'%g',valid_range(1,1));
        else
            fprintf(fid,'%g',valid_range(1,1));
            fprintf(fid,' to ');
            fprintf(fid,'%g',valid_range(1,2));
        end
        fprintf(fid,' GHz\n');
    end
    if ~any(isnan(valid_range(2,:)))
        fprintf(fid,'%% Valid distance range:         ');
        if valid_range(2,1) == valid_range(2,2)
            fprintf(fid,'%g',valid_range(2,1));
        else
            fprintf(fid,'%g',valid_range(2,1));
            fprintf(fid,' to ');
            fprintf(fid,'%g',valid_range(2,2));
        end
        fprintf(fid,' m\n');
    end
    if ~any(isnan(valid_range(3,:)))
        fprintf(fid,'%% Valid BS antenna height:      ');
        if valid_range(3,1) == valid_range(3,2)
            fprintf(fid,'%g',valid_range(3,1));
        else
            fprintf(fid,'%g',valid_range(3,1));
            fprintf(fid,' to ');
            fprintf(fid,'%g',valid_range(3,2));
        end
        fprintf(fid,' m\n');
    end
    if ~any(isnan(valid_range(4,:)))
        fprintf(fid,'%% Valid MT antenna height:      ');
        if valid_range(4,1) == valid_range(4,2)
            fprintf(fid,'%g',valid_range(4,1));
        else
            fprintf(fid,'%g',valid_range(4,1));
            fprintf(fid,' to ');
            fprintf(fid,'%g',valid_range(4,2));
        end
        fprintf(fid,' m\n');
    end
end

if ( h_builder.scenpar.NumClusters == 1 || ...
        ( h_builder.scenpar.NumClusters == 2 && h_builder.scenpar.GR_enabled == 1 ) ) && write_defaults == 0
    % Do not write
else
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Large scale distributions\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{1} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{1},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n\n');
    
    % Overviews
    fprintf(fid,'%% DS  = ');
    write_comment_line( fid, h_builder.scenpar.DS_mu, h_builder.scenpar.DS_gamma, h_builder.scenpar.DS_omega, ...
        h_builder.scenpar.DS_epsilon, h_builder.scenpar.DS_zeta, h_builder.scenpar.DS_alpha );
    fprintf(fid,' + Xds * ( ');
    write_comment_line( fid, h_builder.scenpar.DS_sigma, h_builder.scenpar.DS_delta, h_builder.scenpar.DS_omega, ...
        h_builder.scenpar.DS_kappa, h_builder.scenpar.DS_tau, h_builder.scenpar.DS_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% KF  = ');
    write_comment_line( fid, h_builder.scenpar.KF_mu, h_builder.scenpar.KF_gamma, h_builder.scenpar.KF_omega, ...
        h_builder.scenpar.KF_epsilon, h_builder.scenpar.KF_zeta, h_builder.scenpar.KF_alpha );
    fprintf(fid,' + Xkf * ( ');
    write_comment_line( fid, h_builder.scenpar.KF_sigma, h_builder.scenpar.KF_delta, h_builder.scenpar.KF_omega, ...
        h_builder.scenpar.KF_kappa, h_builder.scenpar.KF_tau, h_builder.scenpar.KF_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% SF  = Xsf * ( ');
    write_comment_line( fid, h_builder.scenpar.SF_sigma, h_builder.scenpar.SF_delta, h_builder.scenpar.SF_omega, ...
        h_builder.scenpar.SF_kappa, h_builder.scenpar.SF_tau, h_builder.scenpar.SF_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% ASD = ');
    write_comment_line( fid, h_builder.scenpar.AS_D_mu, h_builder.scenpar.AS_D_gamma, h_builder.scenpar.AS_D_omega, ...
        h_builder.scenpar.AS_D_epsilon, h_builder.scenpar.AS_D_zeta, h_builder.scenpar.AS_D_alpha );
    fprintf(fid,' + Xasd * ( ');
    write_comment_line( fid, h_builder.scenpar.AS_D_sigma, h_builder.scenpar.AS_D_delta, h_builder.scenpar.AS_D_omega, ...
        h_builder.scenpar.AS_D_kappa, h_builder.scenpar.AS_D_tau, h_builder.scenpar.AS_D_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% ASA = ');
    write_comment_line( fid, h_builder.scenpar.AS_A_mu, h_builder.scenpar.AS_A_gamma, h_builder.scenpar.AS_A_omega, ...
        h_builder.scenpar.AS_A_epsilon, h_builder.scenpar.AS_A_zeta, h_builder.scenpar.AS_A_alpha );
    fprintf(fid,' + Xasa * ( ');
    write_comment_line( fid, h_builder.scenpar.AS_A_sigma, h_builder.scenpar.AS_A_delta, h_builder.scenpar.AS_A_omega, ...
        h_builder.scenpar.AS_A_kappa, h_builder.scenpar.AS_A_tau, h_builder.scenpar.AS_A_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% ESD = ');
    write_comment_line( fid, h_builder.scenpar.ES_D_mu, h_builder.scenpar.ES_D_gamma, h_builder.scenpar.ES_D_omega, ...
        h_builder.scenpar.ES_D_epsilon, h_builder.scenpar.ES_D_zeta, h_builder.scenpar.ES_D_alpha );
    fprintf(fid,' + Xesd * ( ');
    write_comment_line( fid, h_builder.scenpar.ES_D_sigma, h_builder.scenpar.ES_D_delta, h_builder.scenpar.ES_D_omega, ...
        h_builder.scenpar.ES_D_kappa, h_builder.scenpar.ES_D_tau, h_builder.scenpar.ES_D_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% ESA = ');
    write_comment_line( fid, h_builder.scenpar.ES_A_mu, h_builder.scenpar.ES_A_gamma, h_builder.scenpar.ES_A_omega, ...
        h_builder.scenpar.ES_A_epsilon, h_builder.scenpar.ES_A_zeta, h_builder.scenpar.ES_A_alpha );
    fprintf(fid,' + Xesa * ( ');
    write_comment_line( fid, h_builder.scenpar.ES_A_sigma, h_builder.scenpar.ES_A_delta, h_builder.scenpar.ES_A_omega, ...
        h_builder.scenpar.ES_A_kappa, h_builder.scenpar.ES_A_tau, h_builder.scenpar.ES_A_beta );
    fprintf(fid,' )\n');
    
    fprintf(fid,'%% XPR = ');
    write_comment_line( fid, h_builder.scenpar.XPR_mu, h_builder.scenpar.XPR_gamma, h_builder.scenpar.XPR_omega, ...
        h_builder.scenpar.XPR_epsilon, h_builder.scenpar.XPR_zeta, h_builder.scenpar.XPR_alpha );
    fprintf(fid,' + Xxpr * ( ');
    write_comment_line( fid, h_builder.scenpar.XPR_sigma, h_builder.scenpar.XPR_delta, h_builder.scenpar.XPR_omega, ...
        h_builder.scenpar.XPR_kappa, h_builder.scenpar.XPR_tau, h_builder.scenpar.XPR_beta );
    fprintf(fid,' )\n');
    
    % Delay Spread
    fprintf(fid,'\n');
    write_par_line( fid, 'DS_mu', 'delay spread [log10(s)]', h_builder.scenpar.DS_mu, Inf, ...
        h_builder.scenpar.DS_gamma, h_builder.scenpar.DS_omega, h_builder.scenpar.DS_epsilon, ...
        h_builder.scenpar.DS_zeta, h_builder.scenpar.DS_alpha );
    
    write_par_line( fid, 'DS_sigma', 'delay spread STD [log10(s)]', h_builder.scenpar.DS_sigma, Inf, ...
        h_builder.scenpar.DS_delta, h_builder.scenpar.DS_omega, h_builder.scenpar.DS_kappa, ...
        h_builder.scenpar.DS_tau, h_builder.scenpar.DS_beta );
    
    write_par_line( fid, 'DS_omega', 'reference frequency offset for the DS [GHz]', h_builder.scenpar.DS_omega, write_defaults );
    write_par_line( fid, 'DS_gamma', 'freq.-dep. of DS [log10(s)/log10(GHz)]', h_builder.scenpar.DS_gamma, write_defaults );
    write_par_line( fid, 'DS_epsilon', 'TX-RX 2D dist.-dep. of DS [log10(s)/log10(m)]', h_builder.scenpar.DS_epsilon, write_defaults );
    write_par_line( fid, 'DS_zeta', 'TX height-dep. of DS [log10(s)/log10(m)]', h_builder.scenpar.DS_zeta, write_defaults );
    write_par_line( fid, 'DS_alpha', 'elevation-dep. of DS [log10(s)/log10(rad)]', h_builder.scenpar.DS_alpha, write_defaults );
    write_par_line( fid, 'DS_delta', 'freq.-dep. of DS STD [log10(s)/log10(GHz)]', h_builder.scenpar.DS_delta, write_defaults );
    write_par_line( fid, 'DS_kappa', 'TX-RX 2D dist.-dep. of DS STD [log10(s)/log10(m)]', h_builder.scenpar.DS_kappa, write_defaults );
    write_par_line( fid, 'DS_tau', 'TX height-dep. of DS STD [log10(s)/log10(m)]', h_builder.scenpar.DS_tau, write_defaults );
    write_par_line( fid, 'DS_beta', 'elevation-dep. of DS STD [log10(s)/log10(rad)]', h_builder.scenpar.DS_beta, write_defaults );
    
    % K-Factor
    fprintf(fid,'\n');
    write_par_line( fid, 'KF_mu', 'Ricean K-factor [dB]', h_builder.scenpar.KF_mu, Inf, ...
        h_builder.scenpar.KF_gamma, h_builder.scenpar.KF_omega, h_builder.scenpar.KF_epsilon, ...
        h_builder.scenpar.KF_zeta, h_builder.scenpar.KF_alpha );
    
    write_par_line( fid, 'KF_sigma', 'Ricean K-factor STD [dB]', h_builder.scenpar.KF_sigma, Inf, ...
        h_builder.scenpar.KF_delta, h_builder.scenpar.KF_omega, h_builder.scenpar.KF_kappa, ...
        h_builder.scenpar.KF_tau, h_builder.scenpar.KF_beta );
    
    write_par_line( fid, 'KF_omega', 'reference frequency offset for the KF [GHz]', h_builder.scenpar.KF_omega, write_defaults );
    write_par_line( fid, 'KF_gamma', 'freq.-dep. of KF [dB/log10(GHz)]', h_builder.scenpar.KF_gamma, write_defaults );
    write_par_line( fid, 'KF_epsilon', 'TX-RX 2D dist.-dep. of KF [dB/log10(m)]', h_builder.scenpar.KF_epsilon, write_defaults );
    write_par_line( fid, 'KF_zeta', 'TX height-dep. of KF [dB/log10(m)]', h_builder.scenpar.KF_zeta, write_defaults );
    write_par_line( fid, 'KF_alpha', 'elevation-dep. of KF [dB/log10(rad)]', h_builder.scenpar.KF_alpha, write_defaults );
    write_par_line( fid, 'KF_delta', 'freq.-dep. of KF STD [dB/log10(GHz)]', h_builder.scenpar.KF_delta, write_defaults );
    write_par_line( fid, 'KF_kappa', 'TX-RX 2D dist.-dep. of KF STD [dB/log10(m)]', h_builder.scenpar.KF_kappa, write_defaults );
    write_par_line( fid, 'KF_tau', 'TX height-dep. of KF STD [dB/log10(m)]', h_builder.scenpar.KF_tau, write_defaults );
    write_par_line( fid, 'KF_beta', 'elevation-dep. of KF STD [dB/log10(rad)]', h_builder.scenpar.KF_beta, write_defaults );
    
    % SF
    fprintf(fid,'\n');
    write_par_line( fid, 'SF_sigma', 'Shadow Fading STD [dB]', h_builder.scenpar.SF_sigma, Inf, ...
        h_builder.scenpar.SF_delta, h_builder.scenpar.SF_omega, h_builder.scenpar.SF_kappa, ...
        h_builder.scenpar.SF_tau, h_builder.scenpar.SF_beta );
    
    write_par_line( fid, 'SF_omega', 'reference frequency offset for the SF [GHz]', h_builder.scenpar.SF_omega, write_defaults );
    write_par_line( fid, 'SF_delta', 'freq.-dep. of SF STD [dB/log10(GHz)]', h_builder.scenpar.SF_delta, write_defaults );
    write_par_line( fid, 'SF_kappa', 'TX-RX 2D dist.-dep. of SF STD [dB/log10(m)]', h_builder.scenpar.SF_kappa, write_defaults );
    write_par_line( fid, 'SF_tau', 'TX height-dep. of SF STD [dB/log10(m)]', h_builder.scenpar.SF_tau, write_defaults );
    write_par_line( fid, 'SF_beta', 'elevation-dep. of SF STD [dB/log10(rad)]', h_builder.scenpar.SF_beta, write_defaults );
    
    % ASD
    fprintf(fid,'\n');
    write_par_line( fid, 'AS_D_mu', 'azimuth of departure angle spread [log10(deg)]', h_builder.scenpar.AS_D_mu, Inf, ...
        h_builder.scenpar.AS_D_gamma, h_builder.scenpar.AS_D_omega, h_builder.scenpar.AS_D_epsilon, ...
        h_builder.scenpar.AS_D_zeta, h_builder.scenpar.AS_D_alpha );
    
    write_par_line( fid, 'AS_D_sigma', 'azimuth of departure angle spread STD [log10(deg)]', h_builder.scenpar.AS_D_sigma, Inf, ...
        h_builder.scenpar.AS_D_delta, h_builder.scenpar.AS_D_omega, h_builder.scenpar.AS_D_kappa, ...
        h_builder.scenpar.AS_D_tau, h_builder.scenpar.AS_D_beta );
    
    write_par_line( fid, 'AS_D_omega', 'reference frequency offset for the ASD [GHz]', h_builder.scenpar.AS_D_omega, write_defaults );
    write_par_line( fid, 'AS_D_gamma', 'freq.-dep. of ASD [log10(deg)/log10(GHz)]', h_builder.scenpar.AS_D_gamma, write_defaults );
    write_par_line( fid, 'AS_D_epsilon', 'TX-RX 2D dist.-dep. of ASD [log10(deg)/log10(m)]', h_builder.scenpar.AS_D_epsilon, write_defaults );
    write_par_line( fid, 'AS_D_zeta', 'TX height-dep. of ASD [log10(deg)/log10(m)]', h_builder.scenpar.AS_D_zeta, write_defaults );
    write_par_line( fid, 'AS_D_alpha', 'elevation-dep. of ASD [log10(deg)/log10(rad)]', h_builder.scenpar.AS_D_alpha, write_defaults );
    write_par_line( fid, 'AS_D_delta', 'freq.-dep. of ASD STD [log10(deg)/log10(GHz)]', h_builder.scenpar.AS_D_delta, write_defaults );
    write_par_line( fid, 'AS_D_kappa', 'TX-RX 2D dist.-dep. of ASD STD [log10(deg)/log10(m)]', h_builder.scenpar.AS_D_kappa, write_defaults );
    write_par_line( fid, 'AS_D_tau', 'TX height-dep. of ASD STD [log10(deg)/log10(m)]', h_builder.scenpar.AS_D_tau, write_defaults );
    write_par_line( fid, 'AS_D_beta', 'elevation-dep. of ASD STD [log10(deg)/log10(rad)]', h_builder.scenpar.AS_D_beta, write_defaults );
    
    % ASA
    fprintf(fid,'\n');
    write_par_line( fid, 'AS_A_mu', 'azimuth of arrival angle spread [log10(deg)]', h_builder.scenpar.AS_A_mu, Inf, ...
        h_builder.scenpar.AS_A_gamma, h_builder.scenpar.AS_A_omega, h_builder.scenpar.AS_A_epsilon, ...
        h_builder.scenpar.AS_A_zeta, h_builder.scenpar.AS_A_alpha );
    
    write_par_line( fid, 'AS_A_sigma', 'azimuth of arrival angle spread STD [log10(deg)]', h_builder.scenpar.AS_A_sigma, Inf, ...
        h_builder.scenpar.AS_A_delta, h_builder.scenpar.AS_A_omega, h_builder.scenpar.AS_A_kappa, ...
        h_builder.scenpar.AS_A_tau, h_builder.scenpar.AS_A_beta );
    
    write_par_line( fid, 'AS_A_omega', 'reference frequency offset for the ASA [GHz]', h_builder.scenpar.AS_A_omega, write_defaults );
    write_par_line( fid, 'AS_A_gamma', 'freq.-dep. of ASA [log10(deg)/log10(GHz)]', h_builder.scenpar.AS_A_gamma, write_defaults );
    write_par_line( fid, 'AS_A_epsilon', 'TX-RX 2D dist.-dep. of ASA [log10(deg)/log10(m)]', h_builder.scenpar.AS_A_epsilon, write_defaults );
    write_par_line( fid, 'AS_A_zeta', 'TX height-dep. of ASA [log10(deg)/log10(m)]', h_builder.scenpar.AS_A_zeta, write_defaults );
    write_par_line( fid, 'AS_A_alpha', 'elevation-dep. of ASA [log10(deg)/log10(rad)]', h_builder.scenpar.AS_A_alpha, write_defaults );
    write_par_line( fid, 'AS_A_delta', 'freq.-dep. of ASA STD [log10(deg)/log10(GHz)]', h_builder.scenpar.AS_A_delta, write_defaults );
    write_par_line( fid, 'AS_A_kappa', 'TX-RX 2D dist.-dep. of ASA STD [log10(deg)/log10(m)]', h_builder.scenpar.AS_A_kappa, write_defaults );
    write_par_line( fid, 'AS_A_tau', 'TX height-dep. of ASA STD [log10(deg)/log10(m)]', h_builder.scenpar.AS_A_tau, write_defaults );
    write_par_line( fid, 'AS_A_beta', 'elevation-dep. of ASA STD [log10(deg)/log10(rad)]', h_builder.scenpar.AS_A_beta, write_defaults );
    
    % ESD
    fprintf(fid,'\n');
    write_par_line( fid, 'ES_D_mu', 'elevation of departure angle spread [log10(deg)]', h_builder.scenpar.ES_D_mu, Inf, ...
        h_builder.scenpar.ES_D_gamma, h_builder.scenpar.ES_D_omega, h_builder.scenpar.ES_D_epsilon, ...
        h_builder.scenpar.ES_D_zeta, h_builder.scenpar.ES_D_alpha );
    
    write_par_line( fid, 'ES_D_sigma', 'elevation of departure angle spread STD [log10(deg)]', h_builder.scenpar.ES_D_sigma, Inf, ...
        h_builder.scenpar.ES_D_delta, h_builder.scenpar.ES_D_omega, h_builder.scenpar.ES_D_kappa, ...
        h_builder.scenpar.ES_D_tau, h_builder.scenpar.ES_D_beta );
    
    write_par_line( fid, 'ES_D_omega', 'reference frequency offset for the ESD [GHz]', h_builder.scenpar.ES_D_omega, write_defaults );
    write_par_line( fid, 'ES_D_gamma', 'freq.-dep. of ESD [log10(deg)/log10(GHz)]', h_builder.scenpar.ES_D_gamma, write_defaults );
    write_par_line( fid, 'ES_D_epsilon', 'TX-RX 2D dist.-dep. of ESD [log10(deg)/log10(m)]', h_builder.scenpar.ES_D_epsilon, write_defaults );
    write_par_line( fid, 'ES_D_zeta', 'TX height-dep. of ESD [log10(deg)/log10(m)]', h_builder.scenpar.ES_D_zeta, write_defaults );
    write_par_line( fid, 'ES_D_alpha', 'elevation-dep. of ESD [log10(deg)/log10(rad)]', h_builder.scenpar.ES_D_alpha, write_defaults );
    write_par_line( fid, 'ES_D_delta', 'freq.-dep. of ESD STD [log10(deg)/log10(GHz)]', h_builder.scenpar.ES_D_delta, write_defaults );
    write_par_line( fid, 'ES_D_kappa', 'TX-RX 2D dist.-dep. of ESD STD [log10(deg)/log10(m)]', h_builder.scenpar.ES_D_kappa, write_defaults );
    write_par_line( fid, 'ES_D_tau', 'TX height-dep. of ESD STD [log10(deg)/log10(m)]', h_builder.scenpar.ES_D_tau, write_defaults );
    write_par_line( fid, 'ES_D_beta', 'elevation-dep. of ESD STD [log10(deg)/log10(rad)]', h_builder.scenpar.ES_D_beta, write_defaults );
    write_par_line( fid, 'ES_D_mu_min', 'minimum ESD reference value [log10(deg)]', h_builder.scenpar.ES_D_mu_min, -Inf );
    write_par_line( fid, 'ES_D_mu_A', 'TX-RX 2D dist.-dep. of ESD [log10(deg)/km]', h_builder.scenpar.ES_D_mu_A, write_defaults );
    
    % ESA
    fprintf(fid,'\n');
    write_par_line( fid, 'ES_A_mu', 'elevation of arrival angle spread [log10(deg)]', h_builder.scenpar.ES_A_mu, Inf, ...
        h_builder.scenpar.ES_A_gamma, h_builder.scenpar.ES_A_omega, h_builder.scenpar.ES_A_epsilon, ...
        h_builder.scenpar.ES_A_zeta, h_builder.scenpar.ES_A_alpha );
    
    write_par_line( fid, 'ES_A_sigma', 'elevation of arrival angle spread STD [log10(deg)]', h_builder.scenpar.ES_A_sigma, Inf, ...
        h_builder.scenpar.ES_A_delta, h_builder.scenpar.ES_A_omega, h_builder.scenpar.ES_A_kappa, ...
        h_builder.scenpar.ES_A_tau, h_builder.scenpar.ES_A_beta );
    
    write_par_line( fid, 'ES_A_omega', 'reference frequency offset for the ESA [GHz]', h_builder.scenpar.ES_A_omega, write_defaults );
    write_par_line( fid, 'ES_A_gamma', 'freq.-dep. of ESA [log10(deg)/log10(GHz)]', h_builder.scenpar.ES_A_gamma, write_defaults );
    write_par_line( fid, 'ES_A_epsilon', 'TX-RX 2D dist.-dep. of ESA [log10(deg)/log10(m)]', h_builder.scenpar.ES_A_epsilon, write_defaults );
    write_par_line( fid, 'ES_A_zeta', 'TX height-dep. of ESA [log10(deg)/log10(m)]', h_builder.scenpar.ES_A_zeta, write_defaults );
    write_par_line( fid, 'ES_A_alpha', 'elevation-dep. of ESA [log10(deg)/log10(rad)]', h_builder.scenpar.ES_A_alpha, write_defaults );
    write_par_line( fid, 'ES_A_delta', 'freq.-dep. of ESA STD [log10(deg)/log10(GHz)]', h_builder.scenpar.ES_A_delta, write_defaults );
    write_par_line( fid, 'ES_A_kappa', 'TX-RX 2D dist.-dep. of ESA STD [log10(deg)/log10(m)]', h_builder.scenpar.ES_A_kappa, write_defaults );
    write_par_line( fid, 'ES_A_tau', 'TX height-dep. of ESA STD [log10(deg)/log10(m)]', h_builder.scenpar.ES_A_tau, write_defaults );
    write_par_line( fid, 'ES_A_beta', 'elevation-dep. of ESA STD [log10(deg)/log10(rad)]', h_builder.scenpar.ES_A_beta, write_defaults );
    
    % XPR
    fprintf(fid,'\n');
    write_par_line( fid, 'XPR_mu', 'cross-polarization ratio [dB]', h_builder.scenpar.XPR_mu, Inf, ...
        h_builder.scenpar.XPR_gamma, h_builder.scenpar.XPR_omega, h_builder.scenpar.XPR_epsilon, ...
        h_builder.scenpar.XPR_zeta, h_builder.scenpar.XPR_alpha );
    
    write_par_line( fid, 'XPR_sigma', 'cross-polarization ratio STD [dB]', h_builder.scenpar.XPR_sigma, Inf, ...
        h_builder.scenpar.XPR_delta, h_builder.scenpar.XPR_omega, h_builder.scenpar.XPR_kappa, ...
        h_builder.scenpar.XPR_tau, h_builder.scenpar.XPR_beta );
    
    write_par_line( fid, 'XPR_omega', 'reference frequency offset for the XPR [GHz]', h_builder.scenpar.XPR_omega, write_defaults );
    write_par_line( fid, 'XPR_gamma', 'freq.-dep. of XPR [dB/log10(GHz)]', h_builder.scenpar.XPR_gamma, write_defaults );
    write_par_line( fid, 'XPR_epsilon', 'TX-RX 2D dist.-dep. of XPR [dB/log10(m)]', h_builder.scenpar.XPR_epsilon, write_defaults );
    write_par_line( fid, 'XPR_zeta', 'TX height-dep. of XPR [dB/log10(m)]', h_builder.scenpar.XPR_zeta, write_defaults );
    write_par_line( fid, 'XPR_alpha', 'elevation-dep. of XPR [dB/log10(rad)]', h_builder.scenpar.XPR_alpha, write_defaults );
    write_par_line( fid, 'XPR_delta', 'freq.-dep. of XPR STD [dB/log10(GHz)]', h_builder.scenpar.XPR_delta, write_defaults );
    write_par_line( fid, 'XPR_kappa', 'TX-RX 2D dist.-dep. of XPR STD [dB/log10(m)]', h_builder.scenpar.XPR_kappa, write_defaults );
    write_par_line( fid, 'XPR_tau', 'TX height-dep. of XPR STD [dB/log10(m)]', h_builder.scenpar.XPR_tau, write_defaults );
    write_par_line( fid, 'XPR_beta', 'elevation-dep. of XPR STD [dB/log10(rad)]', h_builder.scenpar.XPR_beta, write_defaults );
    
end

if ( h_builder.scenpar.NumClusters == 1 || ...
        ( h_builder.scenpar.NumClusters == 2 && h_builder.scenpar.GR_enabled == 1 ) ) && write_defaults == 0
    % Don't write
else
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Large-Scale fading decorrelation distances\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{2} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{2},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n\n');
    
    write_par_line( fid, 'DS_lambda', 'DS decorrelation distance [m]', h_builder.scenpar.DS_lambda, Inf );
    write_par_line( fid, 'KF_lambda', 'KF decorrelation distance [m]', h_builder.scenpar.KF_lambda, Inf );
    write_par_line( fid, 'SF_lambda', 'SF decorrelation distance [m]', h_builder.scenpar.SF_lambda, Inf );
    write_par_line( fid, 'AS_D_lambda', 'ASD decorrelation distance [m]', h_builder.scenpar.AS_D_lambda, Inf );
    write_par_line( fid, 'AS_A_lambda', 'ASA decorrelation distance [m]', h_builder.scenpar.AS_A_lambda, Inf );
    write_par_line( fid, 'ES_D_lambda', 'ESD decorrelation distance [m]', h_builder.scenpar.ES_D_lambda, Inf );
    write_par_line( fid, 'ES_A_lambda', 'ESA decorrelation distance [m]', h_builder.scenpar.ES_A_lambda, Inf );
    write_par_line( fid, 'XPR_lambda', 'XPR decorrelation distance [m]', h_builder.scenpar.XPR_lambda, Inf );
end

if any( abs( reshape( h_builder.lsp_xcorr - eye(8),[],1 ) ) > 1e-3 ) || write_defaults ~= 0
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Inter-parameter correlations\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{3} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{3},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n\n');
    
    fprintf(fid,'%%         DS     KF     SF     ASD    ASA    ESD    ESA    XPR\n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(1,:) ); fprintf(fid,'| DS \n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(2,:) ); fprintf(fid,'| KF \n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(3,:) ); fprintf(fid,'| SF \n');
    fprintf(fid,'%% R = |  '); write_data_line(fid, h_builder.lsp_xcorr(4,:) ); fprintf(fid,'| ASD \n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(5,:) ); fprintf(fid,'| ASA \n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(6,:) ); fprintf(fid,'| ESD \n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(7,:) ); fprintf(fid,'| ESA \n');
    fprintf(fid,'%%     |  '); write_data_line(fid, h_builder.lsp_xcorr(8,:) ); fprintf(fid,'| XPR \n\n');
    
    write_par_line( fid, 'ds_kf', 'DS vs. KF', h_builder.scenpar.ds_kf, write_defaults );
    write_par_line( fid, 'ds_sf', 'DS vs. SF', h_builder.scenpar.ds_sf, write_defaults );
    write_par_line( fid, 'asD_ds', 'DS vs. ASD', h_builder.scenpar.asD_ds, write_defaults );
    write_par_line( fid, 'asA_ds', 'DS vs. ASA', h_builder.scenpar.asA_ds, write_defaults );
    write_par_line( fid, 'esD_ds', 'DS vs. ESD', h_builder.scenpar.esD_ds, write_defaults );
    write_par_line( fid, 'esA_ds', 'DS vs. ESA', h_builder.scenpar.esA_ds, write_defaults );
    write_par_line( fid, 'xpr_ds', 'DS vs. XPR', h_builder.scenpar.xpr_ds, write_defaults );
    write_par_line( fid, 'sf_kf', 'KF vs. SF', h_builder.scenpar.sf_kf, write_defaults );
    write_par_line( fid, 'asD_kf', 'KF vs. ASD', h_builder.scenpar.asD_kf, write_defaults );
    write_par_line( fid, 'asA_kf', 'KF vs. ASA', h_builder.scenpar.asA_kf, write_defaults );
    write_par_line( fid, 'esD_kf', 'KF vs. ESD', h_builder.scenpar.esD_kf, write_defaults );
    write_par_line( fid, 'esA_kf', 'KF vs. ESA', h_builder.scenpar.esA_kf, write_defaults );
    write_par_line( fid, 'xpr_kf', 'KF vs. XPR', h_builder.scenpar.xpr_kf, write_defaults );
    write_par_line( fid, 'asD_sf', 'SF vs. ASD', h_builder.scenpar.asD_sf, write_defaults );
    write_par_line( fid, 'asA_sf', 'SF vs. ASA', h_builder.scenpar.asA_sf, write_defaults );
    write_par_line( fid, 'esD_sf', 'SF vs. ESD', h_builder.scenpar.esD_sf, write_defaults );
    write_par_line( fid, 'esA_sf', 'SF vs. ESA', h_builder.scenpar.esA_sf, write_defaults );
    write_par_line( fid, 'xpr_sf', 'SF vs. XPR', h_builder.scenpar.xpr_sf, write_defaults );
    write_par_line( fid, 'asD_asA', 'ASD vs. ASA', h_builder.scenpar.asD_asA, write_defaults );
    write_par_line( fid, 'esD_asD', 'ASD vs. ESD', h_builder.scenpar.esD_asD, write_defaults );
    write_par_line( fid, 'esA_asD', 'ASD vs. ESA', h_builder.scenpar.esA_asD, write_defaults );
    write_par_line( fid, 'xpr_asd', 'ASD vs. XPR', h_builder.scenpar.xpr_asd, write_defaults );
    write_par_line( fid, 'esD_asA', 'ASA vs. ESD', h_builder.scenpar.esD_asA, write_defaults );
    write_par_line( fid, 'esA_asA', 'ASA vs. ESA', h_builder.scenpar.esA_asA, write_defaults );
    write_par_line( fid, 'xpr_asa', 'ASA vs. XPR', h_builder.scenpar.xpr_asa, write_defaults );
    write_par_line( fid, 'esD_esA', 'ESD vs. ESA', h_builder.scenpar.esD_esA, write_defaults );
    write_par_line( fid, 'xpr_esd', 'ESD vs. XPR', h_builder.scenpar.xpr_esd, write_defaults );
    write_par_line( fid, 'xpr_esa', 'ESA vs. XPR', h_builder.scenpar.xpr_esa, write_defaults );
end

fprintf(fid,'\n%% ==================================================================================================\n');
fprintf(fid,'%% Model parameters\n');
if ~isempty( ref_sec ) && ~isempty( ref_sec{4} )
    fprintf(fid,['%% See: ',ref,'; ',ref_sec{4},' \n']);
end
fprintf(fid,'%% ==================================================================================================\n\n');

write_par_line( fid, 'NumClusters', 'number of clusters', h_builder.scenpar.NumClusters, write_defaults );

if ( h_builder.scenpar.NumClusters == 1 || ...
        ( h_builder.scenpar.NumClusters == 2 && h_builder.scenpar.GR_enabled == 1 ) ) && write_defaults == 0
    % Don't write
else
    write_par_line( fid, 'NumSubPaths', 'number of paths per (NLOS) cluster', h_builder.scenpar.NumSubPaths, write_defaults );
    write_par_line( fid, 'SubpathMethod', 'subpath mapping method (legacy, Laplacian or mmMAGIC)', h_builder.scenpar.SubpathMethod, write_defaults );
    write_par_line( fid, 'LOS_scatter_radius', 'not used', h_builder.scenpar.LOS_scatter_radius, write_defaults );
    
    fprintf(fid,'\n');
    
    write_par_line( fid, 'r_DS', 'delay scaling factor', h_builder.scenpar.r_DS, Inf );
    write_par_line( fid, 'LNS_ksi', 'per cluster shadowing STD [dB]', h_builder.scenpar.LNS_ksi, write_defaults );
    
    fprintf(fid,'\n');
    
    write_par_line( fid, 'PerClusterDS', 'cluster delay spread [ns]', h_builder.scenpar.PerClusterDS, write_defaults );
    write_par_line( fid, 'PerClusterDS_gamma', 'freq.-dep. of cluster delay spread [ns/log10(GHz)]', h_builder.scenpar.PerClusterDS_gamma, write_defaults );
    write_par_line( fid, 'PerClusterDS_min', 'minimum cluster delay spread [ns]', h_builder.scenpar.PerClusterDS_min, write_defaults );
    
    write_par_line( fid, 'PerClusterAS_D', 'cluster azimuth of departure angle spread [deg]', h_builder.scenpar.PerClusterAS_D, write_defaults );
    write_par_line( fid, 'PerClusterAS_A', 'cluster azimuth of arrival angle spread [deg]', h_builder.scenpar.PerClusterAS_A, write_defaults );
    write_par_line( fid, 'PerClusterES_D', 'cluster elevation of departure angle spread [deg]', h_builder.scenpar.PerClusterES_D, write_defaults );
    write_par_line( fid, 'PerClusterES_A', 'cluster elevation of arrival angle spread [deg]', h_builder.scenpar.PerClusterES_A, write_defaults );
end

if h_builder.scenpar.SC_lambda > 0  || write_defaults ~= 0
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Decorrelation distance for the small-scale fading spatial consistency\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{5} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{5},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n\n');
    write_par_line( fid, 'SC_lambda', 'decorrelation distance [m]; 0 = disabled', h_builder.scenpar.SC_lambda, Inf );
end

if h_builder.scenpar.GR_enabled ~= 0 || write_defaults ~= 0
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Parameters for the ground reflection extension\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{6} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{6},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n\n');
    
    write_par_line( fid, 'GR_enabled', 'Enables the explicit ground reflection model', h_builder.scenpar.GR_enabled, write_defaults );
    write_par_line( fid, 'GR_epsilon', 'Fixed relative permittivity of the ground (0 = automatic)', h_builder.scenpar.GR_epsilon, write_defaults );
end

if h_builder.scenpar.absTOA_mu > -99 || write_defaults ~= 0
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Absolute time of arrival model parameters (optional feature)\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{7} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{7},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n\n');
    
    write_par_line( fid, 'absTOA_mu', 'absolute time of arrival offset reference value [log10(s)]', h_builder.scenpar.absTOA_mu, write_defaults );
    write_par_line( fid, 'absTOA_sigma', 'absolute time of arrival offset reference STD [log10(s)]', h_builder.scenpar.absTOA_sigma, write_defaults );
    write_par_line( fid, 'absTOA_lambda', 'absolute time of arrival offset decorrelation distance [m]', h_builder.scenpar.absTOA_lambda, write_defaults );
end

if ~isempty( h_builder.plpar )
    fprintf(fid,'\n%% ==================================================================================================\n');
    fprintf(fid,'%% Path-loss model\n');
    if ~isempty( ref_sec ) && ~isempty( ref_sec{8} )
        fprintf(fid,['%% See: ',ref,'; ',ref_sec{8},' \n']);
    end
    fprintf(fid,'%% ==================================================================================================\n');
    fprintf(fid,'\n');
    pl = h_builder.plpar;
    
    if strcmp( pl.model , 'logdist' )
        fprintf(fid,'%% Formula for Hata pathloss model:\n');
        fprintf(fid,'%% (Distance in meters, frequency in GHz)\n');
        fprintf(fid,'%%\n');
        fprintf(fid,'%%    PL = A * log10( d3D ) + B + C * log10( fGHz )\n');
        fprintf(fid,'\n');
    end
    
    if strcmp( pl.model , 'dual_slope' ) || strcmp( pl.model , 'nlos' )
        fprintf(fid,'%% Formula for dual-slope (LOS) pathloss model:\n');
        fprintf(fid,'%% (Distance in meters, frequency in GHz)\n');
        fprintf(fid,'%%\n');
        fprintf(fid,'%%     PL = PL1 for d2D <= dBP | PL2 for d2D > dBP\n');
        fprintf(fid,'%%    PL1 = A1 * log10( d3D ) + B + C * log10( fGHz ) + D * d3D\n');
        fprintf(fid,'%%    PL2 = PL1( dBP ) + A2 * log10( d3D / dBP )\n');
        fprintf(fid,'%%    dBP = E * ( hBS-hE ) * ( hMS-hE ) * fGHz\n');
        fprintf(fid,'\n');
    end
    
    if strcmp( pl.model , 'tripple_slope' ) 
        fprintf(fid,'%% Formula for triple-slope (LOS+GR) pathloss model:\n');
        fprintf(fid,'%% (Distance in meters, frequency in GHz)\n');
        fprintf(fid,'%%\n');
        fprintf(fid,'%%     PL = PL1 for d2D <= dBP1 | PL2 for dBP1 < d2D <= dBP2 | PL3 for d2D > dBP2\n');
        fprintf(fid,'%%    PL1 = A1 * log10( d3D ) + B + C * log10( fGHz ) + D * d3D\n');
        fprintf(fid,'%%    PL2 = PL1( dBP1 ) + A2 * log10( d3D / dBP1 )\n');
        fprintf(fid,'%%    PL3 = PL2( dBP2 ) + A3 * log10( d3D / dBP2 )\n');
        fprintf(fid,'%%   dBP1 = E1 * ( hBS-hE1 ) * ( hMS-hE1 ) * fGHz\n');
        fprintf(fid,'%%   dBP2 = E2 * ( hBS-hE2 ) * ( hMS-hE2 ) * fGHz\n');
        fprintf(fid,'\n');
    end
    
    if strcmp( pl.model , 'nlos' )
        fprintf(fid,'%% Formula for 3GPP NLOS pathloss model:\n');
        fprintf(fid,'%% (Distances and heights in meters, frequency in GHz)\n');
        fprintf(fid,'%%\n');
        fprintf(fid,'%%    PLn =  An * log10( d3D )\n');
        fprintf(fid,'%%        +  Bn\n');
        fprintf(fid,'%%        +  Cn * log10( fGHz )\n');
        fprintf(fid,'%%        +  Dn * log10( hBS )\n');
        fprintf(fid,'%%        + D1n * log10( hBS ) / hBS\n');
        fprintf(fid,'%%        + D2n * log10( hBS ) / hBS^2\n');
        fprintf(fid,'%%        + D3n * hBS\n');
        fprintf(fid,'%%        +  En * log10( hUT )\n');
        fprintf(fid,'%%        + E1n * log10( hUT ) / hUT\n');
        fprintf(fid,'%%        + E2n * log10( hUT ) / hUT^2\n');
        fprintf(fid,'%%        + E3n * hUT\n');
        fprintf(fid,'%%        +  Fn * log10( hBS ) * log10( d3d )\n');
        fprintf(fid,'%%        + G1n * log10^2( G2n * hUT )\n');
        fprintf(fid,'%%\n');
        fprintf(fid,'%%     PL = max( PL_LOS, PLn ) \n');
        fprintf(fid,'\n');
    end
    
    names = fieldnames( pl );
    for n = 1 : numel( names )
        if strcmp( names{n},'A' )
            if strcmp( pl.model , 'constant' )
                write_par_line( fid, ['PL_',names{n}], 'Reference PL in [dB]', pl.(names{n}), Inf );
            else
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of the PL [dB/log10(m)]', pl.(names{n}), Inf );
            end
                
                
        elseif strcmp( names{n},'A1' )
            if strcmp( pl.model , 'nlos' )
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of the LOS-PL before break-point [dB/log10(m)]', pl.(names{n}), Inf );
            else
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of the PL before break-point [dB/log10(m)]', pl.(names{n}), Inf );
            end
            
        elseif strcmp( names{n},'A2' )
            if strcmp( pl.model , 'nlos' )
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of the LOS-PL after break-point [dB/log10(m)]', pl.(names{n}), Inf );
            else
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of the PL after break-point [dB/log10(m)]', pl.(names{n}), Inf );
            end
            
        elseif strcmp( names{n},'A3' ) 
            write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of the PL after second break-point [dB/log10(m)]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'An' )
            write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of NLOS-PL [dB/log10(m)]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'B' )
            if strcmp( pl.model , 'nlos' )
                write_par_line( fid, ['PL_',names{n}], 'Reference LOS-PL in [dB]', pl.(names{n}), Inf, 1,0,1,0,0 );
            else
                write_par_line( fid, ['PL_',names{n}], 'Reference PL in [dB]', pl.(names{n}), Inf, 1,0,1,0,0 );
            end
            
        elseif strcmp( names{n},'Bn' )
            write_par_line( fid, ['PL_',names{n}], 'Reference NLOS-PL in [dB]', pl.(names{n}), Inf, 1,0,1,0,0 );
            
        elseif strcmp( names{n},'C' )
            if strcmp( pl.model , 'nlos' )
                write_par_line( fid, ['PL_',names{n}], 'Freq.-dep. of the LOS-PL in [dB/log10(GHz)]', pl.(names{n}), Inf );
            else
                write_par_line( fid, ['PL_',names{n}], 'Freq.-dep. of the PL in [dB/log10(GHz)]', pl.(names{n}), Inf );
            end
            
        elseif strcmp( names{n},'Cn' )
            write_par_line( fid, ['PL_',names{n}], 'Freq.-dep. of the NLOS-PL in [dB/log10(GHz)]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'D' )
            if strcmp( pl.model , 'nlos' )
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of LOS-PL [dB/m]', pl.(names{n}), Inf );
            elseif strcmp( pl.model , 'satellite' )
                write_par_line( fid, ['PL_',names{n}], 'Elevaion angle dep. of PL [dB/rad]', pl.(names{n}), Inf );
            else
                write_par_line( fid, ['PL_',names{n}], 'TX-RX 3D dist.-dep. of PL [dB/m]', pl.(names{n}), Inf );
            end
            
        elseif strcmp( names{n},'Dn' )
            write_par_line( fid, ['PL_',names{n}], 'TX height-dep. of the NLOS-PL in [dB/log10(m)]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'D1n' )
            write_par_line( fid, ['PL_',names{n}], 'TX height-dep. of the NLOS-PL in [dB/log10(m)/m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'D2n' )
            write_par_line( fid, ['PL_',names{n}], 'TX height-dep. of the NLOS-PL in [dB/log10(m)/m^2]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'D3n' )
            write_par_line( fid, ['PL_',names{n}], 'TX height-dep. of the NLOS-PL in [dB/m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'E' )
            write_par_line( fid, ['PL_',names{n}], 'Breakpoint scaling factor', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'E1' )
            write_par_line( fid, ['PL_',names{n}], 'Breakpoint scaling factor for first BP in [s/m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'E2' )
            write_par_line( fid, ['PL_',names{n}], 'Breakpoint scaling factor for second BP in [s/m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'En' )
            write_par_line( fid, ['PL_',names{n}], 'RX height-dep. of the NLOS-PL in [dB/log10(m)]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'E1n' )
            write_par_line( fid, ['PL_',names{n}], 'RX height-dep. of the NLOS-PL in [dB/log10(m)/m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'E2n' )
            write_par_line( fid, ['PL_',names{n}], 'RX height-dep. of the NLOS-PL in [dB/log10(m)/m^2]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'E3n' )
            write_par_line( fid, ['PL_',names{n}], 'RX height-dep. of the NLOS-PL in [dB/m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'Fn' )
            write_par_line( fid, ['PL_',names{n}], 'Combined TX-height and freq.-dep. of the NLOS-PL in [dB/(log10(m)*log10(GHz))]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'G1n' )
            write_par_line( fid, ['PL_',names{n}], 'RX height-dep. of the NLOS-PL on the square of the MT height [dB/(log10(m))^2]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'G2n' )
            write_par_line( fid, ['PL_',names{n}], 'RX height-dep. factor', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'hE' )
            write_par_line( fid, ['PL_',names{n}], 'Environment height in [m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'hE1' )
            write_par_line( fid, ['PL_',names{n}], 'Environment height for first BP in [m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'hE2' )
            write_par_line( fid, ['PL_',names{n}], 'Environment height for second BP in [m]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'sig1' )
            write_par_line( fid, ['PL_',names{n}], 'Shadow Fading STD before breakpoint [dB]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'sig2' )
            write_par_line( fid, ['PL_',names{n}], 'Shadow Fading STD after breakpoint [dB]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'sig3' )
            write_par_line( fid, ['PL_',names{n}], 'Shadow Fading STD after second breakpoint [dB]', pl.(names{n}), Inf );
            
        elseif strcmp( names{n},'usePLa' )
            write_par_line( fid, ['PL_',names{n}], 'Enable/disable attenuation by atmospheric gases', pl.(names{n}), Inf );
            
        else
            write_par_line( fid, ['PL_',names{n}], '', pl.(names{n}), Inf );
        end
    end
end

fprintf(fid,'\n');

fclose(fid);

end


function write_data_line( fid, val )
for n = 1:numel(val)
    v = num2str( val(n),'%1.2g' );
    if ~strcmp(v(1),'-')
        v = [' ',v];
    end
    fprintf(fid,v);
    for m = 1 : 7 - numel(v)
        fprintf(fid,' ');
    end
end
end

function write_comment_line( fid, mu, gamma, omega, epsilon, zeta, alpha )

fprintf( fid, num2str( mu,'%1.5g' ) );
if gamma ~= 0
    if gamma < 0
        fprintf(fid,' - ');
        fprintf(fid,num2str( -gamma,'%1.5g' ));
    else
        fprintf(fid,' + ');
        fprintf(fid,num2str( gamma,'%1.5g' ));
    end
    fprintf(fid,' * log10( ');
    if omega ~= 0
        fprintf(fid,num2str( omega,'%1.5g' ));
        fprintf(fid,' + ');
    end
    fprintf(fid,'fGHz )');
end
if epsilon ~= 0
    if epsilon < 0
        fprintf(fid,' - ');
        fprintf(fid,num2str( -epsilon,'%1.5g' ));
    else
        fprintf(fid,' + ');
        fprintf(fid,num2str( epsilon,'%1.5g' ));
    end
    fprintf(fid,' * log10( d2D )');
end
if zeta ~= 0
    if zeta < 0
        fprintf(fid,' - ');
        fprintf(fid,num2str( -zeta,'%1.5g' ));
    else
        fprintf(fid,' + ');
        fprintf(fid,num2str( zeta,'%1.5g' ));
    end
    fprintf(fid,' * log10( hBS )');
end
if alpha ~= 0
    if alpha < 0
        fprintf(fid,' - ');
        fprintf(fid,num2str( -alpha,'%1.5g' ));
    else
        fprintf(fid,' + ');
        fprintf(fid,num2str( alpha,'%1.5g' ));
    end
    fprintf(fid,' * log10( alpha_rad )');
end

end

function write_par_line( fid, name, comment, value, default, gamma, omega, epsilon, zeta, alpha )
if value ~= default
    fprintf( fid, [ name,' = ' ] );
    for n = 1 : 19 - numel(name)
        fprintf( fid, ' ' );
    end
    if isreal( value ) || imag( value ) == 0
        v = num2str( value,'%1.5g' );
    elseif imag(value) < 0
        v = [num2str( real(value),'%1.5g' ),'-',num2str( abs(imag(value)),'%1.5g' ),'j'];
    else
        v = [num2str( real(value),'%1.5g' ),'+',num2str( imag(value),'%1.5g' ),'j'];
    end
    if ~strcmp(v(1),'-')
        v = [' ',v];
    end
    fprintf( fid, v );
    if ~isempty( comment )
        for n = 1 : 13 - numel(v)
            fprintf( fid, ' ' );
        end
        
        if exist('gamma','var')
            comment = [ comment, ' @ ' ];
            if gamma ~= 0
                comment = [ comment, num2str( 1-omega ),' GHz, '];
            end
            if epsilon ~= 0
                comment = [ comment, '1 m TX-RX dist., '];
            end
            if zeta ~= 0
                comment = [ comment, '1 m TX height, '];
            end
            if alpha ~= 0
                comment = [ comment, '57.3 deg elevation, '];
            end
            comment = comment(1:end-2);
        end
        fprintf( fid, [ ' %% ',comment,'\n' ] );
    else
        fprintf( fid, '\n' );
    end
end

end
