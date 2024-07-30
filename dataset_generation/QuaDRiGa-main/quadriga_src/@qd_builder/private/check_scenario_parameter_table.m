function check_scenario_parameter_table(value, check)
%CHECK_SCENARIO_PARAMETER_TABLE Checks the scenario table
%
%   CHECK_SCENARIO_PARAMETER_TABLE is a private function of the parameter_set
%   class. It can not be called from outside the class members and is thus not
%   directly accessible to the user. However, when the scenpar structure of a
%   parameter_set object is edited, check_scenario_parameter_table is immediately
%   called to check the validity of the new parameters. Changing the Winner table
%   (scenpar) in a loop can thus be very slow. This functions also calculates the
%   cross-correlation matrix (LSP_xcorr_matrix) and checks if it is positive
%   definite (LSP_matrix_isOK). If this matrix is not positive definite, then
%   the generation of correlated LSPs will fail (update_parameters will return an
%   error).
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

if nargin == 1
    check = true;
end

if check
    
    t_int = {'NumClusters','NumSubPaths'};
    
    t_positive = {'r_DS',};
   
    t_positive_or_zero = {'PerClusterAS_D','PerClusterAS_A','PerClusterES_D',...
        'PerClusterES_A','LNS_ksi','LOS_scatter_radius','GR_enabled','SC_lambda',...
        'DS_lambda','AS_D_lambda','AS_A_lambda',...
        'ES_D_lambda','ES_A_lambda','SF_lambda','KF_lambda','XPR_lambda',...
        'PerClusterDS','PerClusterDS_min','absTOA_lambda'};
    
    t_real = {'XPR_mu','XPR_sigma','XPR_gamma','XPR_delta',...
        'DS_mu','DS_sigma','DS_gamma','DS_delta',...
        'AS_D_mu','AS_D_sigma','AS_D_gamma','AS_D_delta',...
        'AS_A_mu','AS_A_sigma','AS_A_gamma','AS_A_delta',...
        'ES_D_mu','ES_D_mu_A','ES_D_mu_min','ES_D_sigma','ES_D_gamma','ES_D_delta',...
        'ES_A_mu','ES_A_sigma','ES_A_gamma','ES_A_delta',...
        'SF_sigma','SF_delta',...
        'KF_mu','KF_sigma','KF_gamma','KF_delta','PerClusterDS_gamma', 'absTOA_mu','absTOA_sigma'};
    
    t_abs1 = { 'asD_ds','asA_ds','asA_sf','asD_sf','ds_sf','asD_asA','asD_kf',...
        'asA_kf','ds_kf','sf_kf','esD_ds', 'esA_ds','esA_sf','esD_sf','esD_esA',...
        'esD_asD','esD_asA','esA_asD','esA_asA','esD_kf','esA_kf'};
    
    names = fieldnames(value);
    if numel(names) ~= 140
        error('QuaDRiGa:qd_builder:WrongInput','??? Wrong number of fields in "scenpar".');
    end
    for n = 1:numel(names)
        v = value.(names{n});
        if ismember( names(n) , t_int ) &&...
                ~( all(size(v) == [1 1]) && isnumeric(v) && isreal(v) && mod(v,1)==0 && v > 0 )
            error('QuaDRiGa:qd_builder:WrongInput',['??? "',names{n},'" must be integer, scalar and > 0'])
        end
        if ismember( names(n) , t_positive ) &&...
                ~( all(size(v) == [1 1]) && isnumeric(v) && isreal(v) && v > 0 )
            error('QuaDRiGa:qd_builder:WrongInput',['??? "',names{n},'" must be real, scalar and > 0'])
        end
        if ismember( names(n) , t_positive_or_zero ) &&...
                ~( all(size(v) == [1 1]) && isnumeric(v) && isreal(v) && v >= 0 )
            error('QuaDRiGa:qd_builder:WrongInput',['??? "',names{n},'" must be real, scalar and >= 0'])
        end
        if ismember( names(n) , t_real ) &&...
                ~( all(size(v) == [1 1]) && isnumeric(v) && isreal(v) )
            error('QuaDRiGa:qd_builder:WrongInput',['??? "',names{n},'" must be real and scalar'])
        end
        if ismember( names(n) , t_abs1 ) &&...
                ~( all(size(v) == [1 1]) && isnumeric(v) && isreal(v) && v<=1 && v>=-1 )
            error('QuaDRiGa:qd_builder:WrongInput',['??? "',names{n},'" must be real, scalar and have values from -1 to 1'])
        end
    end
end
