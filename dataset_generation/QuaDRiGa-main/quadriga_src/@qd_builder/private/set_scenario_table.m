function file_name = set_scenario_table( h_builder, scenario, check )
%SET_SCENARIO_TABLE Parse conf files and load values into MATLAB
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

if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "set_scenario_table" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

% We can optionally disable the validity check for speedup.
if nargin == 3 && numel( check ) == 1
    check = logical( check );
else
    check = true;
end

if ischar(scenario)
    
    % Get the list of conf-files
    [ sup_scenarios , file_names , file_dirs ] = qd_builder.supported_scenarios(0);
    
    % Parse all shortnames inside the conf-files
    if ~any( logical( strcmp(scenario,sup_scenarios) ) )
        [ sup_scenarios , file_names , file_dirs ] = qd_builder.supported_scenarios(1);
    end
    
    if ~any( logical( strcmp(scenario,sup_scenarios) ) )
        str = ['??? Scenario "',scenario,'" not found; supported are: '];
        no = numel(sup_scenarios);
        for n = 1:no
            str = [str,sup_scenarios{n}];
            if n<no
                str = [str,', '];
            end
        end
        error('QuaDRiGa:qd_builder:WrongInput',str);
    end
    
    % Initialize the structs with default values
    scen = struct( ...
        'NumClusters' , 6 , ...            % Number of Paths
        'NumSubPaths' , 20 , ...           % Sub-paths per cluster
        'DS_mu' , -6 , ...                 % Reference value [log10(s)]
        'DS_sigma' , 0.5 , ...             % Reference STD [log10(s)]
        'DS_lambda' , 20 , ...             % Decorrelation distance [m]
        'DS_omega' , 0 , ...               % Reference frequency offset [GHz]
        'DS_gamma' , 0 , ...               % Freq.-dep. [log10(s)/log10(GHz)]
        'DS_epsilon' , 0 , ...             % Dist.-dep. [log10(s)/log10(m)]
        'DS_zeta' , 0 , ...                % BS-height-dep. [log10(s)/log10(m)]
        'DS_alpha' , 0 , ...               % TX-elevation-dep. [log10(s)/log10(rad)]
        'DS_delta' , 0 , ...               % Freq.-dep. of STD [log10(s)/log10(GHz)]
        'DS_kappa' , 0 , ...               % Dist.-dep. of STD [log10(s)/log10(m)]
        'DS_tau' , 0 , ...                 % BS-height-dep. of STD [log10(s)/log10(m)]
        'DS_beta' , 0 , ...                % TX-elevation-dep. of STD [log10(s)/log10(rad)]
        'AS_D_mu' , 1 , ...                % Reference value [log10(deg)]
        'AS_D_sigma' , 0.2 , ...           % Reference STD [log10(deg)]
        'AS_D_lambda' , 20 , ...           % Decorrelation distance [m]
        'AS_D_omega' , 0 , ...             % Reference frequency offset [GHz]
        'AS_D_gamma' , 0 , ...             % Freq.-dep. [log10(deg)/log10(GHz)]
        'AS_D_epsilon' , 0 , ...           % Dist.-dep. [log10(deg)/log10(m)]
        'AS_D_zeta' , 0 , ...              % BS-height-dep. [log10(deg)/log10(m)]
        'AS_D_alpha' , 0 , ...             % TX-elevation-dep. [log10(deg)/log10(rad)]
        'AS_D_delta' , 0 , ...             % Freq.-dep. of STD [log10(deg)/log10(GHz)]
        'AS_D_kappa' , 0 , ...             % Dist.-dep. of STD [log10(deg)/log10(m)]
        'AS_D_tau' , 0 , ...               % BS-height-dep. of STD [log10(deg)/log10(m)]
        'AS_D_beta' , 0 , ...              % TX-elevation-dep. of STD [log10(deg)/log10(rad)]
        'AS_A_mu' , 1 , ...                % Reference value [log10(deg)]
        'AS_A_sigma' , 0.2 , ...           % Reference STD [log10(deg)]
        'AS_A_lambda' , 20 , ...           % Decorrelation distance [m]
        'AS_A_omega' , 0 , ...             % Reference frequency offset [GHz]
        'AS_A_gamma' , 0 , ...             % Freq.-dep. [log10(deg)/log10(GHz)]
        'AS_A_epsilon' , 0 , ...           % Dist.-dep. [log10(deg)/log10(m)]
        'AS_A_zeta' , 0 , ...              % BS-height-dep. [log10(deg)/log10(m)]
        'AS_A_alpha' , 0 , ...             % TX-elevation-dep. [log10(deg)/log10(rad)]
        'AS_A_delta' , 0 , ...             % Freq.-dep. of STD [log10(deg)/log10(GHz)]
        'AS_A_kappa' , 0 , ...             % Dist.-dep. of STD [log10(deg)/log10(m)]
        'AS_A_tau' , 0 , ...               % BS-height-dep. of STD [log10(deg)/log10(m)]
        'AS_A_beta' , 0 , ...              % TX-elevation-dep. of STD [log10(deg)/log10(rad)]
        'ES_D_mu' , 0.3 , ...              % Reference value [log10(deg)]
        'ES_D_sigma' , 0.2 , ...           % Reference STD [log10(deg)]
        'ES_D_lambda' , 20 , ...           % Decorrelation distance [m]
        'ES_D_omega' , 0 , ...             % Reference frequency offset [GHz]
        'ES_D_gamma' , 0 , ...             % Freq.-dep. [log10(deg)/log10(GHz)]
        'ES_D_epsilon' , 0 , ...           % Dist.-dep. [log10(deg)/log10(m)]
        'ES_D_zeta' , 0 , ...              % BS-height-dep. [log10(deg)/log10(m)]
        'ES_D_alpha' , 0 , ...             % TX-elevation-dep. [log10(deg)/log10(rad)]
        'ES_D_delta' , 0 , ...             % Freq.-dep. of STD [log10(deg)/log10(GHz)]
        'ES_D_kappa' , 0 , ...             % Dist.-dep. of STD [log10(deg)/log10(m)]
        'ES_D_tau' , 0 , ...               % BS-height-dep. of STD [log10(deg)/log10(m)]
        'ES_D_beta' , 0 , ...              % TX-elevation-dep. of STD [log10(deg)/log10(rad)]
        'ES_D_mu_A' , 0 , ...              % departure elevation spread distance dependency (legacy)
        'ES_D_mu_min' , -Inf , ...         % departure elevation spread minimum value
        'ES_A_mu' , 0.3 , ...              % Reference value [log10(deg)]
        'ES_A_sigma' , 0.2 , ...           % Reference STD [log10(deg)]
        'ES_A_lambda' , 20 , ...           % Decorrelation distance [m]
        'ES_A_omega' , 0 , ...             % Reference frequency offset [GHz]
        'ES_A_gamma' , 0 , ...             % Freq.-dep. [log10(deg)/log10(GHz)]
        'ES_A_epsilon' , 0 , ...           % Dist.-dep. [log10(deg)/log10(m)]
        'ES_A_zeta' , 0 , ...              % BS-height-dep. [log10(deg)/log10(m)]
        'ES_A_alpha' , 0 , ...             % TX-elevation-dep. [log10(deg)/log10(rad)]
        'ES_A_delta' , 0 , ...             % Freq.-dep. of STD [log10(deg)/log10(GHz)]
        'ES_A_kappa' , 0 , ...             % Dist.-dep. of STD [log10(deg)/log10(m)]
        'ES_A_tau' , 0 , ...               % BS-height-dep. of STD [log10(deg)/log10(m)]
        'ES_A_beta' , 0 , ...              % TX-elevation-dep. of STD [log10(deg)/log10(rad)]
        'SF_sigma' , 8 , ...               % Reference STD [log10(s)]
        'SF_lambda' , 20 , ...             % Decorrelation distance [m]
        'SF_omega' , 0 , ...               % Reference frequency offset [GHz]
        'SF_delta' , 0 , ...               % Freq.-dep. of STD [log10(s)/log10(GHz)]
        'SF_kappa' , 0 , ...               % Dist.-dep. of STD [log10(s)/log10(m)]
        'SF_tau' , 0 , ...                 % BS-height-dep. of STD [log10(s)/log10(m)]
        'SF_beta' , 0 , ...                % TX-elevation-dep. of STD [log10(s)/log10(rad)]
        'KF_mu' , 0 , ...                  % Reference value [log10(s)]
        'KF_sigma' , 0 , ...               % Reference STD [log10(s)]
        'KF_lambda' , 20 , ...             % Decorrelation distance [m]
        'KF_omega' , 0 , ...               % Reference frequency offset [GHz]
        'KF_gamma' , 0 , ...               % Freq.-dep. [log10(s)/log10(GHz)]
        'KF_epsilon' , 0 , ...             % Dist.-dep. [log10(s)/log10(m)]
        'KF_zeta' , 0 , ...                % BS-height-dep. [log10(s)/log10(m)]
        'KF_alpha' , 0 , ...               % TX-elevation-dep. [log10(s)/log10(rad)]
        'KF_delta' , 0 , ...               % Freq.-dep. of STD [log10(s)/log10(GHz)]
        'KF_kappa' , 0 , ...               % Dist.-dep. of STD [log10(s)/log10(m)]
        'KF_tau' , 0 , ...                 % BS-height-dep. of STD [log10(s)/log10(m)]
        'KF_beta' , 0 , ...                % TX-elevation-dep. of STD [log10(s)/log10(rad)]
        'XPR_mu' , 0 , ...                 % Reference value [log10(s)]
        'XPR_sigma' , 10 , ...             % Reference STD [log10(s)]
        'XPR_lambda' , 20 , ...            % Decorrelation distance [m]
        'XPR_omega' , 0 , ...              % Reference frequency offset [GHz]
        'XPR_gamma' , 0 , ...              % Freq.-dep. [log10(s)/log10(GHz)]
        'XPR_epsilon' , 0 , ...            % Dist.-dep. [log10(s)/log10(m)]
        'XPR_zeta' , 0 , ...               % BS-height-dep. [log10(s)/log10(m)]
        'XPR_alpha' , 0 , ...              % TX-elevation-dep. [log10(s)/log10(rad)]
        'XPR_delta' , 0 , ...              % Freq.-dep. of STD [log10(s)/log10(GHz)]
        'XPR_kappa' , 0 , ...              % Dist.-dep. of STD [log10(s)/log10(m)]
        'XPR_tau' , 0 , ...                % BS-height-dep. of STD [log10(s)/log10(m)]
        'XPR_beta' , 0 , ...               % TX-elevation-dep. of STD [log10(s)/log10(rad)]
        'r_DS' , 2 , ...                   % Delays spread proportionality factor
        'PerClusterDS', 0 , ...            % Cluster Delay Spread factor cDS in [ns]
        'PerClusterDS_gamma', 0 , ...      % Freq.-dep. of Cluster DS [ns/log10(GHz)]
        'PerClusterDS_min', 0 , ...        % Minimum Cluster Delay Spread in [ns]
        'PerClusterAS_D' , 0 , ...         % Per cluster BS azimuth spread [deg]
        'PerClusterAS_A' , 0 , ...         % Per cluster MS azimuth spread [deg]
        'PerClusterES_D' , 0 , ...         % Per cluster BS elevation spread [deg]
        'PerClusterES_A' , 0 , ...         % Per cluster MS elevation spread [deg]
        'absTOA_mu' , -100 , ...           % Absolute time of arrival offset reference value [log10(s)]
        'absTOA_sigma' , 0 , ...           % Absolute time of arrival offset reference STD [log10(s)]
        'absTOA_lambda' , 0 , ...          % Absolute time of arrival offset decorrelation distance [m]
        'SubpathMethod', 'legacy', ...     % Subpath-generation method
        'LOS_scatter_radius', 0, ...       % Scattering around the LOS cluster
        'SC_lambda', 0 ,...                % Decorrelation distance [m] for spatial consistency
        'LNS_ksi' , 3 , ...                % ZDSC LNS ksi [dB], per cluster shadowing
        'GR_enabled' , 0 , ...             % Enable (1) or disable (0) Ground Reflection
        'GR_epsilon' , 0 , ...             % Manual value for the relative permittivity, 0 = auto
        'asD_ds' , 0 , ...                 % departure AS vs delay spread
        'asA_ds' , 0 , ...                 % arrival AS vs delay spread
        'asA_sf' , 0 , ...                 % arrival AS vs shadowing std
        'asD_sf' , 0 , ...                 % departure AS vs shadowing std
        'ds_sf' , 0 , ...                  % delay spread vs shadowing std
        'asD_asA' , 0 , ...                % departure AS vs arrival AS
        'asD_kf' , 0 , ...                 % departure AS vs k-factor
        'asA_kf' , 0 , ...                 % arrival AS vs k-factor
        'ds_kf' , 0 , ...                  % delay spread vs k-factor
        'sf_kf' , 0 , ...                  % shadowing std vs k-factor
        'esD_ds' , 0 , ...                 % departure ES vs delay spread
        'esA_ds' , 0 , ...                 % arrival ES vs delay spread
        'esA_sf' , 0 , ...                 % arrival ES vs shadowing std
        'esD_sf' , 0 , ...                 % departure ES vs shadowing std
        'esD_esA' , 0 , ...                % departure ES vs arrival ES
        'esD_asD' , 0 , ...                % departure ES vs departure AS
        'esD_asA' , 0 , ...                % departure ES vs arrival AS
        'esA_asD' , 0 , ...                % arrival ES vs departure AS
        'esA_asA' , 0 , ...                % arrival ES vs arrival AS
        'esD_kf' , 0 , ...                 % departure ES vs k-factor
        'esA_kf' , 0 , ...                 % arrival ES vs k-factor
        'xpr_ds' , 0 , ...                 % XPR vs. DS
        'xpr_kf' , 0 , ...                 % XPR vs. DS
        'xpr_sf' , 0 , ...                 % XPR vs. DS
        'xpr_asd' , 0 , ...                % XPR vs. DS
        'xpr_asa' , 0 , ...                % XPR vs. DS
        'xpr_esd' , 0 , ...                % XPR vs. DS
        'xpr_esa' , 0 );                   % XPR vs. DS

    names = fieldnames(scen);
    
    % Open config file for reading the parameters
    ind = find( strcmp( scenario , sup_scenarios ),1 );
    file_name = file_names{ind};
    file_dir  = file_dirs{ind};
    
    file = fopen(fullfile(file_dir, file_name), 'r');
    
    % Read file line by line and parse the data of scenpar
    lin = fgetl(file);
    while ischar(lin)
        
        p1 = regexp(lin,'=');                           % Check if there is an equal sign
        if ~isempty(p1)                                 % If there is a "=" sign
            p2 = regexp(lin(1:p1(1)-1),'%', 'once');    % Check if the line is commented
            if isempty(p2)                              % If the line is not commented
                name = regexp( lin(1:p1(1)-1) ,'[A-Za-z0-9_]+','match');           % Read name
                if ~isempty(name)                       % If there is a name
                    
                    % Here we have two options. Either the current parameter is
                    % for "scenpar" or for "plpar". Parameters for the
                    % path-loss model start with "PL_". We check first, if the
                    % current line is a PL-parameter.
                    
                    p0 = regexp(name{1},'PL_', 'once');
                    if isempty(p0)                      % If it is not a plpar
                        ind = strcmp(name,names);       % Get the index in the parameter list
                        if any(ind)                     % If the index exists
                            tmp = lin(p1(1)+1:end);     % Copy line
                            p2 = regexp(tmp,'%');       % Check for comment
                            if ~isempty( p2 )
                                tmp = tmp(1:p2(1)-1 );      % Remove commented part
                            end
                            p3 = regexp( tmp,'[0-9.-]+','match');         % Check for numbers
                            if ~isempty( p3 )
                                p3 = regexp( tmp,'[0-9Eeji.+-]+','match'); % Include exponentials and complex numbers
                                scen.( names{ind} ) = str2double( p3{1} );
                            else
                                p3 = regexp( tmp,'[A-Za-z]+' ,'match');   % Check for strings
                                if ~isempty( p3 )
                                    scen.( names{ind} ) = p3{1};
                                else
                                    error('QuaDRiGa:qd_builder:WrongInput',...
                                        ['Could not understand value of "',name{1},'" in file "',scenario,'.conf"']);
                                end
                            end
                        end
                    else
                        PLname = name{1}(4:end);        % Parse the name of the field in PLPAR
                        if strcmp(PLname,'model')       % If it is the model name
                            p3 = regexp( lin(p1(1)+1:end)  ,'[A-Za-z0-9_]+','match');
                            if isempty( p3 )            % If no value is given or wrongly formatted
                                error('QuaDRiGa:qd_builder:WrongInput',...
                                    ['Could not understand value of "',name{1},'" in file "',scenario,'.conf"']);
                            else
                                Lpl.model = p3{1};      % Set the model name
                            end
                        else
                            p4 = regexp( lin, '%' );    % Determine the beginning of the comment
                            if isempty( p4 )            % If there is no comment
                                p4 = numel(lin);        % Set the position to endline
                            end
                            p3 = regexp( lin(p1(1)+1:p4)  ,'[0-9e.-]+','match');
                            Lpl.( PLname ) = str2double(p3);
                        end
                    end
                end
            end
        end
        lin = fgetl(file);          % Get next line
    end
    fclose(file);
    
    % If we have a PLPAR, write it to data object
    if exist('Lpl','var')
        h_builder.plpar = Lpl;
    else
        h_builder.plpar = [];
    end
    
    % Check the scenpar and determine the LSP correlation matrix
    check_scenario_parameter_table( scen , check );
    
    h_builder.Pscenpar = scen;
    h_builder.Pscenario = scenario;
    
elseif isstruct( scenario )
    check_scenario_parameter_table( scenario , check );
    h_builder.Pscenario = 'Custom';
    h_builder.Pscenpar = scenario;
    
else
    error('QuaDRiGa:qd_builder:WrongInput','Scenpar has wrong format');
end

end
