function [ scenarios, file_names, file_dir ] = supported_scenarios( parse_shortnames )
%SUPPORTED_SCENARIOS Returns a list of supported scenarios
%
% Calling object:
%   None (static method)
%
% Input:
%   parse_shortnames
%   This optional parameter can disable (0) the shortname parsing. This is significantly faster.
%   By default shortname parsing is enabled (1)
%
% Output:
%   scenarios
%   A cell array containing the scenario names. Those can be used in 'qd_track.scenario'
%
%   file_names
%   The names of the configuration files for each scenario
%
%   file_dir
%   The directory where each file was found. You can place configuration file also in you current
%   working directory
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

% Parse input variables.
if nargin > 0
    parse_shortnames = logical(parse_shortnames(1));
else
    parse_shortnames = true;
end

% Parse config files in current folder
files_tmp = dir('*.conf');

files = [];
m = 1;
for n = 1:numel(files_tmp)
    files(m).name = files_tmp(n).name;
    files(m).dir  = '';
    m = m + 1;
end

% Parse files in config-folder
quadriga_path = path;
if isempty( regexp( quadriga_path,';' ) )   % Linux separates path entries by ":"
    quadriga_path = regexp(quadriga_path, ':?([^:]*quadriga_src)', 'tokens'); % Linux
else % Windows separated path entries by ";"
    quadriga_path = regexp(quadriga_path, ':?([^;]*quadriga_src)', 'tokens'); % Windows
end
config_folder = fullfile(quadriga_path{1}{1}, 'config');
files_tmp = dir(fullfile(config_folder, '*.conf'));

for n = 1:numel(files_tmp)
    files(m).name = files_tmp(n).name;
    files(m).dir  = config_folder;
    m = m + 1;
end

nfiles = numel(files);

scenarios = cell(1,1000);
if parse_shortnames
    file_names = cell(1,1000);
    file_dir   = cell(1,1000);
end

m = 0;
for n = 1 : nfiles
    m = m + 1;
    [~, scenarios{m}] = fileparts( files(n).name );
    file_names{m} = files(n).name;
    file_dir{m}   = files(n).dir;
    
    if parse_shortnames
        % Read the file content to find short names for the scenario
        file = fopen(fullfile(files(n).dir, files(n).name), 'r');
        lin = fgets(file);
        while ischar(lin)
            p1 = regexp(lin,'[ \t]*ShortName[ \t]*=[ \t]*[A-Za-z0-9_]+','match');
            if ~isempty(p1)
                p2 = regexp( p1 ,'=');
                short_name = regexp( p1{1}(p2{1}+1:end) , '[A-Za-z0-9_]+' ,'match' );
                
                if any( strcmp( short_name{1} , scenarios(1:m) ) )
                    error( ['ShortName "',short_name{1},'" in "',files(n).name,'" is ambiguous.'] )
                end
                
                m = m+1;
                scenarios{m}  = short_name{1};
                file_names{m} = files(n).name;
                file_dir{m}   = files(n).dir;
            end
            lin = fgets(file);
        end
        fclose(file);
    end
end

scenarios = scenarios(1:m);
if parse_shortnames
    file_names = file_names(1:m);
    file_dir = file_dir(1:m);
end

end
