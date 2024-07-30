function set_scenario( h_track, scenario, probability, seg_length_min, seg_length_mu, seg_length_std )
%SET_SCENARIO Assigns random scenarios and creates segments.
%
% Calling object:
%   Object array
%
% Description:
%   This function can be used to create segments along the trajectory and assign scenarios to the
%   segments.  If there are less than 3 input arguments (i.e. only 'scenario' and/or 'probability'
%   is given), then no segments will be created. To create segments with the default settings call
%   'set_scenario(scenario,[],[])'. Alternatively, it is possible to only create segments by
%   leaving the scenario empty, e.g. by calling 'set_scenario([],[],[])'.
%
% Input:
%   scenario
%   A cell array of scenario-names. Each scenario (synonym for propagation environment) is
%   described by a string (e.g. "MIMOSA_16-25_LOS" or "WINNER_SMa_C1_NLOS"). A list of supported
%   scenarios can be obtained by calling 'parameter_set.supported_scenarios'. The scenario
%   parameters are stored in the configuration folder "config" in the QuaDRiGa main folder. The
%   filenames (e.g. "MIMOSA_16-25_LOS.conf") also serves as scenario name.
%
%   probability
%   The probability for which the scenario occurs. This parameter must be a vector of the same
%   length as there are scenarios. Probabilities must be specified in between 0 and 1. The sum of
%   the probabilities must be 1. By default (or when 'probability' is set to '[]'), each scenario
%   is equally likely.
%
%   seg_length_min
%   the minimal segment length in [m]. The default is 10 m.
%
%   seg_length_mu
%   the median segment length in [m]. The default is 30 m.
%
%   seg_length_std
%   the standard deviation of the street length in [m]. The default is 12 m.
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


% Parse "scenario"
if ~exist( 'scenario' , 'var' )
    error('"scenario" is not given.')
end
if ischar(scenario)
    scenario_list_old = scenario;
    scenario = cell(1,1);
    scenario{1} = scenario_list_old;
end
no_scenario_list = numel( scenario );

% Parse "probability"
if ~exist( 'probability' , 'var' ) || isempty(probability)
    probability = ones(no_scenario_list,1)./no_scenario_list;
elseif numel(probability) ~= numel(scenario)
    error('??? number of elements in "probability" must match the number of elements in "scenario".');
elseif ~( isnumeric(probability) && isreal(probability) &&...
        all( probability >= 0) && all( probability <= 1) )
    error('??? "probability"has wrong format.');
end
probability = probability(:) ./ sum( probability(:) );
cum_probability = cumsum( probability );

% If there are more than 3 inputs, we create segments on the track
create_segments = false;
if nargin > 3
    create_segments = true;
    
    % Parse "seg_length_min"
    if ~exist( 'seg_length_min' , 'var' ) || isempty( seg_length_min )
        seg_length_min = 10;
    elseif ~(isnumeric( seg_length_min ) && seg_length_min>=0 &&...
            isreal( seg_length_min ) && all(size(seg_length_min) == [1 1]))
        error('??? "seg_length_min"  has wrong format');
    end
    
    % Parse "seg_length_mu"
    if ~exist( 'seg_length_mu' , 'var' ) || isempty( seg_length_mu )
        seg_length_mu = 30;
    elseif ~(isnumeric( seg_length_mu ) && seg_length_mu>=0 &&...
            isreal( seg_length_mu ) && all(size(seg_length_mu) == [1 1]))
        error('??? "seg_length_mu"  has wrong format');
    end
    
    % Parse "seg_length_std"
    if ~exist( 'seg_length_std' , 'var' ) || isempty( seg_length_std )
        seg_length_std = 12;
    elseif ~(isnumeric( seg_length_std ) && seg_length_std>=0 &&...
            isreal( seg_length_std ) && all(size(seg_length_std) == [1 1]))
        error('??? "seg_length_mu"  has wrong format');
    end
end


if numel(h_track) > 1
    
    sic = size( h_track );
    prc = false( sic );
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            
            if create_segments
                set_scenario( h_track(i1,i2,i3,i4), scenario, probability,...
                    seg_length_min, seg_length_mu, seg_length_std );
            else
                set_scenario( h_track(i1,i2,i3,i4), scenario, probability );
            end
            
            prc( qf.eqo( h_track(i1,i2,i3,i4), h_track ) ) = true;
        end
    end

else
    h_track = h_track(1,1);
    
    % There are several scenarios, one for each tx in the layout
    % Determine the number of Tx:
    n_tx = size( h_track.scenario ,1);
    
    if create_segments
                
        [ trk_length , dist ] = get_length( h_track );
        dist_offset = min(diff(dist))/2;
        no_snapshots = h_track.no_snapshots;
        
        pos = 0;
        no_segments = 1;
        while pos < trk_length
            
            % Get the length of the current segment
            seg_length = randn*seg_length_std + seg_length_mu;
            while seg_length < seg_length_min
                seg_length = randn*seg_length_std + seg_length_mu;
            end
            ind = find( pos + seg_length - dist_offset < dist,1 );
            pos = dist( ind );
            
            if ind < no_snapshots     % Exception for the last segment
                no_segments = h_track.no_segments + 1;
                h_track.no_segments = no_segments;
                h_track.segment_index( no_segments ) = ind;
                
                if ~isempty( scenario )
                    for i_tx = 1:n_tx
                        i_scen = find( rand < cum_probability,1 );
                        h_track.scenario{ i_tx,no_segments-1 } = scenario{i_scen};
                    end
                end
            end
        end
        % Set last scenario
        if ~isempty( scenario )
            for i_tx = 1:n_tx
                i_scen = find( rand < cum_probability,1 );
                h_track.scenario{ i_tx,no_segments } = scenario{i_scen};
            end
        end
        
    else
        % If segments are already given, assign random scenarios
        if ~isempty( scenario )
            for i_segment = 1:h_track.no_segments
                for i_tx = 1:n_tx
                    i_scen = find( rand < cum_probability,1 );
                    h_track.scenario{ i_tx,i_segment } = scenario{i_scen};
                end
            end
        end
    end

end

end
