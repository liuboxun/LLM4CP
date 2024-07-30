function h_builder_out = split_rx( h_builder )
%SPLIT_RX Split the builder objects into single builders for each rx location 
%
% Calling object:
%   Object array
%
% Description:
%   The method "split_rx" separates the builder objects so that each builder creates channels for
%   only one rx location. If you call "split_rx" before any LSF and SSF parameters are generated,
%   then LSF parameters (e.g. SF, DS, AS) will be uncorrelated for each user. If you call
%   "gen_lsf_parameters" before "split_rx", then all LSF parameters will be fully correlated. If
%   you call "split_rx" and "gen_ssf_parameters" before "split_rx", then SSF will also be
%   correlated.
%
% Output:
%   h_builder_out
%   An array of "qd_builder" objects where where each builder only serves one rx.
%
%
% QuaDRiGa Copyright (C) 2011-2020
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

if numel(h_builder) > 1
    sic  = size( h_builder );
    h_builder_out = qd_builder;
    i_out = 0;
    for i_cb = 1 : numel(h_builder)
        [ i1,i2 ] = qf.qind2sub( sic, i_cb );
        n_rx = h_builder(i1,i2).no_rx_positions;
        if n_rx > 0.5
            h_builder_out( 1, i_out+1:i_out+n_rx) = split_rx( h_builder(i1,i2) );
            i_out = i_out + n_rx;
        end
    end
    
elseif h_builder(1,1).no_rx_positions > 0
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    if h_builder.dual_mobility == -1
        h_builder.check_dual_mobility;
    end
    
    n_rx = h_builder.no_rx_positions;
    
    split_lsf = true;       % Check if LSF parameters have been initialized
    if h_builder.no_rx_positions > 0 && size( h_builder.ds,2 ) ~= n_rx
        split_lsf = false;
    end
    split_ssf = true;       % Check if SSF parameters have been initialized
    if h_builder.no_rx_positions > 0 && size( h_builder.gain,1 ) ~= n_rx
        split_ssf = false;
    end
    
    h_builder_out = qd_builder;
    for r = 1 : n_rx
        if r > 1
            h_builder_out(1,r) = qd_builder;
        end
        
        % Copy data that is identical for all rx
        h_builder_out(1,r).name         = h_builder.name;
        h_builder_out(1,r).Pscenario    = h_builder.Pscenario;
        h_builder_out(1,r).simpar       = h_builder.simpar;
        h_builder_out(1,r).scenpar_nocheck = h_builder.scenpar;
        h_builder_out(1,r).plpar        = h_builder.plpar;
        h_builder_out(1,r).NumClusters  = h_builder.NumClusters;
        h_builder_out(1,r).NumSubPaths  = h_builder.NumSubPaths;
        h_builder_out(1,r).subpath_coupling = h_builder.subpath_coupling;
        
        % Copy SOS object handles
        h_builder_out(1,r).sos          = h_builder.sos;
        h_builder_out(1,r).gr_sos       = h_builder.gr_sos;
        h_builder_out(1,r).path_sos     = h_builder.path_sos;
        h_builder_out(1,r).clst_dl_sos  = h_builder.clst_dl_sos;
        h_builder_out(1,r).xpr_sos      = h_builder.xpr_sos;
        h_builder_out(1,r).pin_sos      = h_builder.pin_sos;
        h_builder_out(1,r).absTOA_sos   = h_builder.absTOA_sos;

        % Split tracks and antennas
        h_builder_out(1,r).tx_position  = h_builder.tx_position(:,r);
        h_builder_out(1,r).rx_positions = h_builder.rx_positions(:,r);
        h_builder_out(1,r).tx_array     = h_builder.tx_array(:,r);
        h_builder_out(1,r).rx_array     = h_builder.rx_array(:,r);
        h_builder_out(1,r).rx_track     = h_builder.rx_track(1,r);
        h_builder_out(1,r).tx_track     = h_builder.tx_track(1,r);
        
        % Split the LSF variables
        if split_lsf
            h_builder_out(1,r).ds       = h_builder.ds(:,r);
            h_builder_out(1,r).kf       = h_builder.kf(:,r);
            h_builder_out(1,r).sf       = h_builder.sf(:,r);
            h_builder_out(1,r).asD      = h_builder.asD(:,r);
            h_builder_out(1,r).asA      = h_builder.asA(:,r);
            h_builder_out(1,r).esD      = h_builder.esD(:,r);
            h_builder_out(1,r).esA      = h_builder.esA(:,r);
            h_builder_out(1,r).xpr      = h_builder.xpr(:,r);
        end
        if ~isempty( h_builder.gr_epsilon_r )
            h_builder_out(1,r).gr_epsilon_r = h_builder.gr_epsilon_r(:,r);
        end
        if ~isempty( h_builder.absTOA_offset )
            h_builder_out(1,r).absTOA_offset = h_builder.absTOA_offset(:,r);
        end
        
        % Split SSF variables
        if split_ssf
            h_builder_out(1,r).taus     = h_builder.taus(r,:,:);
            h_builder_out(1,r).gain     = h_builder.gain(r,:,:);
            h_builder_out(1,r).AoD      = h_builder.AoD(r,:);
            h_builder_out(1,r).AoA      = h_builder.AoA(r,:);
            h_builder_out(1,r).EoD      = h_builder.EoD(r,:);
            h_builder_out(1,r).EoA      = h_builder.EoA(r,:);
            h_builder_out(1,r).xprmat   = h_builder.xprmat(:,:,r,:);      
            h_builder_out(1,r).pin      = h_builder.pin(r,:,:);
        end
        if ~isempty( h_builder.fbs_pos )
            h_builder_out(1,r).fbs_pos  = h_builder.fbs_pos(:,:,r,:);
            h_builder_out(1,r).lbs_pos  = h_builder.lbs_pos(:,:,r,:);
        end
        h_builder_out(1,r).dual_mobility = h_builder.dual_mobility;
    end
    
else
    h_builder_out = copy( h_builder );
end

end
