function h_builder_out = split_multi_freq( h_builder )
%SPLIT_MULTI_FREQ Split the builder objects for multi-frequency simulations 
%
% Calling object:
%   Object array
%
% Description:
%   QuaDRiGa allows multi-frequency-simulations by setting multiple carrier frequencies in
%   "qd_simulation_parameters.center_frequency". Correlated SSF for multi-frequency simulations is
%   an additional feature of the 3GPP model (see Section 7.6.5, pp 57 of TR 38.901 V14.1.0).
%
%   Identical parameters for each Frequency:
%    * LOS / NLOS state must be the same
%    * BS and MT positions are the same (antenna element positions are different!)
%    * Cluster delays and angles for each multipath component are the same
%    * Spatial consistency of the LSPs is identical
%
%   Differences:
%    * Antenna patterns are different for each frequency
%    * Path-loss is different for each frequency
%    * Path-powers are different for each frequency
%    * Delay- and angular spreads are different
%    * K-Factor is different
%    * XPR of the NLOS components is different
%
%    The method "split_multi_freq" separates the builder objects so that each builder creates
%    channels for only one frequency. If you call "split_multi_freq" before any LSF and SSF
%    parameters are generated as it is done the the following code, then LSF parameters (e.g. SF,
%    DS, AS) will be uncorrelated for each frequency. If you call "gen_lsf_parameters" before
%    "split_multi_freq", then all LSF parameters will be fully correlated. However, frequency
%    dependent averages and variances still apply. If you call "gen_lsf_parameters" and
%    "gen_ssf_parameters" before "split_multi_freq", then SSF will also  correlated, i.e. the same
%    paths will be seen at each frequency.
%
% Output:
%   h_builder_out
%   An array of "qd_builder" objects where where each builder only served one frequency.
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

% Get the number of frequencies
n_freq = numel( h_builder(1,1).simpar(1,1).center_frequency );

if numel(h_builder) > 1
    
    sic  = size( h_builder );
    n_bs = sic(2);
    h_builder_out = qd_builder;
    for i_cb = 1 : numel(h_builder)
        [ i1,i2 ] = qf.qind2sub( sic, i_cb );
        
        % Check if the number of frequencies is the same in all builder objects
        if numel( h_builder(i1,i2).simpar(1,1).center_frequency ) ~= n_freq
            error('QuaDRiGa:qd_builder:split_multi_freq','Number of frequencies is inconsistent in builder objects.');
        end
        
        % io = (i2-1)*n_freq + (1:n_freq);
        io = i2 + ( 0 : n_freq-1 ) * n_bs;
        h_builder_out( i1,io ) = split_multi_freq( h_builder(i1,i2) );
    end
    
elseif n_freq > 1
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    split_lsf = true;       % Check if LSF parameters have been initialized
    if h_builder.no_rx_positions > 0 && size( h_builder.ds,1 ) ~= n_freq
        split_lsf = false;
    end
    split_ssf = true;       % Check if SSF parameters have been initialized
    if h_builder.no_rx_positions > 0 && size( h_builder.gain,3 ) ~= n_freq
        split_ssf = false;
    end
    
    h_builder_out = qd_builder;
    for f = 1 : n_freq
        if f > 1
            h_builder_out(1,f) = qd_builder;
        end
        
        % Add the frequency information to the Tx name
        % It will later be written to the channel objects
        ii = regexp( h_builder.name,'_' );
        if numel( ii ) ~= 1
            error('QuaDRiGa:qd_builder:split_multi_freq','Builder name must have the form "scenario_txname"');
        end
        name = [ h_builder.name(1:ii), 'F', num2str(f,'%02d'), '-',h_builder.name(ii+1:end) ];
        h_builder_out(1,f).name         = name;
           
        if h_builder.no_rx_positions ~= 0
            
            % Copy data that is identical for all frequencies
            h_builder_out(1,f).scenpar_nocheck = h_builder.scenpar;
            h_builder_out(1,f).Pscenario    = h_builder.Pscenario;
            h_builder_out(1,f).plpar        = h_builder.plpar;
            h_builder_out(1,f).tx_position  = h_builder.tx_position;
            h_builder_out(1,f).rx_positions = h_builder.rx_positions;
            h_builder_out(1,f).tx_track     = h_builder.tx_track;
            h_builder_out(1,f).NumClusters  = h_builder.NumClusters;
            h_builder_out(1,f).NumSubPaths  = h_builder.NumSubPaths;
            h_builder_out(1,f).AoD          = h_builder.AoD;
            h_builder_out(1,f).AoA          = h_builder.AoA;
            h_builder_out(1,f).EoD          = h_builder.EoD;
            h_builder_out(1,f).EoA          = h_builder.EoA;
            h_builder_out(1,f).absTOA_offset = h_builder.absTOA_offset;
            
            % Split SOS objects
            h_builder_out(1,f).sos          = h_builder.sos;
            h_builder_out(1,f).gr_sos       = h_builder.gr_sos;
            h_builder_out(1,f).path_sos     = h_builder.path_sos;
            h_builder_out(1,f).clst_dl_sos  = h_builder.clst_dl_sos;
            if ~isempty( h_builder.xpr_sos )
                h_builder_out(1,f).xpr_sos  = h_builder.xpr_sos(:,:,f);
            end
            if ~isempty( h_builder.pin_sos )
                h_builder_out(1,f).pin_sos  = h_builder.pin_sos(:,:,f);
            end
            h_builder_out(1,f).absTOA_sos   = h_builder.absTOA_sos;
            
            % Split the LSF variables
            if split_lsf
                h_builder_out(1,f).ds     	= h_builder.ds(f,:);
                h_builder_out(1,f).kf       = h_builder.kf(f,:);
                h_builder_out(1,f).sf       = h_builder.sf(f,:);
                h_builder_out(1,f).asD      = h_builder.asD(f,:);
                h_builder_out(1,f).asA      = h_builder.asA(f,:);
                h_builder_out(1,f).esD      = h_builder.esD(f,:);
                h_builder_out(1,f).esA      = h_builder.esA(f,:);
                h_builder_out(1,f).xpr      = h_builder.xpr(f,:);
            end
            if ~isempty( h_builder.gr_epsilon_r )
                h_builder_out(1,f).gr_epsilon_r = h_builder.gr_epsilon_r(f,:);
            end
            
            % Copy the frequency-dependent SSF variables
            if split_ssf
                if size( h_builder.taus,3 ) == n_freq
                    h_builder_out(1,f).taus     = h_builder.taus(:,:,f);
                else
                    h_builder_out(1,f).taus     = h_builder.taus;
                end
                h_builder_out(1,f).gain         = h_builder.gain(:,:,f);
                h_builder_out(1,f).xprmat       = h_builder.xprmat(:,:,:,f);
                h_builder_out(1,f).pin          = h_builder.pin(:,:,f);
                h_builder_out(1,f).subpath_coupling = h_builder.subpath_coupling(:,:,f);
            end
            if ~isempty( h_builder.fbs_pos ) && ~isempty( h_builder.lbs_pos )
                h_builder_out(1,f).fbs_pos      = h_builder.fbs_pos(:,:,:,f);
                h_builder_out(1,f).lbs_pos      = h_builder.lbs_pos(:,:,:,f);
            end
            
            % Split the frequencies in simpar
            simpar = copy( h_builder.simpar );
            simpar.center_frequency = simpar.center_frequency(f);
            h_builder_out(1,f).simpar = simpar;
            
            % Split the tx-array (only copy the handles)
            if size( h_builder.tx_array,1 ) == n_freq
                h_builder_out(1,f).tx_array = h_builder.tx_array(f,:);
            elseif size( h_builder.tx_array,1 ) == 1
                h_builder_out(1,f).tx_array = h_builder.tx_array(1,:);
            else
                error('QuaDRiGa:qd_builder:split_multi_freq','Tx-array object size does not match the number of frequencies.');
            end
            
            % Split the rx-array (only copy the handles)
            if size( h_builder.rx_array,1 ) == n_freq
                h_builder_out(1,f).rx_array = h_builder.rx_array(f,:);
            elseif size( h_builder.rx_array,1 ) == 1
                h_builder_out(1,f).rx_array = h_builder.rx_array(1,:);
            else
                error('QuaDRiGa:qd_builder:split_multi_freq','Rx-array object size does not match the number of frequencies.');
            end
            
            % Split the track objects including the LSPs
            rx_track = qd_track;
            for t = 1 : size( h_builder.rx_track , 2 )
                if isempty( h_builder.rx_track(1,t).par )
                    rx_track(1,t) = h_builder.rx_track(1,t);
                else
                    rx_track(1,t) = qd_track([]);
                    copy( rx_track(1,t), h_builder.rx_track(1,t) );
                    data = rx_track(1,t).par;
                    fn = fieldnames( data );
                    for d = 1 : numel( fn )
                        val = data.( fn{d} );
                        if ~isempty( val )
                            if size( val,3 ) == n_freq 
                                data.( fn{d} ) = val( :,:,f );
                            else
                                data.( fn{d} ) = val( :,:,1 );     % For 'o2i_d3din'
                            end
                        end
                    end
                    rx_track(1,t).par = data;
                end
            end
            h_builder_out(1,f).rx_track = rx_track;
            
            % Check if the builders are for dual mobility
            check_dual_mobility( h_builder_out );
        end
    end
    
else % n_freq == 1    
    h_builder_out = h_builder; % Copy handle
end

end
