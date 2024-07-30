function indoor_rx = set_scenario( h_layout, scenario, rx, tx, indoor_frc, SC_lambda_rx, SC_lambda_tx, parse_shortnames )
%SET_SCENARIO Assigns scenarios to tracks and segments.
%
% Calling object:
%   Single object
%
% Description:
%   This function can be used to assign scenarios to tracks and segments of tracks. This takes the
%   distance-dependent LOS probability into account for some specific scenarios. Currently,
%   distance-dependent scenario selection is available for:
%
%      * 3GPP_3D_UMi
%      * 3GPP_3D_UMa
%      * 3GPP_38.901_UMi
%      * 3GPP_38.901_UMa
%      * 3GPP_38.901_RMa
%      * 3GPP_38.901_Indoor_Mixed_Office
%      * 3GPP_38.901_Indoor_Open_Office
%      * 3GPP_38.901_InF_SL
%      * 3GPP_38.901_InF_DL
%      * 3GPP_38.901_InF_SH
%      * 3GPP_38.901_InF_DH      
%      * 5G-ALLSTAR_DenseUrban (Satellite)
%      * 5G-ALLSTAR_Urban (Satellite)
%      * 5G-ALLSTAR_Suburban (Satellite)
%      * 5G-ALLSTAR_Rural (Satellite)
%      * mmMAGIC_UMi
%      * mmMAGIC_Indoor
%      * QuaDRiGa_Industrial
%      * QuaDRiGa_UD2D
%
%
%    Alternatively, you can use all scenarios specified in 'qd_builder.supported_scenarios'.
%
% Input:
%   scenario
%   A string containing the scenario name
%
%   rx
%   A vector containing the receiver indices for which the scenarios should be set. Default: all
%   receivers
%
%   tx
%   A vector containing the transmitter indices for which the scenarios should be set. Default: all
%   transmitters
%
%   indoor_frc
%   The fraction of the users (number between 0 and 1) that are indoors
%
%   SC_lambda_rx
%   Decorrelation distance for spatially consistent LOS states based on the RX position. It set to
%   0, LOS sates are uncorrelated with respect to RX position. The default setting depends on the
%   values of '3GPP_baseline' in the simulation setting. For '3GPP_baseline = 1', spatially
%   consistent LOS states are uncorrelated otherwise, the values set as defined in 3GPP TR 38.901,
%   Table 7.6.3.1-2.
%
%   SC_lambda_tx
%   Decorrelation distance for spatially consistent LOS states based on the TX position. It set to
%   0, LOS sates are uncorrelated with respect to TX position.
%
% Output:
%   indoor_rx
%   A logical vector indicating if a user is indoors (1) or outdoors (0)
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

if numel( h_layout ) > 1
    error('QuaDRiGa:qd_layout:set_scenario','set_scenario not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

no_rx = h_layout.no_rx;
no_tx = h_layout.no_tx;

if ~exist( 'scenario' , 'var' )
    scenario = [];
elseif iscell( scenario ) || ~ischar( scenario )
    error('Scenario must be a string.')
end

if ~exist( 'rx' , 'var' ) || isempty(rx)
    rx = 1 : no_rx;
end

if ~exist( 'tx' , 'var' ) || isempty(tx)
    tx = 1 : no_tx;
end

if ~exist( 'indoor_frc' , 'var' ) || isempty(indoor_frc)
    indoor_frc = 0;
end

if ~exist( 'SC_lambda_rx' , 'var' ) || isempty(SC_lambda_rx)
    SC_lambda_rx = [];
end

if ~exist( 'SC_lambda_tx' , 'var' ) || isempty(SC_lambda_tx)
    SC_lambda_tx = [];
end

if ~exist( 'parse_shortnames' , 'var' ) || isempty(parse_shortnames)
    parse_shortnames = true;
end

indoor_rx = [];

if any( strcmpi( scenario, qd_builder.supported_scenarios(parse_shortnames) ))
    % Set single scenarios
    
    scen_current = {scenario};  % Convert to 1x1 cell array
    for i_rx = 1 : numel(rx)
        
        % Extract existing scenario information
        no_segments = h_layout.rx_track( 1,rx( i_rx ) ).no_segments;
        tmp = size( h_layout.rx_track( 1,rx( i_rx ) ).scenario );
        if tmp(1) == no_tx && tmp(2) == no_segments
            scen_old = h_layout.rx_track( 1,rx( i_rx ) ).scenario;
        elseif tmp(1) == 1 && tmp(2) == no_segments
            scen_old = h_layout.rx_track( 1,rx( i_rx ) ).scenario( ones(1,no_tx),: );
        elseif tmp(1) == 1 && tmp(2) == 1
            scen_old = h_layout.rx_track( 1,rx( i_rx ) ).scenario( ones(no_tx,no_segments ));
        elseif tmp(1) == no_tx && tmp(2) == 1
            scen_old = h_layout.rx_track( 1,rx( i_rx ) ).scenario( :,ones(1,no_segments ));
        else
            error(['Scenario definition dimension mismatch for Rx ',num2str(rx( i_rx ))]);
        end
               
        % Assign scenario to track
        for i_tx = 1 : numel( tx )
            scen_old( tx( i_tx ),: ) = scen_current;
        end
        h_layout.rx_track( 1,rx( i_rx ) ).scenario = scen_old;
    end
    
else
    
    % Set the default SC_lambda_rx values
    if h_layout.simpar(1,1).use_3GPP_baseline && isempty( SC_lambda_rx )
        SC_lambda_rx = 0;   % Disable
    elseif isempty( SC_lambda_rx )
        switch scenario % 3GPP TR 38.901 v16.1.0, Table 7.6.3.1-2
            case { '3GPP_38.901_RMa' }
                SC_lambda_rx = 60;
            case { '3GPP_38.901_UMi', '3GPP_38.901_UMa','mmMAGIC_UMi',...
                    '5G-ALLSTAR_DenseUrban','5G-ALLSTAR_Urban','5G-ALLSTAR_Suburban','5G-ALLSTAR_Rural' }
                SC_lambda_rx = 50;
            case { '3GPP_38.901_Indoor_Mixed_Office','3GPP_38.901_Indoor_Open_Office','mmMAGIC_Indoor' }  
                SC_lambda_rx = 10;
            case { 'QuaDRiGa_Industrial' }
                SC_lambda_rx = 25;
            case { '3GPP_38.901_InF_SL', '3GPP_38.901_InF_SH' }
                SC_lambda_rx = 5; % dclutter/2 (38.901 table 7.2-4)
            case { '3GPP_38.901_InF_DL', '3GPP_38.901_InF_DH' }
                SC_lambda_rx = 1; % dclutter/2 (38.901 table 7.2-4)
            otherwise
                SC_lambda_rx = 0;
        end
        if h_layout.simpar(1,1).show_progress_bars && SC_lambda_rx > 0
            disp(['Setting RX LOS state correlation distance to ',num2str(SC_lambda_rx),' m'])
        end
    end
    
    % Set the default SC_lambda_tx values
    if h_layout.simpar(1,1).use_3GPP_baseline && isempty( SC_lambda_tx )
        SC_lambda_tx = 0;   % Disable
    elseif isempty( SC_lambda_tx ) && numel(tx) > 1
        switch scenario
            case {'QuaDRiGa_UD2D' , 'QuaDRiGa_Industrial' }
                % Dual Mobility Scenarios
                SC_lambda_tx = SC_lambda_rx;
                
            case { '5G-ALLSTAR_DenseUrban','5G-ALLSTAR_Urban','5G-ALLSTAR_Suburban','5G-ALLSTAR_Rural' }
                % Satellite Scenarios
                hnn = positions_abs( h_layout.tx_track(:,tx) );
                hnn = cat(2,hnn{:});
                hnn = max(hnn(3,:));                % Orbit height above NN
                
                % Calculate length of the visible orbit from the UE positions
                pos = trans_global2ue( -90:0.01:90, zeros(1,18001), hnn(1,ones(1,18001)), [0,0] );
                pos = pos(:,pos(3,:)>0);
                visible_orbit_lengt = sum( sqrt(sum((pos(1,2:end) - pos(1,1:end-1)).^2,1)) );
                
                % The TX decorrelation distance is a fraction of the visible sky
                SC_lambda_tx = 0.1 * visible_orbit_lengt;
                
            otherwise
                SC_lambda_tx = 0;
        end
        if h_layout.simpar(1,1).show_progress_bars && SC_lambda_tx > 0
            disp(['Setting TX LOS state correlation distance to ',num2str(SC_lambda_tx),' m'])
        end
    elseif isempty( SC_lambda_tx )
        SC_lambda_tx = 0;   % Disable
    end
    
    % Generate O2I penetration loss assuming that all users are indoor
    if indoor_frc > 0
        switch scenario
            case { '3GPP_38.901_UMi', '3GPP_38.901_UMa' }
                h_layout.gen_o2i_loss( '3GPP_38.901', 0.5, SC_lambda_rx, 25 );
            case { '3GPP_38.901_RMa' }
                h_layout.gen_o2i_loss( '3GPP_38.901', 1, SC_lambda_rx, 10 );
            case { 'mmMAGIC_UMi' }
                h_layout.gen_o2i_loss( 'mmMAGIC', 0.5, SC_lambda_rx, 25 );
        end
    end
    
    % Determine if the user is indoor. If the MT is mobile, it stays indoors.
    switch scenario
        case { '3GPP_38.901_UMi', '3GPP_38.901_UMa', '3GPP_38.901_RMa', 'mmMAGIC_UMi',...
                '3GPP_3D_UMi', '3GPP_3D_UMa'   }
            if SC_lambda_rx == 0 || h_layout.no_rx < 1.5
                tmp = rand( 1, numel(rx) );                                 % Random
            else
                tmp = qd_sos.rand( SC_lambda_rx , h_layout.rx_position );  	% Spatially consistent
                
                % Renormalize the distribution
                bins = 0:0.01:1;
                cdf = qf.acdf(tmp,bins);
                [cdfU,ii] = unique(cdf);
                binsU = bins(ii);
                binsU(end) = 1;
                tmp = qf.interp( binsU, [], cdfU.', tmp );
                
            end
            indoor_rx = tmp < indoor_frc;
            
            % If MT is outdoor, remove indoor distance and indoor-loss
            for i_rx = 1 : numel(rx)
                if indoor_rx( i_rx ) == 0 && ~isempty( h_layout.rx_track(1,i_rx).par )
                    h_layout.rx_track(1,i_rx).par = [];
                end
            end
            
        otherwise
            indoor_rx = false( 1, numel(rx) );
    end
    
    % Get the MT positions, including segments on tracks
    [ rx_pos_3d, tx_pos_3d, rx_ind ] = parse_positions( h_layout );
    rx_pos_3d = rx_pos_3d(:,:,ones(1,no_tx));
    no_rx_pos = size( rx_pos_3d,2 );
    
    % Generate spatially consistent random variables for the LOS / NLOS state
    if SC_lambda_rx == 0 && SC_lambda_tx == 0       % LOS state is uncorrelated
        randC = rand(no_tx,no_rx_pos);
        
    elseif SC_lambda_tx == 0                        % LOS state does not depend on TX position
        randC = qd_sos.rand( SC_lambda_rx(1,ones(1,no_tx)), rx_pos_3d(:,:,1) );
        randC = reshape( randC', no_rx_pos, no_tx ).';
        
    elseif SC_lambda_rx == 0                        % LOS state does not depend on RX position  
        randC = zeros( no_tx,no_rx_pos );
        for n = 1:no_rx_pos
            randC(:,n) = qd_sos.rand( SC_lambda_tx, reshape( tx_pos_3d(:,n,:),3,[]) ).';
        end
        
    elseif SC_lambda_rx == SC_lambda_tx             % TX and RX decorrelation distance are identical
        randC = qd_sos.rand( SC_lambda_rx, reshape(rx_pos_3d,3,[]), reshape(tx_pos_3d,3,[]) );
        randC = reshape( randC', no_rx_pos, no_tx ).';
        
    else                                            % TX and RX decorrelation distances are different
        h_sos = qd_sos( 'Comb300', 'Uniform', SC_lambda_rx );
        h_sos.dist_decorr(1,2) = SC_lambda_tx;
        randC = h_sos.val( reshape(rx_pos_3d,3,[]),reshape(tx_pos_3d,3,[]) );
        randC = reshape( randC', no_rx_pos, no_tx ).';
    end
    
    % Spatial COnsistency might change the distribution. We fix this here.
    % Calculate the CDF in the output values.
    if numel( randC ) > 1.5
        bins = 0:0.01:1;
        cdf = qf.acdf(randC(:),bins);
        
        % Obtain unique values for the renormalization
        [cdfU,ii] = unique(cdf);
        binsU = bins(ii);
        binsU(end) = 1;
        
        % Interpolate output such that the values match a Uniform distribution
        sic = size(randC);
        randC = qf.interp( binsU, [], cdfU.', randC(:) );
        randC = reshape( randC, sic );
    end
    
    if 0  % Debugging plot
        plot(bins,cdf,'--k');
        cdfA = qf.acdf(randC',bins);
        cdfI = qf.acdf(randC(:),bins);
        hold on
        plot(bins,cdfA);
        plot(bins,cdfI,'-k','Linewidth',2);
        plot([0,1],[0,1],':k')
        hold off
    end
   
    % Determine LOS probabilities and set the scenario accordingly
    rC = 1;                   % Index of the MT in randC
    for i_rx = 1 : numel(rx)
        rxi     = rx_ind(1,:) == rx( i_rx );
        nS      = sum(rxi);
        randR   = randC(:,rC:rC+nS-1);
        rC      = rC+nS;
        
        % Make a local copy of the current track handle to improve performance
        rx_trackL = h_layout.Prx_track( 1,rx( i_rx ) );

        segment_index = rx_trackL.segment_index;
        
        dist_3d = rx_pos_3d(:,rxi,:) - tx_pos_3d(:,rxi,:);
        dist_2d = sqrt( sum( dist_3d(1:2,:).^2 ,1) );
        dist_2d = reshape( dist_2d, nS, [] ).';
        dist_3d = sqrt( sum( dist_3d.^2 ,1) );
        dist_3d = reshape( dist_3d, nS, [] ).';
        
        % Calculate the outdoor 2D and 3D distance (for indoor scenarios)
        if ~isempty( rx_trackL.par )
            dist_3d_in = rx_trackL.par.o2i_d3din;
            dist_2d_in = dist_2d .* dist_3d_in ./ dist_3d;
            dist_2d = dist_2d - dist_2d_in;
        end
        
        % Determine LOS probability
        switch scenario
            case '3GPP_3D_UMi'
                % See: 3GPP TR 36.873 V12.1.0 (2015-03)
                scen = { '3GPP_3D_UMi_LOS', '3GPP_3D_UMi_NLOS',...
                    '3GPP_3D_UMi_LOS_O2I', '3GPP_3D_UMi_NLOS_O2I' };
                
                % Determine the LOS probability for each BS-MT
                p_LOS = min( 18./dist_2d , 1 ) .* (1-exp(-dist_2d/36)) + exp(-dist_2d/36);
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case '3GPP_3D_UMa'
                % See: 3GPP TR 36.873 V12.1.0 (2015-03)
                scen = { '3GPP_3D_UMa_LOS', '3GPP_3D_UMa_NLOS',...
                    '3GPP_3D_UMa_LOS_O2I', '3GPP_3D_UMa_NLOS_O2I' };
                
                % Include height-dependency of the user terminals
                h_UT = rx_trackL.positions( 3,segment_index ) + rx_trackL.initial_position(3);
                
                C = zeros( size( dist_2d ));
                
                % Exclude outdoor users from height-dependency
                if indoor_rx(i_rx)
                    g = C;
                    
                    ii = dist_2d > 18;
                    g(ii) = 1.25e-6 .* dist_2d(ii).^2 .* exp( -dist_2d(ii)/150 );
                    
                    ii = h_UT > 13 & h_UT < 23;
                    if any(ii)
                        C( :,ii ) = ones(no_tx,1) * ((h_UT(ii)-13)/10).^1.5;
                    end
                    
                    C( :,h_UT >= 23 ) = 1;
                    C = C .* g;
                end
                
                % Determine the LOS probability for each BS-MT
                p_LOS = ( min( 18./dist_2d , 1 ) .* (1-exp(-dist_2d/63)) + exp(-dist_2d/63) )...
                    .* (1+C);
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case '3GPP_38.901_UMi'
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_UMi_LOS', '3GPP_38.901_UMi_NLOS',...
                    '3GPP_38.901_UMi_LOS_O2I', '3GPP_38.901_UMi_NLOS_O2I' };
                
                % Determine the LOS probability for each BS-MT
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 18;
                p_LOS( ii )  = 18./dist_2d(ii) + exp(-dist_2d(ii)/36) .* (1 - 18./dist_2d(ii));
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case '3GPP_38.901_UMa' 
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_UMa_LOS', '3GPP_38.901_UMa_NLOS',...
                    '3GPP_38.901_UMa_LOS_O2I', '3GPP_38.901_UMa_NLOS_O2I' };
                
                % Include height-dependency of the user terminals
                h_UT = rx_trackL.positions( 3,segment_index ) + rx_trackL.initial_position(3);
                
                C = zeros( size( dist_2d ));
                ii = h_UT > 13 & h_UT < 23;
                if any( ii )
                    C( :,ii ) = ones(no_tx,1) * ((h_UT(ii)-13)/10).^1.5;
                end
                C( :,h_UT >= 23 ) = 1;
                
                % Determine the LOS probability for each BS-MT
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 18;
                p_LOS( ii )  = ( 18./dist_2d(ii) + exp(-dist_2d(ii)/63) .* (1 - 18./dist_2d(ii)) ) .* ...
                    ( 1 + 5/4*C(ii) .* ( dist_2d(ii)./100 ).^3 .* exp(-dist_2d(ii)/150) );
                
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case '3GPP_38.901_RMa' 
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_RMa_LOS', '3GPP_38.901_RMa_NLOS',...
                    '3GPP_38.901_RMa_LOS_O2I', '3GPP_38.901_RMa_NLOS_O2I' };
                
                % Determine the LOS probability for each BS-MT
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 10;
                p_LOS( ii )  = exp( -(dist_2d(ii)-10)/1000 );
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case '3GPP_38.901_Indoor_Mixed_Office'
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                scen = { '3GPP_38.901_Indoor_LOS', '3GPP_38.901_Indoor_NLOS' };
                
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 1.2 & dist_2d < 6.5;
                p_LOS( ii )  = exp( -(dist_2d(ii)-1.2)/4.7 );
                ii = dist_2d >= 6.5;
                p_LOS( ii )  = exp( -(dist_2d(ii)-6.5)/32.6 )*0.32;
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case '3GPP_38.901_Indoor_Open_Office'
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                scen = { '3GPP_38.901_Indoor_LOS', '3GPP_38.901_Indoor_NLOS' };
                
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 5 & dist_2d <= 49;
                p_LOS( ii )  = exp( -(dist_2d(ii)-5)/70.8 );
                ii = dist_2d > 49;
                p_LOS( ii )  = exp( -(dist_2d(ii)-49)/211.7 )*0.54;
                i_LOS = ( randR >= p_LOS ) + 1;
                
            case '5G-ALLSTAR_DenseUrban'
                % See: 3GPP TR 38.811 V15.2.0 (2019-09), pp 48, Table 6.6.1-1
                scen = { '5G-ALLSTAR_DenseUrban_LOS', '5G-ALLSTAR_DenseUrban_NLOS', 'Null' };
                
                % Height of the TX above the receivers tangential plane
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                
                % Elevation angle in [degree]
                elevation = atand( h_tx ./ dist_2d );
                
                % LOS probabilities vs. elevation angle in [%] from 3GPP TR 38.811
                los_prob = [ 0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1 ];
                p_LOS = qf.interp( [-90,0:10:90], [], los_prob, elevation(:).'  ).';
                p_LOS = reshape( p_LOS,[],nS );
                
                % Get the scenario index
                i_LOS = ( randR >= p_LOS/100 ) + 1;
                i_LOS( h_tx<0 ) = 3;    % Satellite is below the horizon
                
            case '5G-ALLSTAR_Urban'
                % See: 3GPP TR 38.811 V15.2.0 (2019-09), pp 48, Table 6.6.1-1
                scen = { '5G-ALLSTAR_Urban_LOS', '5G-ALLSTAR_Urban_NLOS', 'Null' };
                
                % Height of the TX above the receivers tangential plane
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                
                % Elevation angle in [degree]
                elevation = atand( h_tx ./ dist_2d );
                
                % LOS probabilities vs. elevation angle in [%] from 3GPP TR 38.811
                los_prob = [ 0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2 ];
                p_LOS = qf.interp( [-90,0:10:90], [], los_prob, elevation(:).'  ).';
                p_LOS = reshape( p_LOS,[],nS );
                
                % Get the scenario index
                i_LOS = ( randR >= p_LOS/100 ) + 1;
                i_LOS( h_tx<0 ) = 3;    % Satellite is below the horizon
                
            case '5G-ALLSTAR_Suburban'
                % See: 3GPP TR 38.811 V15.2.0 (2019-09), pp 48, Table 6.6.1-1
                scen = { '5G-ALLSTAR_Suburban_LOS', '5G-ALLSTAR_Suburban_NLOS', 'Null' };
                
                % Height of the TX above the receivers tangential plane
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                
                % Elevation angle in [degree]
                elevation = atand( h_tx ./ dist_2d );
                
                % LOS probabilities vs. elevation angle in [%] from 3GPP TR 38.811
                los_prob = [ 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8 ];
                p_LOS = qf.interp( [-90,0:10:90], [], los_prob, elevation(:).'  ).';
                p_LOS = reshape( p_LOS,[],nS );
                
                % Get the scenario index
                i_LOS = ( randR >= p_LOS/100 ) + 1;
                i_LOS( h_tx<0 ) = 3;    % Satellite is below the horizon
                
            case '5G-ALLSTAR_Rural'
                % See: 3GPP TR 38.811 V15.2.0 (2019-09), pp 48, Table 6.6.1-1
                scen = { '5G-ALLSTAR_Rural_LOS', '5G-ALLSTAR_Rural_NLOS', 'Null' };
                
                % Height of the TX above the receivers tangential plane
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                
                % Elevation angle in [degree]
                elevation = atand( h_tx ./ dist_2d );
                
                % LOS probabilities vs. elevation angle in [%] from 3GPP TR 38.811
                los_prob = [ 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8 ];
                p_LOS = qf.interp( [-90,0:10:90], [], los_prob, elevation(:).'  ).';
                p_LOS = reshape( p_LOS,[],nS );
                
                % Get the scenario index
                i_LOS = ( randR >= p_LOS/100 ) + 1;
                i_LOS( h_tx<0 ) = 3;    % Satellite is below the horizon
                
            case 'mmMAGIC_UMi'
                % Same LOS probability model as in 3GPP is used
                % See: mmMAGIC Deliverable D2.2, Table 4.1
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { 'mmMAGIC_UMi_LOS', 'mmMAGIC_UMi_NLOS',...
                    'mmMAGIC_UMi_LOS_O2I', 'mmMAGIC_UMi_NLOS_O2I' };
                
                % Determine the LOS probability for each BS-MT
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 18;
                p_LOS( ii )  = 18./dist_2d(ii) + exp(-dist_2d(ii)/36) .* (1 - 18./dist_2d(ii));
                
                i_LOS = ( randR >= p_LOS ) + 1;
                
                
            case 'mmMAGIC_Indoor'
                % Same LOS probability model as in 3GPP is used (mixed office)
                % See: mmMAGIC Deliverable D2.2, Table 4.1
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { 'mmMAGIC_Indoor_LOS', 'mmMAGIC_Indoor_NLOS' };

                % Determine the LOS probability for each BS-MT
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 1.2 & dist_2d < 6.5;
                p_LOS( ii )  = exp( -(dist_2d(ii)-1.2)/4.7 );
                ii = dist_2d >= 6.5;
                p_LOS( ii )  = exp( -(dist_2d(ii)-6.5)/32.6 )*0.32;
                i_LOS = ( randR >= p_LOS ) + 1;
                
            case 'QuaDRiGa_Industrial'
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { 'QuaDRiGa_Industrial_LOS', 'QuaDRiGa_Industrial_NLOS' };
                
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                h_tx = min( h_tx, 8 );
                
                p_LOS = exp( -(dist_2d+20)/80 ) .* exp( h_tx./5 - 1 );
                p_LOS = min( p_LOS, 1 );
                
                i_LOS = ( randR >= p_LOS ) + 1; 
                
            case 'QuaDRiGa_UD2D'
                % Same as 3GPP TR 38.901 UMi
                % See: 3GPP TR 38.901 V14.1.0 (2017-06) p27 Table 7.4.2-1
                
                % The corresponding scenario configuration files
                scen = { 'QuaDRiGa_UD2D_LOS', 'QuaDRiGa_UD2D_NLOS' };
                
                % Determine the LOS probability for each BS-MT
                p_LOS = ones( size( dist_2d ));
                ii = dist_2d > 18;
                p_LOS( ii )  = 18./dist_2d(ii) + exp(-dist_2d(ii)/36) .* (1 - 18./dist_2d(ii));
                i_LOS = ( randR >= p_LOS ) + 1;
                
            case '3GPP_38.901_InF_SL'
                % See: 3GPP TR 38.901 V16.0.0, pp 30, Table 7.4.2-1
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_InF_LOS', '3GPP_38.901_InF_NLOS_SL'};
                % Determine the LOS probability for each BS-MT
                r = 0.2; % clutter density (fix at 20% or use a random value under 40%) *38.901 table 7.2-4*
                if ~exist('r','var')
                    r = rand(1,1);
                    while (r>= 0.4)
                        r = rand(1,1);
                    end
                end
               
                d_clutter = 10; %clutter size *38.901 table 7.2-4*
                ksub = -d_clutter/log(1-r);
                p_LOS = ones( size( dist_2d ));
                p_LOS = exp(-dist_2d./ksub);
                i_LOS = ( randR >= p_LOS ) + 1;
                
            case '3GPP_38.901_InF_DL'
                % See: 3GPP TR 38.901 V16.0.0, pp 30, Table 7.4.2-1
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_InF_LOS', '3GPP_38.901_InF_NLOS_DL'};
                % Determine the LOS probability for each BS-MT
                r = 0.6; % clutter density (fix at 60% or use a random value greater than oe equal 40%) *38.901 table 7.2-4*
                if ~exist('r','var')
                    r = rand(1,1);
                    while (r< 0.4)
                        r = rand(1,1);
                    end
                end
                
                d_clutter = 2; % clutter size *38.901 table 7.2-4*
                ksub = -d_clutter/log(1-r);
                p_LOS = ones( size( dist_2d ));
                p_LOS = exp(-dist_2d./ksub);
                i_LOS = ( randR >= p_LOS ) + 1;
                
            case '3GPP_38.901_InF_SH'
                % See: 3GPP TR 38.901 V16.0.0, pp 30, Table 7.4.2-1
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_InF_LOS', '3GPP_38.901_InF_NLOS_SH'};
                % Determine the LOS probability for each BS-MT
                r = 0.2; % clutter density (fix at 20% or use a random value under 40%) *38.901 table 7.2-4*
                if ~exist('r','var')
                    r = rand(1,1);
                    while (r>= 0.4)
                        r = rand(1,1);
                    end
                end
                
                % Include height-dependency of the user terminals
                h_UT = rx_trackL.positions( 3,segment_index ) + rx_trackL.initial_position(3);
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                
                %h_c = 10*rand(1,1);
                h_c = 2; % clutter height *38.901 table 7.2-4*
                d_clutter = 10; % clutter size *38.901 table 7.2-4*
                ksub = -d_clutter/log(1-r)*((h_tx-h_UT)/(h_c-h_UT));
                p_LOS = ones( size( dist_2d ));
                p_LOS = exp(-dist_2d./ksub);
                i_LOS = ( randR >= p_LOS ) + 1;
                
            case '3GPP_38.901_InF_DH'
                % See: 3GPP TR 38.901 V16.0.0, pp 30, Table 7.4.2-1
                % The corresponding scenario configuration files
                scen = { '3GPP_38.901_InF_LOS', '3GPP_38.901_InF_NLOS_DH'};    
                % Determine the LOS probability for each BS-MT
                r = 0.4; % clutter density (fix at 40% or use a random value greater than or equal 40%) *38.901 table 7.2-4*
                if ~exist('r','var')
                    r = rand(1,1);
                    while (r< 0.4)
                        r = rand(1,1);
                    end
                end
                
                % Include height-dependency of the user terminals
                h_UT = rx_trackL.positions( 3,segment_index ) + rx_trackL.initial_position(3);
                h_tx = reshape( tx_pos_3d(3,rxi,:) , nS, [] ).';
                
                %h_c = 10*rand(1,1);
                h_c = 6; % Clutter height *38.901 table 7.2-4*
                d_clutter = 2; %Clutter size *38.901 table 7.2-4*
                ksub = -d_clutter/log(1-r)*((h_tx-h_UT)/(h_c-h_UT));
                p_LOS = ones( size( dist_2d ));
                p_LOS = exp(-dist_2d./ksub);
                i_LOS = ( randR >= p_LOS ) + 1;
                
            otherwise
                error('QuaDRiGa:qd_layout:set_scenario:scenario_not_supported','Scenario is not supported.')
        end
        
        % Set the indoor scenarios
        if indoor_rx(i_rx)
            i_LOS = i_LOS + 2;
        end
        
        no_segments = numel(segment_index);
        tmp = size( rx_trackL.scenario );
        if tmp(1) == no_tx && tmp(2) == no_segments
            scen_old = rx_trackL.scenario;
        elseif tmp(1) == 1 && tmp(2) == no_segments
            scen_old = rx_trackL.scenario( ones(1,no_tx),: );
        elseif tmp(1) == 1 && tmp(2) == 1
            scen_old = rx_trackL.scenario( ones(no_tx,no_segments ));
        elseif tmp(1) == no_tx && tmp(2) == 1
            scen_old = rx_trackL.scenario( :,ones(1,no_segments ));
        else
            error(['Scenario definition dimension mismatch for Rx ',num2str(rx( i_rx ))]);
        end
        
        scen_current = reshape( scen( i_LOS(:) ) , no_tx,[] );
        
        % Assign scenario to track
        for i_tx = 1 : numel( tx )
            scen_old( tx( i_tx ),: ) = scen_current( tx( i_tx ),: );
        end
        rx_trackL.scenario = scen_old;
        
        % Reassign track handle to layout
        h_layout.Prx_track( 1,rx( i_rx ) ) = rx_trackL;
    end
end
end
