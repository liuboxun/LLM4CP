function [ loss, scale_sf, loss_init, scale_sf_init ] = get_pl( h_builder, evaltrack, alt_plpar, txpos )
%GET_PL Implements various path-loss models
%
% Calling object:
%   Single object
%
% Description:
%   This function implements various path-loss models such as defined by 3GPP 36.873 or 38.901. The
%   parameters of the various models are given in the configuration files in the folder
%   'quadriga_src config'. When a builder object is initialized, the parameters appear in the
%   structure 'qd_builder.plpar'.
%
% Input:
%   evaltrack
%   A 'qd_track' object for which the PL should be calculated. If 'evaltrack' is not given, then
%   the path loss is calculated for each Rx position. Otherwise the path loss is calculated for the
%   positions provided in 'evaltrack'.
%
%   alt_plpar
%   An optional alternative plpar which is used instead of 'qd_builder.plpar'.
%
%   txpos
%   The corresponding tx positions. This variable can be provided by: 
%   (1) A 'qd_track' object having the same number of snapshots as the 'evaltrack'; 
%   (2) A 3-element vector containing the fixed x,y, and z coordinate of the tx; 
%   (3) A [3 x N] matrix containing one tx position for each position on the 'evaltrack'; 
%   (4) The variable is unprovided or empty. In this case, the tx-positions are obtained from the
%       'qd_builder' object. 
%
% Output:
%   loss
%   The path loss in [dB]
%
%   scale_sf
%   In some scenarios, the SF might change with distance between Tx and Rx. Hence, the shadow
%   fading provided by the LSFs generator has to be changed accordingly. The second output
%   parameter "scale_sf" can be used for scaling the (logarithmic) SF value.
%
%   loss_init
%   The path loss in [dB] for the initial position of the "evaltrack" object. Returns an empty array
%   if no "evaltrack" is given. 
%
%   scale_sf_init
%   "scale_sf" for the initial position of the "evaltrack" object. Returns an empty array
%   if no "evaltrack" is given. 
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

if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "get_pl" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

% Get the number of frequencies
CenterFrequency = h_builder.simpar(1,1).center_frequency' / 1e9;    % in GHz
nF = numel( CenterFrequency );
oF = ones( nF,1 );

% Get the rx positions (either from evaltrack of from the builder)
if ~exist( 'evaltrack','var' ) || isempty(evaltrack)
    evaltrack = [];
    rxpos = h_builder.rx_positions;
    o2i_loss = zeros(nF,1);
    d_3d_in = 0;
    nP = size( rxpos, 2 );
    oP = ones(1,nP);
    use_track = false;
    
elseif isa(evaltrack, 'qd_track')
    
    % Calculate the rx-position for each snapshot on the evaltrack
    initial_position = evaltrack.initial_position;
    rxpos = evaltrack.positions;
    rxpos = rxpos + initial_position * ones(1,size(rxpos,2));
    
    % Add the initial position to the PL calculate as well
    rxpos = [ rxpos, initial_position ];
    
    % There might be a O2I penetration loss and a Indoor distance
    trk_par = evaltrack.par;
    if isempty( trk_par ) || ~isfield( trk_par,'o2i_loss' ) || isempty( trk_par.o2i_loss )
        o2i_loss = zeros(nF,1);
        d_3d_in = 0;
    else
        if evaltrack.no_segments == 1
            o2i_loss  = trk_par.o2i_loss(:,1,:);
            d_3d_in = trk_par.o2i_d3din(:,1);
        elseif evaltrack.no_segments == 2
            o2i_loss = trk_par.o2i_loss(:,2,:);
            d_3d_in = trk_par.o2i_d3din(:,2);
        else
            error('QuaDRiGa:qd_builder:get_pl','Too many segments in the evaltrack');
        end
        if size( o2i_loss,1 ) ~= 1 || size( d_3d_in,1 ) ~= 1
            error('QuaDRiGa:qd_builder:get_pl','O2I-loss is defined for more than on BS.');
        end
        if size( o2i_loss,3 ) ~= nF
            error('QuaDRiGa:qd_builder:get_pl','O2I-loss must be given for each frequency.');
        end
        o2i_loss = permute( o2i_loss,[3,1,2] );
    end
    
    nP = size( rxpos,2 );
    oP = ones(1,nP);
    use_track = true;
  
else % We a 3xN vector of positions
    rxpos = evaltrack;
    evaltrack = [];
    o2i_loss = zeros(nF,1);
    d_3d_in = 0;
    nP = size( rxpos, 2 );
    oP = ones(1,nP);
    use_track = false;
end

% Use the alternative plpar, if given
if exist( 'alt_plpar','var' ) && ~isempty(alt_plpar)
    par = alt_plpar;
else
    par = h_builder.plpar;
end

% Get the tx_position(s)
if ~exist('txpos','var') || isempty( txpos )
    if all( size( h_builder.tx_position ) == [3,1] )
        txpos = repmat( h_builder.tx_position,1,nP );
    elseif any( size( h_builder.tx_position ) ~= [3,nP] )
        error('QuaDRiGa:qd_builder:get_pl','Tx-position is ambiguous');
    else
        txpos = h_builder.tx_position;
    end
    
elseif isa( txpos, 'qd_track')
    if txpos.no_snapshots == 1 && nP > 2
        txpos = txpos.initial_position(:,oP);
        
    elseif txpos.no_snapshots ~= nP-1
        error('QuaDRiGa:qd_builder:get_pl',...
            'Number of positions in tx track does not match number of positions in evaltrack.');
    else
        initial_position = txpos.initial_position;
        txpos = txpos.positions;
        txpos = [txpos + initial_position * oP(1:end-1) , initial_position];
    end
    
elseif nP > 1 && size( txpos,2 ) == 1
    txpos = txpos(:,oP);
    
elseif any( size( txpos ) ~= [3,nP] )
     error('QuaDRiGa:qd_builder:get_pl','Tx-position is ambiguous.');
end

% Calculate the distance between Tx and Rx
d_3d = oF * sqrt( sum( (rxpos - txpos).^2 , 1 ) );

% Calculate the 2D distance
d_2d = oF * sqrt( sum( (rxpos([1,2],:) - txpos([1,2],:)).^2 , 1 ) );

if 1
    % percentage of the 3D distance that is indoor
    p_indoor = d_3d_in ./ d_3d;
    
    % Subtract the 3D indoor distance from the 3D distance to get the 3D outdoor distance
    d_3d = d_3d - d_3d_in;
    
    % Scale the 2D distance to obtain the 2D outdoor distance
    d_2d = d_2d .* (1-p_indoor);
end

% Set a minimum distance of 0.1 m to avoid artifacts
d_3d( d_3d < 0.1 ) = 0.1;
d_2d( d_2d < 0.1 ) = 0.1;

% Initialize output variables
scenpar     = h_builder.scenpar;
loss        = 0 .* (oF * oP);
sf_sigma    = oF * oP * scenpar.SF_sigma + scenpar.SF_delta * log10( CenterFrequency ) * oP;

% This implements the path loss models
if isfield( par , 'model' )
    switch par.model
        
        case 'logdist'
            loss = par.A * log10(d_3d) + par.B + par.C * log10(CenterFrequency) * oP;
            if isfield( par , 'SF' )
                sf_sigma(:) = par.SF;
            end
            
        case 'logdist_simple'
            loss = par.A * log10(d_3d) + par.B;
            
        case 'constant'
            loss = par.A * oF * oP;
            
        case 'winner_los'
            % From WINNER+ D5.3 pp. 74
            
            hBS = txpos(3,:);
            hMS = rxpos(3,:);
            
            hBS( hBS < 1.5  ) = 1.5;
            hMS( hMS < 1.5  ) = 1.5;
            
            % Calculate the breakpoint
            G = par.B1 + par.C1*log10(CenterFrequency) + par.D1*log10(mean(hBS)) + par.E1*log10(mean(hMS));
            H = par.B2 + par.C2*log10(CenterFrequency) + par.D2*log10(mean(hBS)) + par.E2*log10(mean(hMS));
            bp = 10.^( (H-G)./( par.A1-par.A2 ) );
            bp = bp * oP;
            
            hMS = oF * hMS;
            hBS = oF * hBS;
            
            ind = d_2d<=bp;
            if any( ind(:) )
                freq_dep = par.C1*log10(CenterFrequency)*oP;
                loss(ind) = par.A1*log10(d_2d(ind)) + par.B1 + freq_dep(ind)...
                    + par.D1*log10(hBS(ind)) + par.E1*log10(hMS(ind)) + par.F1*hMS(ind);
                sf_sigma(ind) = par.sig1;
            end
            
            ind = ~ind;
            if any( ind(:) )
                freq_dep = par.C1*log10(CenterFrequency)*oP;
                loss(ind) = par.A2*log10(d_2d(ind)) + par.B2 + freq_dep(ind)...
                    + par.D2*log10(hBS(ind)) + par.E2*log10(hMS(ind)) + par.F2*hMS(ind);
                sf_sigma(ind) = par.sig2;
            end
            
        case 'winner_nlos'
            % From WINNER+ D5.3 pp. 74
            
            hBS = txpos(3,:);
            hMS = rxpos(3,:);
            
            hBS( hBS < 1.5  ) = 1.5;
            hMS( hMS < 1.5  ) = 1.5;
            
            loss1 = oF*( par.A1 + par.Ah1 * log10( hBS )).*log10(d_2d) + par.B1 + ...
                par.C1*log10(CenterFrequency)*oP + ...
                oF * par.D1*log10(hBS) + ...
                oF * par.E1*log10(hMS) + ...
                oF * par.F1*hMS;
            
            loss2 = oF*( par.A2 + par.Ah2 * log10( hBS )).*log10(d_2d) + par.B2 + ...
                par.C2*log10(CenterFrequency)*oP + ...
                oF * par.D2*log10(hBS) + ...
                oF * par.E2*log10(hMS) + ...
                oF * par.F2*hMS;
            
            loss3 = oF*( par.A3 + par.Ah3 * log10( hBS )).*log10(d_2d) + par.B3 + ...
                par.C3*log10(CenterFrequency)*oP + ...
                oF * par.D3*log10(hBS) + ...
                oF * par.E3*log10(hMS) + ...
                oF * par.F3*hMS;
            
            i1 = CenterFrequency < 1.5;
            i2 = CenterFrequency >= 1.5 & CenterFrequency < 2;
            i3 = CenterFrequency >= 2;
            
            loss = i1 * oP .* loss1 + i2 * oP .* loss2 + i3 * oP .* loss3;
            
        case 'winner_pathloss'
            % See WINNER II D1.1.2 V1.2 (2007-09) p43 Equation (4.23)
            % PL/[dB] = A log10(d/[m]) + B + C log10(fc/[GHz]/5) + X
            
            loss = par.A * log10(d_3d) + par.B +...
                par.C * log10(CenterFrequency/5)*oP + par.X;
            
            if isfield( par , 'SF' )
                sf_sigma(:) = par.SF;
            end
            
        case 'dual_slope'
            
            % Set defaults
            if ~isfield( par,'A1' );    par.A1 = par.A;         end
            if ~isfield( par,'A2' );    par.A2 = par.A1;        end
            if ~isfield( par,'C' );     par.C = 0;              end
            if ~isfield( par,'D' );     par.D = 0;              end
            if ~isfield( par,'hE' );    par.hE = 0;             end
            
            hBS = txpos(3,:);           % BS height
            hBS(hBS < 1.1*par.hE) = 1.1*par.hE;
            hBS(hBS < 0.1) = 0.1;
            
            hMS = rxpos(3,:);           % MS height
            hMS(hMS < 1.1*par.hE) = 1.1*par.hE;
            hMS(hMS < 0.1) = 0.1;
            
            % Breakpoint Distance
            dBP    = par.E * CenterFrequency * ((hBS - par.hE) .* (hMS - par.hE));
            dBP_3D = sqrt( dBP.^2 + repmat( (hBS-hMS).^2, nF, 1 ) );
            
            % Path loss
            loss     = par.A1*log10( d_3d )   + par.B + par.C*log10( CenterFrequency )*oP + par.D*d_3d;
            loss_dBP = par.A1*log10( dBP_3D ) + par.B + par.C*log10( CenterFrequency )*oP + par.D*dBP_3D;
            loss_2   = loss_dBP + par.A2*log10( d_3d ./ dBP_3D );
            loss( d_2d>dBP ) = loss_2( d_2d>dBP );
            
            if isfield( par,'sig1' )
                sf_sigma( d_2d<=dBP ) = par.sig1;
            end
            if isfield( par,'sig2' )
                sf_sigma( d_2d>dBP )  = par.sig2;
            end
            
        case 'tripple_slope'
            
            % Set defaults
            if ~isfield( par,'A1' );    par.A1 = par.A;         end
            if ~isfield( par,'A2' );    par.A2 = par.A1;        end
            if ~isfield( par,'A3' );    par.A3 = par.A2;        end
            if ~isfield( par,'C' );     par.C = 0;              end
            if ~isfield( par,'D' );     par.D = 0;              end
            if ~isfield( par,'E1' );    par.E1 = par.E;         end
            if ~isfield( par,'E2' );    par.E2 = par.E1;        end
            if ~isfield( par,'hE1' );   par.hE1 = 0;            end
            if ~isfield( par,'hE2' );   par.hE2 = 0;            end
            
            hBS = txpos(3,:);           % BS height
            hBS(hBS < 1.1*par.hE1) = 1.1*par.hE1;
            hBS(hBS < 1.1*par.hE2) = 1.1*par.hE2;
            hBS(hBS < 0.1) = 0.1;
            
            hMS = rxpos(3,:);           % MS height
            hMS(hMS < 1.1*par.hE1) = 1.1*par.hE1;
            hMS(hMS < 1.1*par.hE2) = 1.1*par.hE2;
            hMS(hMS < 0.1) = 0.1;
            
            % Breakpoint Distance (2D)
            dBP1 = par.E1 * CenterFrequency * ((hBS - par.hE1) .* (hMS - par.hE1));
            dBP2 = par.E2 * CenterFrequency * ((hBS - par.hE2) .* (hMS - par.hE2));
            dBP1( dBP2<dBP1 ) = dBP2( dBP2<dBP1 );
            
            % Path length at BP distance (3D)
            dBP1_3D = sqrt( dBP1.^2 + repmat( (hBS-hMS).^2, nF, 1 ) );
            dBP2_3D = sqrt( dBP2.^2 + repmat( (hBS-hMS).^2, nF, 1 ) );
            
            % Path Loss at BP distance (dB)
            loss_dBP1 = par.A1*log10( dBP1_3D ) + par.B + par.C*log10( CenterFrequency )*oP + par.D*dBP1_3D;
            loss_dBP2 = loss_dBP1 + par.A2*log10( dBP2_3D ./ dBP1_3D );
            
            % Path loss
            loss     = par.A1*log10( d_3d ) + par.B + par.C*log10( CenterFrequency )*oP + par.D*d_3d;
            loss_2   = loss_dBP1 + par.A2*log10( d_3d ./ dBP1_3D );
            loss_3   = loss_dBP2 + par.A3*log10( d_3d ./ dBP2_3D );
            
            loss( d_2d>dBP1 & d_2d<=dBP2 ) = loss_2( d_2d>dBP1 & d_2d<=dBP2 );
            loss( d_2d>dBP2 ) = loss_3( d_2d>dBP2 );
            
            % Scaling of the SF
            if isfield( par,'sig1' )
                sf_sigma( d_2d<=dBP1 ) = par.sig1;
            end
            if isfield( par,'sig2' )
                sf_sigma( d_2d>dBP1 & d_2d<=dBP2 ) = par.sig2;
            end
            if isfield( par,'sig3' )
                sf_sigma( d_2d>dBP2 ) = par.sig3;
            end
            
        case 'nlos'
            
            %	PLn =   A * log10( d3d )
            %		 +  B
            %		 +  C * log10( fc )
            %		 +  D * log10( hBS + Dx )
            %		 + D1 * log10( hBS ) / hBS
            %		 + D2 * log10( hBS ) / hBS^2
            %		 + D3 * hBS
            %		 +  E * log10( hUT )
            %		 + E1 * log10( hUT ) / hUT
            %		 + E2 * log10( hUT ) / hUT^2
            %        + E3 * hUT
            %		 +  F * log10( hBS ) * log10( d3d )
            %		 + G1 * log10^2( G2 * hUT )
            
            % Set defaults
            if ~isfield( par,'Cn' );    par.Cn = 0;             end
            if ~isfield( par,'Dn' );    par.Dn = 0;             end
            if ~isfield( par,'D1n' );   par.D1n = 0;            end
            if ~isfield( par,'D2n' );   par.D2n = 0;            end
            if ~isfield( par,'D3n' );   par.D3n = 0;            end
            if ~isfield( par,'En' );    par.En = 0;             end
            if ~isfield( par,'E1n' );   par.E1n = 0;            end
            if ~isfield( par,'E2n' );   par.E2n = 0;            end
            if ~isfield( par,'E3n' );   par.E3n = 0;            end
            if ~isfield( par,'Fn' );    par.Fn = 0;             end
            if ~isfield( par,'G1n' );   par.G1n = 0;            end
            if ~isfield( par,'G2n' );   par.G2n = 1;            end
            
            % Get values from dual-slope LOS model
            par.model = 'dual_slope';
            [ tmp1, ~, tmp2 ] = h_builder.get_pl( evaltrack, par, txpos );
            loss_1 = [ tmp1, tmp2 ];
            
            hBS = txpos(3,:);           % BS height
            hBS(hBS < 0.1) = 0.1;
            hBS = oF * hBS;
            
            hMS = rxpos(3,:);           % MS height
            hMS(hMS < 0.1) = 0.1;
            hMS = oF * hMS;
            
            % NLOS model
            loss = par.An * log10(d_3d) + par.Bn + par.Cn * log10(CenterFrequency) * oP ...
                + par.Dn * log10(hBS) + par.D1n * (log10(hBS)./hBS) + par.D2n * (log10(hBS)./hBS.^2) + par.D3n * hBS ...
                + par.En * log10(hMS) + par.E1n * log10(hMS)./hMS + par.E2n * log10(hMS)./hMS.^2 + par.E3n * hMS...
                + par.Fn * log10(hBS) .* log10(d_3d) ...
                + par.G1n * ( log10( par.G2n * hMS ) ).^2;
            
            loss( loss_1>loss ) = loss_1( loss_1>loss );
            
            if isfield( par,'sig' )
                sf_sigma(:) = par.sig;
            end
            
        case 'satellite'
            
            % Set defaults
            if ~isfield( par,'A' );   par.A = 20;            end
            if ~isfield( par,'B' );   par.B = 32.45;         end
            if ~isfield( par,'C' );   par.C = 20;            end
            if ~isfield( par,'D' );   par.D = 0;             end
            if ~isfield( par,'usePLa' ); par.usePLa = 1;     end
            
            % Atmospheric attenuation at zenith angle
            % Frequency values
            zenith_att(1,:) = [ 1, 1.99, 2.97, 3.96, 4.95, 5.92, 6.87, 7.87, 8.83, 9.77, 11.03, 12.32, 13.17, 14.44,...
                15.65, 16.37, 17.18, 18.03, 18.8, 19.34, 19.91, 20.42, 20.71, 21.16, 21.49, 21.7, 22.09, 22.52, 22.99,...
                23.98, 24.93, 26.13, 27.38, 29.18, 30.44, 32.18, 33.8, 36.09, 38.87, 40.59, 42.33, 44, 44.96, 47.2,...
                48.42, 49.42, 50.17, 50.96, 51.73, 52.62, 53.32, 53.6, 53.75, 54.06, 54.38, 55.12, 55.73, 56.54,...
                56.97, 57.52, 58.01, 58.18, 58.33, 58.62, 58.99, 59.38, 59.57, 59.84, 60.27, 60.59, 60.93, 61.22,...
                61.44, 61.93, 62.27, 63.34, 64.27, 65.42, 66.35, 67.19, 68.16, 69.22, 70.33, 71.66, 73.53, 75.28,...
                77.48, 80.64, 83.77, 87.07, 92.1, 96.39, 100];
            % Attenuation values
            zenith_att(2,:) = [ 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08,...
                0.09, 0.11, 0.13, 0.16, 0.19, 0.23, 0.28, 0.35, 0.4, 0.46, 0.51, 0.52, 0.52, 0.49, 0.44, 0.36, 0.3,...
                0.26, 0.24, 0.23, 0.24, 0.25, 0.27, 0.32, 0.4, 0.46, 0.56, 0.68, 0.8, 1.16, 1.53, 2.18, 2.99, 4.47,...
                8.08, 15.49, 30.37, 39.94, 47.33, 60.66, 80.97, 102.6, 120.55, 137.92, 144.01, 149.57, 153.68, 155.18,...
                156.63, 157.9, 158.81, 157.57, 156.04, 151.64, 141.33, 130.52, 119.21, 97.42, 68, 52.74, 39.81, 17.77,...
                8.61, 4.2, 2.8, 2.19, 1.69, 1.42, 1.22, 1.07, 0.93, 0.85, 0.79, 0.75, 0.74, 0.75, 0.81, 0.87, 0.97];
            
            % PL = A * log10( d3D[m] ) + B + C * log10( f[GHz] ) + D * log10( alpha_rad ) + PLa
            
            % Elevation angle in [rad]
            alpha = asin( txpos(3,:)./d_3d );
            
            % Calculate atmospheric attenuation
            if par.usePLa
               PLa = qf.interp( zenith_att(1,:), 0, zenith_att(2,:), CenterFrequency ) ./ sin(alpha);
            else
               PLa = zeros( 1,nP );                
            end            
            
            % Calculate path-loss
            loss = par.A * log10(d_3d) + par.B + par.C * log10(CenterFrequency) * oP +...
                par.D * log10( alpha ) + PLa;
            
        otherwise
            error('??? PL model not defined in qd_parameter_set.get_pl')
    end
end

% Add outdoor-to-indoor penetration loss
loss = loss + o2i_loss * oP;

% The SF cannot change within a segment (sudden power changes)
if use_track
    sf_sigma = mean(sf_sigma,2) * oP;
end

% The shadow fading might change with distance. Hence, if
% the value did change, we have to rescale the values from
% the map.
SF_sigma_scenpar = scenpar.SF_sigma + scenpar.SF_delta * log10( CenterFrequency );
if any( sf_sigma(:) ~= 0 ) && all( SF_sigma_scenpar ~= 0 )
    scale_sf = sf_sigma ./ (SF_sigma_scenpar * oP);
else
    scale_sf = ones( size( sf_sigma ) );
end

% Return results for track positions and initial position separately
if use_track
    loss_init       = loss(:,end);
    scale_sf_init   = scale_sf(:,end);
    loss            = loss(:,1:end-1);
    scale_sf        = scale_sf(:,1:end-1);
else
    loss_init       = [];
    scale_sf_init   = [];
end

end
