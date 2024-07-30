function h_channel = get_los_channels( h_builder, precision, return_coeff, tx_array_mask )
%GET_LOS_CHANNELS Generates channel coefficients for the LOS path only. 
%
% Calling object:
%   Object array
%
% Description:
%   This function generates static coefficients for the LOS path only. This includes the following
%   properties:
%      * antenna patterns
%      * orientation of the Rx (if provided)
%      * polarization rotation for the LOS path
%      * plane-wave approximation of the phase
%      * path loss
%      * shadow fading
%
%    No further features of QuaDRiGa are used (i.e. no drifting, spherical waves, time evolution,
%    multipath fading etc.). This function can thus be used to acquire quick previews of the
%    propagation conditions for a given layout.
%
% Input:
%   precision
%   The additional input parameter 'precision' can be used to set the numeric precision to
%   'single', thus reducing the memory requirements for certain computations. Default: double
%   precision.
%
%   return_coeff
%   Adjusts the format of the output. This is mainly used internally.
%   
%   return_coeff = 'coeff'; 
%   If this is set to 'coeff', only the raw channel coefficients are returned, but no QuaDRiGa
%   channel object is created. This may help to reduce memory requirements. In this case, 'h_channel'
%   has the dimensions: [ n_rx, n_tx, n_pos ] 
%
%   return_coeff = 'raw'; 
%   Same as 'coeff' but without applying the distance-dependent phase and the path loss. The
%   rx-antenna is assumed to be dual-polarized with two elements (i.e. the rx interpolation is
%   omitted). This mode is used QuaDRiGa-internally by [qd_arrayant.combine_pattern]
%
%   tx_array_mask
%   Indices for selected transmit antenna elements
%
% Output:
%   h_channel
%   A 'qd_channel' object. The output contains one coefficient for each position in
%   'qd_builder.rx_position'.
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

single_precision = true;
if ~exist('precision','var') || ~strcmp( precision , 'single' )
    precision = 'double';
    single_precision = false;
end

if exist('return_coeff','var') && ~isempty( return_coeff )
    switch return_coeff
        case 'coeff'
            return_coeff = true;
            raw_coeff = false;
        case 'raw'
            return_coeff = true;
            raw_coeff = true;
        otherwise
            return_coeff = false;
            raw_coeff = false;
    end
else
    return_coeff = false;
    raw_coeff = false;
end

if return_coeff && numel( h_builder ) ~= 1
    error('Raw channel coefficients can only be generated for scalar builder objects.')
end

if ~exist( 'tx_array_mask','var' )
    tx_array_mask = [];
end

if numel( h_builder ) > 1
    
    sic = size( h_builder );
    h_channel = qd_channel;
    for i_cb = 1 : numel( h_builder )
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_cb );
        if h_builder( i1,i2,i3,i4 ).no_rx_positions > 0
            h_channel( i1,i2,i3,i4 ) = get_los_channels( h_builder( i1,i2,i3,i4 ), precision, [], tx_array_mask );
        end
    end
    
else
    % Fix for Octave (conversion from object-array to single object)
    h_builder = h_builder(1,1);
    
    % Test if there is only one Rx-array in the builder
    if numel( h_builder.rx_array ) > 1 && any( ~qf.eqo( h_builder.rx_array(1,1) , h_builder.rx_array ) )
        error('QuaDRiGa:qf_builder:get_los_channels:Rx_array_ambiguous',...
            ['There is more than one Rx array antenna in "h_builder".\n ',...
            '"get_los_channels" can only use one array antenna.']);
    elseif isempty( h_builder.rx_array )
        rx_array = qd_arrayant('omni');
    else
        rx_array = h_builder.rx_array(1,1);
    end
    
    % Test if there is only one Tx-array in the builder
    if numel( h_builder.tx_array ) > 1 && any( ~qf.eqo( h_builder.tx_array(1,1) , h_builder.tx_array ) )
        error('QuaDRiGa:qf_builder:get_los_channels:Tx_array_ambiguous',...
            ['There is more than one Tx array antenna in "h_builder".\n ',...
            '"get_los_channels" can only use one array antenna.']);
    elseif isempty( h_builder.tx_array )
        tx_array = qd_arrayant('omni');
    else
        tx_array = h_builder.tx_array(1,1);
    end
    
    % Test if there is only a single tx track in the builder
    single_tx_track = false;
    if numel( h_builder.tx_track ) == 0 || numel( h_builder.tx_track ) == 1
        single_tx_track = true;
    end
    
    % Test if there is only a single rx track in the builder
    single_rx_track = false;
    if numel( h_builder.rx_track ) == 0 || numel( h_builder.rx_track ) == 1
        single_rx_track = true;
    end
    
    % Check if the builder is a dual-mobility builder and that the inputs are correctly formatted
    if h_builder.dual_mobility == -1
        h_builder.check_dual_mobility;
    end
    
    % Check if we have a single-frequency builder
    if numel( h_builder.simpar(1,1).center_frequency ) > 1
         error('QuaDRiGa:qd_builder:get_los_channels','get_los_channels only works for single-frequency simulations.');
    end
     
    lambda  = h_builder.simpar(1,1).wavelength;
    n_positions = h_builder.no_rx_positions;
    o_positions = ones( 1,n_positions );
    
    wave_no = 2*pi/lambda;
    if single_precision
        wave_no = single( wave_no );
    end
    
    % Extract the travel directions at the initial position
    rx_orientation = zeros(3,n_positions,precision);
    tx_orientation = zeros(3,n_positions,precision);
    for i_pos = 1 : n_positions
        if i_pos == 1 || ( ~single_rx_track && ~qf.eqo( h_builder.rx_track(1,i_pos),h_builder.rx_track(1,i_pos-1) ) )
            [~,ind] = min( sum( h_builder.rx_track(1,i_pos).positions.^2 ) );
            rx_orientation(:,i_pos) = h_builder.rx_track(1,i_pos).orientation( :,ind );
        else
            rx_orientation(:,i_pos) = rx_orientation(:,i_pos-1);
        end
        if i_pos > 1 && ( single_tx_track || qf.eqo( h_builder.tx_track(1,i_pos),h_builder.tx_track(1,i_pos-1) ) )
            tx_orientation(:,i_pos) = tx_orientation(:,i_pos-1);
        elseif h_builder.tx_track(1,i_pos).no_snapshots > 1
            tx_orientation(:,i_pos) = h_builder.tx_track(1,i_pos).orientation( :,ind );
        else
            tx_orientation(:,i_pos) = h_builder.tx_track(1,i_pos).orientation( :,1 );
        end
    end
    
    % If the orientation values are identical, we can use the scalar values to save computing time
    % in "qf.calc_ant_rotation".
    if all(all( abs( rx_orientation - rx_orientation(:,o_positions) ) < 1e-6 ))
        rx_orientation = rx_orientation(:,1);
    end
    if all(all( abs( tx_orientation - tx_orientation(:,o_positions) ) < 1e-6 ))
        tx_orientation = tx_orientation(:,1);
    end  
    
    % Get the arrival and departure angles
    angles = h_builder.get_angles*pi/180;
    if single_precision
        angles = single( angles );
        rx_orientation = single( rx_orientation );
        tx_orientation = single( tx_orientation );
    end
    
    % Interpolate the transmit antenna patterns
    [ Vt,Ht,Pt ] = interpolate( tx_array, angles(1,:), angles(3,:), tx_array_mask, tx_orientation, 14.3239 );
    n_tx = size(Vt,1);
    
    Vt = reshape( Vt,[1,n_tx,n_positions] );
    Ht = reshape( Ht,[1,n_tx,n_positions] );
    Pt = reshape( Pt,[1,n_tx,n_positions] );
    
    if isempty( tx_array_mask )
        Ct = tx_array.coupling;
    elseif size(tx_array.coupling,2) == size(tx_array.coupling,1)
        Ct = tx_array.coupling(tx_array_mask,tx_array_mask);
    else
        Ct = tx_array.coupling(tx_array_mask,:);
    end
    if single_precision && ~isa(Ct,'single')
        Ct = single(Ct);
    end
    
    if raw_coeff
        
        % Use two Rx elements for the two polarizations
        n_rx = 2;
        
        % It is possible to provide different Jones matrices for the raw coefficients
        if rx_array.no_elements == 2
            Cr = rx_array.coupling.';
        else
            Cr = eye(2);
        end
        if single_precision && ~isa(Cr,'single')
            Cr = single(Cr);
        end
        
        % Calculate the LOS channel coefficients (LOS polarization transfer matrix is [ 1 0 ; 0 -1 ])
        c = repmat([1;0],[1,n_tx,n_positions]) .* repmat(Vt,[n_rx,1,1]) - repmat([0;1],[1,n_tx,n_positions]) .* repmat(Ht,[n_rx,1,1]);
        
        % Apply the per-element phase offset
        c = c.* exp( -1j*(wave_no.*(repmat(Pt,[2,1,1])))) ;
         
    else
        % Interpolate the RX antenna patterns
        [ Vr,Hr,Pr ] = interpolate( rx_array, angles(2,:), angles(4,:), [], rx_orientation, 14.3239 );
        n_rx = rx_array.no_elements;
        
        Vr = reshape( Vr,[n_rx,1,n_positions] );
        Hr = reshape( Hr,[n_rx,1,n_positions] );
        Pr = reshape( Pr,[n_rx,1,n_positions] );
        
        Cr = rx_array.coupling.';
        if single_precision && ~isa(Cr,'single')
            Cr = single(Cr);
        end
        
        % Calculate the LOS channel coefficients (LOS polarization transfer matrix is [ 1 0 ; 0 -1 ])
        c = repmat(Vr,[1,n_tx,1]) .* repmat(Vt,[n_rx,1,1]) - repmat(Hr,[1,n_tx,1]) .* repmat(Ht,[n_rx,1,1]);
        
        % Calculate the phase
        phase = 2*pi/lambda * mod( sqrt(sum((h_builder.rx_positions - h_builder.tx_position).^2)), lambda );
        if single_precision
            phase = single( phase );
        end
        phase = reshape( phase,1,1,n_positions );
        
        % Apply phase to channel coefficients
        c = c.* exp( -1j*(wave_no.*(repmat(Pr,[1,n_tx,1]) + repmat(Pt,[n_rx,1,1])) + repmat(phase,[n_rx,n_tx,1])) );
        
        % Calculate the path loss
        [ path_loss , scale_sf ] = h_builder.get_pl;
        if isempty( h_builder.sf )
            rx_power = -path_loss;
        else
            rx_power = -path_loss + 10*log10( h_builder.sf ) .* scale_sf;
        end
        rx_power = sqrt( 10.^( 0.1 * rx_power ) );
        
        % Apply the path loss
        c = c .* repmat(reshape( rx_power,1,1,n_positions ),[n_rx,n_tx,1]);
    end
        
    % Apply antenna coupling
    if single_precision
        Ct = single( Ct );
        Cr = single( Cr );
    end
    
    if all(size(Ct) == [ n_tx , n_tx ]) && ...
            all(all( abs( Ct - eye(n_tx)) < 1e-10 )) && ...
            all(size(Cr) == [ n_rx , n_rx ]) && ...
            all(all( abs( Cr - eye(n_rx)) < 1e-10 ))
        
        % Both coupling matrices are identity matrices.
        coeff = c;
        
    elseif all(size(Ct) == [ n_tx , n_tx ]) && ...
            all(all( abs( Ct - diag(diag(Ct)) ) < 1e-10 )) && ...
            all(size(Cr) == [ n_rx , n_rx ]) && ...
            all(all( abs( Cr - eye(n_rx)) < 1e-10 ))
        
        % The tx has a diagonal matrix and the rx an identity matrix
        coeff = zeros( n_rx , n_tx , n_positions , precision  );
        for i_tx = 1 : n_tx
            coeff( :,i_tx,: ) = c(:,i_tx,:) .* Ct( i_tx,i_tx );
        end
        
        
    elseif all(size(Cr) == [ n_rx , n_rx ]) && ...
            all(all( abs(Cr - eye(n_rx)) < 1e-10 ))
        
        % Only the Rx is an identity matrix.
        coeff = zeros( n_rx , size(Ct,2) , n_positions , precision  );
        for i_tx_in = 1 : n_tx
            for i_tx_out = 1 : size(Ct,2)
                coeff( :,i_tx_out,: ) = coeff( :,i_tx_out,: ) +...
                    c(:,i_tx_in,:) .* Ct(i_tx_in,i_tx_out );
            end
        end
        
    elseif all(size(Ct) == [ n_tx , n_tx ]) && ...
            all(all( abs(Ct - eye(n_tx)) < 1e-10 ))
        
        % Only the Tx is an identity matrix.
        coeff = zeros( size(Cr,1) , n_tx , n_positions , precision );
        for n = 1:n_positions
            coeff(:,:,n) = Cr * c(:,:,n);
        end
        
    else
        % Both coupling matrices are not identity matrices.
        coeff = zeros( size(Cr,1) , size(Ct,2) , n_positions , precision );
        for n = 1:n_positions
            coeff(:,:,n) = Cr * c(:,:,n) * Ct;
        end
        
    end
    
    if return_coeff
        h_channel = coeff;
    else
        % Create output channel object
        coeff = reshape(coeff,size(coeff,1), size(coeff,2),1,n_positions);
        h_channel = qd_channel( coeff , zeros(1,n_positions,precision) , 1  );
        h_channel.name = h_builder.name;
        h_channel.center_frequency = h_builder.simpar(1,1).center_frequency(1,1);
        h_channel.tx_position = h_builder.tx_position;
        h_channel.rx_position = h_builder.rx_positions;
    end
end

end
