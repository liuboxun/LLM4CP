function out = copy( obj, in, copy_simpar, copy_tx_array, copy_rx_array, copy_sos )
%COPY Creates a copy of the handle class object or array of objects
%
% Calling object:
%   Object array
%
% Description:
%   While the standard copy command creates new physical objects for each element of the object
%   array (in case obj is an array of object handles), copy checks whether there are object handles
%   pointing to the same object and keeps this information.
%
% Output:
%   out
%   Copy of the current object or object array
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

if ~exist( 'in','var' ) 
    
    sic = size( obj );
    prc = zeros( sic ); % Processed elements
    out = obj; % Placeholder
   
    i_tx = 0;
    i_rx = 0;
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2 ] = qf.qind2sub( sic, n );
            
            copy_simpar = true;
            copy_sos = true;
            if n ~= 1
                prev_bld = find( prc(:) == 1 );
                for m = 1 : numel( prev_bld )
                    [ j1,j2 ] = qf.qind2sub( sic, prev_bld(m) );
                    if copy_simpar && ~isempty( obj(i1,i2).simpar ) && ~isempty( obj(j1,j2).simpar ) && qf.eqo( obj(i1,i2).simpar, obj(j1,j2).simpar )
                        copy_simpar = false;
                        h_simpar = out(j1,j2).simpar;
                    end
                    if copy_sos && ~isempty( obj(i1,i2).sos ) && ~isempty( obj(j1,j2).sos ) && qf.eqo( obj(i1,i2).sos(1,1), obj(j1,j2).sos(1,1) )
                        copy_sos = false;
                        h_sos = [j1,j2];
                    end
                end
            end
            
            out( i1,i2 ) = qd_builder( [] );   % Create empty object
            copy( out(i1,i2), obj(i1,i2), copy_simpar, 0,0,copy_sos ); % Copy content
            prc( i1,i2 ) = 1;
            
            if ~copy_simpar
                out(i1,i2).simpar = h_simpar;
            end
            if ~copy_sos
                j1 = h_sos(1);
                j2 = h_sos(2);
                out(i1,i2).sos = out(j1,j2).sos;
                out(i1,i2).path_sos = out(j1,j2).path_sos;
                out(i1,i2).xpr_sos = out(j1,j2).xpr_sos;
                out(i1,i2).pin_sos = out(j1,j2).pin_sos;
                out(i1,i2).clst_dl_sos = out(j1,j2).clst_dl_sos;
                out(i1,i2).gr_sos = out(j1,j2).gr_sos;
                out(i1,i2).absTOA_sos = out(j1,j2).absTOA_sos;
            end
            
            m = qf.eqo( obj(i1,i2), obj ); % Determine equal handles
            m(i1,i2) = false; % Remove own handle

            if any( m(:) )
                out( m ) = out( i1,i2 ); % Copy references
                prc( m ) = 2;
            end
            
            % Assemble transmit antenna arrays into a vector of handles
            n_tx = numel(obj(i1,i2).tx_array);
            if i_tx == 0 && n_tx > 0
                tx_array = qf.reshapeo( obj(i1,i2).tx_array, [1,n_tx] );
                i_tx = n_tx;
            elseif n_tx > 0
                tx_array( i_tx+1 : i_tx+n_tx ) = qf.reshapeo( obj(i1,i2).tx_array, [1,n_tx] );
                i_tx = i_tx + n_tx;
            else
                tx_array = qd_arrayant([]);
            end
            
            % Assemble receive antenna arrays into a vector of handles
            n_rx = numel(obj(i1,i2).rx_array);
            if i_rx == 0 && n_rx > 0
                rx_array = qf.reshapeo( obj(i1,i2).rx_array, [1,n_rx] );
                i_rx = n_rx;
            elseif n_rx > 0
                rx_array( i_rx+1 : i_rx+n_rx ) = qf.reshapeo( obj(i1,i2).rx_array, [1,n_rx] );
                i_rx = i_rx + n_rx;
            else
                rx_array = qd_arrayant([]);
            end
        end
    end
    
    % Copy the array antennas (this copies all the handles correctly)
    if ~isempty( tx_array )
        tx_array = copy( tx_array );
    end
    if ~isempty( rx_array )
        rx_array = copy( rx_array );
    end
    
    % Assign copied array antennas to output builder array
    i_tx = 0;
    i_rx = 0;
    for n = 1 : prod( sic )
        if prc( n ) == 1
            [ i1,i2 ] = qf.qind2sub( sic, n );
            n_tx = numel(obj(i1,i2).tx_array);
            if n_tx > 0
                out(i1,i2).tx_array = qf.reshapeo( tx_array(1,i_tx+1:i_tx+n_tx), size(obj(i1,i2).tx_array) );
                i_tx = i_tx + n_tx;
            end
            n_rx = numel(obj(i1,i2).rx_array);
            if n_rx > 0
                out(i1,i2).rx_array = qf.reshapeo( rx_array(1,i_rx+1:i_rx+n_rx), size(obj(i1,i2).rx_array) );
                i_rx = i_rx + n_rx;
            end
        end
    end
    
    % Workaround for octave
    if numel( obj ) == 1
        out = out(1,1);
    end
    
else
    % The list of properties that need to be copied
    prop = {'name','plpar','tx_position','rx_positions','ds','kf','sf','asD','asA','esD','esA','xpr',...
        'gr_epsilon_r','absTOA_offset','NumClusters','NumSubPaths','taus','gain','AoD','AoA','EoD','EoA',...
        'xprmat','pin','subpath_coupling','fbs_pos','lbs_pos','dual_mobility','Pscenario','Pscenpar','pow_wo_kf'};
    
    % Empty outout
    out = [];
    
    % Copy the data
    for n = 1 : numel(prop)
        obj.( prop{n} ) = in.( prop{n} );
    end
    
    % Copy objects of other classes
    if copy_simpar
        obj.simpar =  copy( in.simpar );
    end
    if ~isempty( in.tx_array ) && copy_tx_array
        obj.tx_array = copy( in.tx_array );
    end
    if ~isempty( in.rx_array ) && copy_rx_array
        obj.rx_array = copy( in.rx_array );
    end
    if ~isempty( in.tx_track )
        obj.tx_track = copy( in.tx_track );
    end
    if ~isempty( in.rx_track )
        obj.rx_track = copy( in.rx_track );
    end
    
    % SOS Generators
    if copy_sos
        prop = {'sos','path_sos','xpr_sos','pin_sos','clst_dl_sos','gr_sos','absTOA_sos'};
        for m = 1 : numel(prop)
            if ~isempty( in.( prop{m} ) )
                sic = size( in.( prop{m} ) );                   % Size of SOS object array
                obj.( prop{m} ) = in.( prop{m} );               % Placeholder - copy handles
                for n = 1 : prod( sic )
                    [ i1,i2 ] = qf.qind2sub( sic, n );          % Determine indices
                    obj.( prop{m} )( i1,i2 ) = qd_sos( [] );    % Create empty object
                    copy( obj.( prop{m} )( i1,i2 ), in.( prop{m} )( i1,i2 ) ); % Copy content
                end
            end
        end
    end
end

end
