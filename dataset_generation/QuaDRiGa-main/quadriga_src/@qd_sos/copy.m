function out = copy( obj, in )
%COPY Creates a copy of the SOS object
%
% While the standard copy command creates new physical objects for each
% element of the object array (in case obj is an array of object handles), copy checks
% whether there are object handles pointing to the same object and keeps
% this information.
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

if ~exist( 'in','var' ) 

    sic = size( obj );
    prc = false( sic ); % Processed elements
    out = obj; % Placeholder
    
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            out( i1,i2,i3,i4 ) = qd_sos( [] );  % Create empty object
            copy( out(i1,i2,i3,i4), obj(i1,i2,i3,i4) ); % Copy content
            prc( i1,i2,i3,i4 ) = true;
            
            m = qf.eqo( obj(i1,i2,i3,i4), obj ); % Determine equal handles
            m(i1,i2,i3,i4) = false; % Remove own handle

            if any( m(:) )
                out( m ) = out( i1,i2,i3,i4 ); % Copy references
                prc( m ) = true;
            end
        end
    end
    
    % Workaround for octave
    if numel( obj ) == 1
        out = out(1,1);
    end
    
else
    % The list of properties that need to be copied
    prop = {'name','distribution','Pdist_decorr','dist','acf',...
        'sos_freq','sos_phase','sos_amp'};
    
    % Empty outout
    out = [];
    
    % Copy the data
    for n = 1 : numel(prop)
        obj.( prop{n} ) = in.( prop{n} );
    end
end

end
