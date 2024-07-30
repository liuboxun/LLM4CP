function xml_write( h_array, fn, pfx, id, fid )
%XML_WRITE Writes antenna patterns into a QDANT XML file
%
% Calling object:
%   Object array
%
% Description:
%   The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna
%   pattern data in XML. The file format specification is described in the documentation. This
%   method saves a qd_arrayant object array to a XML file.
%
% Input:
%   fn
%   Filename of the QDANT XML file.
%
%   pfx
%   String defining the namespace (optional). It is possible to use a prefix to avoid name
%   conflicts when embedding QuaDRiGa antennas in other XML formats. When using a prefix in XML, a
%   namespace for the prefix must be defined.
%
%   id
%   Integer number defining the array antenna ID (optional). If multiple array antennas are stored
%   in the same file, each antenna must be identified by an unique ID.
%
%   fid
%   An integer that identifies an already opened file for subsequent low-level file I/O operations
%   (optional).
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

if ~exist( 'id','var' ) || isempty( id )
    id = 1;
end

if ~exist( 'pfx','var' ) || isempty( pfx )
    pfx = '';
else
    pfx = [pfx,':'];
end

remember_to_close_file = false;
if ~exist( 'fid','var' ) || isempty( fid )
    if ~exist( 'fn','var' ) || isempty( fn )
        error('QuaDRiGa:qd_arrayant:xml_write:filename_not_given',...
            'You did not specify a filename.');
    end
    fid = fopen( fn, 'w');
    fprintf(fid,'<?xml version="1.0" encoding="UTF-8"?>\n');
    if isempty( pfx )
        fprintf(fid,'<qdant xmlns="http://www.quadriga-channel-model.de">\n');
    else
        fprintf(fid,['<qdant xmlns:',pfx(1:end-1),'="http://www.quadriga-channel-model.de">\n']);
    end
    remember_to_close_file = true;
end

if numel( h_array ) > 1
    % If we get an array of of qd_arrayant objects, we check which objects are unique and which
    % objects are duplicate pointers. Only the unique array are written to the file.
    
    % Process array indices
    a = qd_arrayant([]);                                            % Empy array of qd_arrayant objects
    a_cnt = 0;                                                      % Counter
    a_ind = zeros( size( h_array) );                                % Index list for the array antennas
    for ifrq = 1 : size( h_array,1 )
        for iue = 1 : size( h_array,2 )
            t_array = h_array( ifrq,iue );                          % Copy handle
            iseq = qf.eqo( t_array, a );                            % Check if antenna already exists
            if any( iseq )
                a_ind( ifrq,iue ) = find( iseq );                   % Save index
            else
                a_cnt = a_cnt + 1;                                  % Increase counter
                a(1,a_cnt) = t_array;                               % Copy handle
                a_ind( ifrq,iue ) = a_cnt;                          % Save index
            end
        end
    end
    
    % Write layout to file
    fprintf(fid,['<',pfx,'layout>']);
    if size( a_ind,1 ) == 1
        val = sprintf('%u ',a_ind);
        fprintf(fid,val(1:end-1));
    else
        for n = 1 : size( a_ind,2 )
            val = sprintf('%u,',a_ind(:,n));
            if n == size( a_ind,2 )
                fprintf(fid,val(1:end-1));
            else
                fprintf(fid,[val(1:end-1),' ']);
            end
        end
    end
    fprintf(fid,['</',pfx,'layout>\n']);
    
    % Process each antenna object individually
    for n = 1 : a_cnt
        if isempty( pfx )
            xml_write( a(1,n), [], [], n, fid );
        else
            xml_write( a(1,n), [], pfx(1:end-1), n, fid );
        end
    end
    
else
    h_array = h_array(1,1);                                         % Fix for Octave
    no_elements = h_array.no_elements;                              % Save number of elements
    
    % Write data to file    
    fprintf(fid,['<',pfx,'arrayant id="',num2str(id,'%u'),'">\n']);
    fprintf(fid,['\t<',pfx,'name>',h_array.name,'</',pfx,'name>\n']);
    fprintf(fid,['\t<',pfx,'CenterFrequency>%1.14g</',pfx,'CenterFrequency>\n'],h_array.center_frequency);
    fprintf(fid,['\t<',pfx,'NoElements>%1.14g</',pfx,'NoElements>\n'],no_elements);
    
    val = sprintf('%1.6g,%1.6g,%1.6g ',h_array.element_position);
    fprintf(fid,['\t<',pfx,'ElementPosition>',val(1:end-1),'</',pfx,'ElementPosition>\n']);
    
    val = h_array.elevation_grid*180/pi;
    digits = round( max(log10(1/min(diff(val))),0)+3);
    val = sprintf(['%1.',num2str(digits),'g '],val);
    fprintf(fid,['\t<',pfx,'ElevationGrid>',val(1:end-1),'</',pfx,'ElevationGrid>\n']);
    
    val = h_array.azimuth_grid*180/pi;
    digits = round( max(log10(1/min(diff(val))),0)+3);
    val = sprintf(['%1.',num2str(digits),'g '],val);
    fprintf(fid,['\t<',pfx,'AzimuthGrid>',val(1:end-1),'</',pfx,'AzimuthGrid>\n']);
    
    fprintf(fid,['\t<',pfx,'CouplingAbs>']);
    for n = 1 : no_elements
        val = sprintf('%1.6g,',abs(h_array.coupling(:,n)));
        if n == no_elements
            fprintf(fid,val(1:end-1));
        else
            fprintf(fid,[val(1:end-1),' ']);
        end
    end
    fprintf(fid,['</',pfx,'CouplingAbs>\n']);
    fprintf(fid,['\t<',pfx,'CouplingPhase>']);
    for n = 1 : no_elements
        val = sprintf('%1.6g,',angle(h_array.coupling(:,n))*180/pi);
        if n == no_elements
            fprintf(fid,val(1:end-1));
        else
            fprintf(fid,[val(1:end-1),' ']);
        end
    end
    fprintf(fid,['</',pfx,'CouplingPhase>\n']);
    
    nel = h_array.no_el;                                            % Number of elevation angles
    for n = 1 : no_elements
        
        val = 10*log10( abs( h_array.Fa(:,:,n) ).^2 );
        if ~all( val(:) < -99 )                                     % Pattern is 0
            val(val<-99) = -99;                                     % Avoid -Inf
            val(val>99) = 99;                                       % Avoid + Inf
            val( val<0.005 & val > -0.005 ) = 0;                    % Values around 0
            val = round(val*100)/100;                               % Round to 2 digits after comma
            fprintf(fid,['\t<',pfx,'EthetaMag el="',num2str(n),'">\n\t']);
            for m = 1 : nel                                         % Write data
                fprintf(fid,'%1.4g ',val(m,:));
                fprintf(fid,'\n\t');
            end
            fprintf(fid,['</',pfx,'EthetaMag>\n']);
            
            val = angle( h_array.Fa(:,:,n) )*180/pi;                % Read phase
            if ~all( abs(val(:)) < 0.005 )                          % All 0?
                val( val<0.005 & val > -0.005 ) = 0;                % Values around 0
                val( val<-179.995 ) = 180;                          % Values around pi
                val = round(val*100)/100;                           % Round to 2 digits after comma
                fprintf(fid,['\t<',pfx,'EthetaPhase el="',num2str(n),'">\n\t']);
                for m = 1 : nel
                    fprintf(fid,'%1.5g ',val(m,:));
                    fprintf(fid,'\n\t');
                end
                fprintf(fid,['</',pfx,'EthetaPhase>\n']);
            end
        end
        
        val = 10*log10( abs( h_array.Fb(:,:,n) ).^2 );
        if ~all( val(:) < -99 )                                     % Pattern is 0
            val(val<-99) = -99;                                     % Avoid -Inf
            val(val>99) = 99;                                       % Avoid + Inf
            val( val<0.005 & val > -0.005 ) = 0;                    % Values around 0
            val = round(val*100)/100;                               % Round to 2 digits after comma
            fprintf(fid,['\t<',pfx,'EphiMag el="',num2str(n),'">\n\t']);
            for m = 1 : nel                                         % Write data
                fprintf(fid,'%1.4g ',val(m,:));
                fprintf(fid,'\n\t');
            end
            fprintf(fid,['</',pfx,'EphiMag>\n']);
            
            val = angle( h_array.Fb(:,:,n) )*180/pi;                % Read phase
            if ~all( abs(val(:)) < 0.005 )                          % All 0?
                val( val<0.005 & val > -0.005 ) = 0;                % Values around 0
                val( val<-179.995 ) = 180;                          % Values around pi
                val = round(val*100)/100;                           % Round to 2 digits after comma
                fprintf(fid,['\t<',pfx,'EphiPhase el="',num2str(n),'">\n\t']);
                for m = 1 : nel
                    fprintf(fid,'%1.5g ',val(m,:));
                    fprintf(fid,'\n\t');
                end
                fprintf(fid,['</',pfx,'EphiPhase>\n']);
            end
        end
        
    end
    fprintf(fid,['</',pfx,'arrayant>\n']);
end

if remember_to_close_file
    fprintf(fid,'</qdant>\n');
    fclose(fid);
end

end
