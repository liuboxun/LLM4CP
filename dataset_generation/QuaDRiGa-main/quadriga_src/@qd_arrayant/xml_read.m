function [ h_array, l ] = xml_read( fn, fid, pfx, l, ignore_layout )
%XML_READ Reads antenna patterns from a QDANT XML file
%
% Calling object:
%   None (static method)
%
% Description:
%   The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna
%   pattern data in XML and load them into QuaDRiGa. The file format specification is described in
%   the documentation. This method loads the correctly formatted XML file into a qd_arrayant object
%   array.
%
% Input:
%   fn
%   Filename of the QDANT XML file.
%
%   fid
%   An integer that identifies an already opened file for subsequent low-level file I/O operations
%   (e.g. used when loading antenna data embedded in a KML file).
%
%   pfx
%   XML namespace declaration (string). It is possible to use a prefix to avoid name conflicts when
%   embedding QuaDRiGa antennas in other XML formats. The variable "pfx" is only required if the XML
%   file uses a namespace to identify the antenna objects.
%
%   l
%   Current line of the already opened file identified by fid.
%
%   ignore_layout
%   Boolean value. By default (0), the layout of multiple qd_arrayant objects is stored in the
%   QDANT file. This layout is restored by xml_read. Setting ignore_layout to 1 loads all
%   qd_arrayant objects in a [1 x N] object array.
%
% Output:
%   h_array
%   Array of qd_arrayant objects.
%
%   l
%   Last line of an already opened file identified by fid that was processed by xml_read.
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

if ~exist( 'pfx','var' ) || isempty( pfx )
    pfx = '';
else
    pfx = [pfx,':'];
end

if ~exist( 'ignore_layout','var' ) || isempty( ignore_layout )
    ignore_layout = false;
end

remember_to_close_file = false;
if ~exist( 'fid','var' ) || isempty( fid )
    if ~exist( 'fn','var' ) || isempty( fn )
        error('QuaDRiGa:qd_arrayant:xml_read:filename_not_given','You did not specify a filename.');
    end
    
    fid = fopen( fn ,'r' );                                    	% Open file for reading
    remember_to_close_file = true;
    
    l = fgets(fid);                                            	% Get first line
    
    % Check if an XML File is provided, print error message if not
    p1 = regexp(l,'<\?xml','once');                           	% Check for XML
    if isempty( p1 )
        fclose(fid);                                           	% Close file
        error('QuaDRiGa:qd_arrayant:xml_read:no_xml_found','No XML file.');
    end
    
    % Detect the namespace
    scan = true;
    while scan
        l = fgets(fid);
        if ischar(l)
            p1 = regexp(l,'xmlns');                           	% Check for XML namespace item
            if ~isempty( p1 )                                   % Check for Quadriga Namespace namespace item
                p2 =  regexp(l,'"http://www.quadriga-channel-model.de"','once');  
                if ~isempty( p2 )
                    p1 = p1(p2>p1); p1 = p1(end);             	% Last entry in p1
                    pfx = l(p1+6:p2-2);
                    if isempty( pfx )
                        pfx = '';
                    else
                        pfx = [pfx,':'];
                    end
                    scan = false;
                end
            end
        else
            fclose(fid);
            error('QuaDRiGa:qd_arrayant:xml_read:no_xml_found','No QuaDRiGa namespace found.');
        end
    end
end
npfx = numel(pfx);

% Read the layout and number of array antennas
scan = true;
while scan
    p1 = regexp(l,['<',pfx,'layout>']);                         % Check for Layout item
    if ~isempty( p1 )                                           % Layout found
        p2 = regexp(l,['</',pfx,'layout>']);                    % Check closing Layout item
        
        val = l(p1+npfx+8:p2-1);                                % Values
        nc = numel(regexp( val,' '))+1;                         % Number of columns in the array
        val = str2double( regexp( val, '[0-9]+', 'match' ) );
        a_cnt = max(val);                                       % Number of array antennas in the file
        a_ind = reshape( val,[],nc );                           % Index list for the array antennas
        
        pe = p2 + npfx + 9;                                     % End pointer
        if pe + 1 < numel(l)
            l = l(pe:end);                                      % Process rest of current line
        else
            l = fgets(fid);                                     % Read next line
        end
        scan = false;
    else
        p1 = regexp(l,['<',pfx,'arrayant'], 'once');            % Check for Arrayant item
        if ~isempty( p1 )
            a_cnt = 1;
            a_ind = 1;
            scan = false;
        end
    end
    if scan
        l = fgets(fid);                                         % Get new line
        if ~ischar(l)                                           % End of file reached before any antennas were found
            fclose(fid);
            error('QuaDRiGa:qd_arrayant:xml_read:no_antennas',...
                'File does not contain any array antennas.');
        end
    end
end

if ignore_layout
    a_ind = 1:a_cnt;                                            % Option to ignore the layout
end

% Read array antennas from file
a = qd_arrayant([]);
for n = 1 : a_cnt
    
    p1 = regexp(l,['<',pfx,'arrayant'],'once');                	% Check for Arrayant item
    while isempty( p1 )
        l = fgets(fid);                                         % Read next line
        p1 = regexp(l,['<',pfx,'arrayant'],'once');            	% Check for Arrayant item
    end
    pe = p1 + npfx + 10;                                        % End pointer
    
    % Read id
    p1 = regexp(l,'id="','once');                               % Check for id
    if isempty(p1)                                              % Single antenn without id
        id = 1;
    else
        p2 = regexp(l,'">','once');                             % Check for closing
        id = str2double( l(p1+4:p2-1) );                        % Convert
        pe = p2+2;                                              % End pointer
    end
    if id ~= n                                                  % Check if the ID matches the array index
        fclose(fid);
        error('QuaDRiGa:qd_arrayant:xml_read:wrong_id',...
            'Array IDs must be linear increasing numbers, starting from 1.');
    end
    
    % Default variables are used when they are not in the file
    name = 'New array';                                         % Default name
    CenterFrequency = 299792458;                                % Speed of light
    NoElements = 1;                                             % Single Element
    ElementPosition = [0;0;0];                                  % Position = Phase center
    Coupling = 1;                                               % No coupling
    
    % We only need to check for each item once. Hence, we track if an item was given.
    scan = true;
    c_name = true;
    c_CenterFrequency = true;
    c_NoElements = true;
    c_ElementPosition = true;
    c_ElevationGrid = true;
    c_AzimuthGrid = true;
    c_CouplingAbs = true;
    c_CouplingPhase = true;
    c_EthetaMag = true;
    c_EphiMag = true;
    
    while scan                                                  % Read contents
        if pe+1 < numel(l)
            l = l(pe:end);
        else
            l = fgets(fid);                                     % Get next line
        end
        
        if c_name && ~isempty( regexp(l,['<',pfx,'name>'], 'once') )
            c_name = false;                                     % Don't check again
            p1 = regexp(l,'name>','once');
            p2 = regexp(l,['</',pfx,'name>'],'once');
            pe = p2 + npfx + 7;                                 % End pointer
            name = l(p1+5:p2-1);
            
        elseif c_CenterFrequency && ~isempty( regexp(l,['<',pfx,'CenterFrequency>'], 'once') )
            c_CenterFrequency = false;
            p1 = regexp(l,'CenterFrequency>','once');
            p2 = regexp(l,['</',pfx,'CenterFrequency>'],'once');
            pe = p2 + npfx + 18;                                % End pointer
            CenterFrequency = str2double( l(p1+16:p2-1) );
            
        elseif c_NoElements && ~isempty( regexp(l,['<',pfx,'NoElements>'], 'once') )
            c_NoElements = false;
            p1 = regexp(l,'NoElements>','once');
            p2 = regexp(l,['</',pfx,'NoElements>'],'once');
            pe = p2 + npfx + 13;                                % End pointer
            NoElements = str2double( l(p1+11:p2-1) );
            c_EthetaMag = true( 1,NoElements );                 % Indicator for checking if Margnitude was loaded before Phase
            c_EphiMag   = true( 1,NoElements );
            ElementPosition = zeros( 3,NoElements );            % Default ElementPosition
            Coupling = eye( NoElements );                       % Default Coupling
            
        elseif c_ElementPosition && ~isempty( regexp(l,['<',pfx,'ElementPosition>'], 'once') )
            c_ElementPosition = false;
            p1 = regexp(l,'ElementPosition>','once');
            p2 = regexp(l,['</',pfx,'ElementPosition>'],'once');
            pe = p2 + npfx + 18;                                % End pointer
            
            val = l(p1+16:p2-1);
            val = str2double( regexp( val, '[0-9e.-]+', 'match' ) );
            ElementPosition = reshape( val,3,[]);
            if size( ElementPosition,2 ) ~= NoElements
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:wrong_NoElements',...
                    'NoElements does not match the number of columns in ElementPosition. You must define NoElements first.');
            end
            
        elseif c_ElevationGrid && ~isempty( regexp(l,['<',pfx,'ElevationGrid>'], 'once') )
            c_ElevationGrid = false;
            p1 = regexp(l,'ElevationGrid>','once');
            p2 = regexp(l,['</',pfx,'ElevationGrid>'],'once');
            pe = p2 + npfx + 16;                                % End pointer
            val = l(p1+14:p2-1);
            val = str2double( regexp( val, '[0-9e.-]+', 'match' ) );
            ElevationGrid = val * pi/180;
            if ~c_ElevationGrid && ~c_AzimuthGrid               % Initialize the variables for the pattern data
                size_read = size( ElevationGrid,2 ) * size( AzimuthGrid,2 );
                Etheta = zeros( size( ElevationGrid,2 ), size( AzimuthGrid,2 ), NoElements );
                Ephi = Etheta;
            end
            
        elseif c_AzimuthGrid && ~isempty( regexp(l,['<',pfx,'AzimuthGrid>'], 'once') )
            c_AzimuthGrid = false;
            p1 = regexp(l,'AzimuthGrid>','once');
            p2 = regexp(l,['</',pfx,'AzimuthGrid>'],'once');
            pe = p2 + npfx + 14;                                % End pointer
            val = l(p1+12:p2-1);
            val = str2double( regexp( val, '[0-9e.-]+', 'match' ) );
            AzimuthGrid = val * pi/180;
            if ~c_ElevationGrid && ~c_AzimuthGrid               % Initialize the variables for the pattern data
                size_read = size( ElevationGrid,2 ) * size( AzimuthGrid,2 );
                Etheta = zeros( size( ElevationGrid,2 ), size( AzimuthGrid,2 ), NoElements );
                Ephi = Etheta;
            end
            
        elseif c_CouplingAbs && ~isempty( regexp(l,['<',pfx,'CouplingAbs>'], 'once') )
            c_CouplingAbs = false;
            p1 = regexp(l,'CouplingAbs>','once');
            p2 = regexp(l,['</',pfx,'CouplingAbs>'],'once');
            pe = p2 + npfx + 14;                                % End pointer
            val = l(p1+12:p2-1);
            val = str2double( regexp( val, '[0-9e.-]+', 'match' ) );
            if numel(val) ~= NoElements*NoElements
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:wrong_NoElements',...
                    'NoElements does not match the size of CouplingAbs. You must define NoElements first.');
            end
            Coupling = reshape( val, NoElements, NoElements );
            
        elseif c_CouplingPhase && ~isempty( regexp(l,['<',pfx,'CouplingPhase>'], 'once') )
            if c_CouplingAbs
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:CouplingAbs_undefined',...
                    'You must define CouplingAbs before CouplingPhase.');
            end
            c_CouplingPhase = false;
            p1 = regexp(l,'CouplingPhase>','once');
            p2 = regexp(l,['</',pfx,'CouplingPhase>'],'once');
            pe = p2 + npfx + 16;                                % End pointer
            val = l(p1+14:p2-1);
            val = str2double( regexp( val, '[0-9e.-]+', 'match' ) );
            Coupling = Coupling .* exp( 1j * reshape( val, NoElements, NoElements )*pi/180 );
            
        elseif ~isempty( regexp(l,['<',pfx,'EthetaMag'], 'once') )
            if c_ElevationGrid || c_AzimuthGrid
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:samling_grid_undefined',...
                    'You must define ElevationGrid and AzimuthGrid before EthetaMag.');
            end
            p1 = regexp(l,'EthetaMag el="','once');
            if isempty( p1 )                                    % Only one element
                el = 1;
            else
                p2 = regexp(l,'">','once');
                el = str2double( l(p1+14:p2-1) );
            end
            if el > NoElements
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:wrong_NoElements',...
                    'NoElements does not match the element index in EthetaMag. You must define NoElements first.');
            end
            c_EthetaMag(el) = false;
            
            % Read pattern
            val = fscanf( fid, '%f', size_read );
            val = reshape( val, size( AzimuthGrid,2 ), size( ElevationGrid,2 ) ).';
            Etheta(:,:,el) = sqrt(10.^(0.1*val));
            
            doRead = true;
            while doRead
                l = fgets(fid);                                 % Get next line
                p1 = regexp(l,['</',pfx,'EthetaMag>'],'once');
                if ~isempty( p1 )
                    doRead = false;
                end
            end
            pe = p1 + npfx + 12;
            
        elseif ~isempty( regexp(l,['<',pfx,'EthetaPhase'], 'once') )
            p1 = regexp(l,'EthetaPhase el="','once');
            if isempty( p1 )                                    % Only one element
                el = 1;
            else
                p2 = regexp(l,'">','once');
                el = str2double( l(p1+16:p2-1) );
            end
            if c_EthetaMag(el)
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:EthetaMag_undefined',...
                    'You must define EthetaMag before EthetaPhase.');
            end
            
            % Read pattern
            val = fscanf( fid, '%f', size_read );
            val = reshape( val, size( AzimuthGrid,2 ), size( ElevationGrid,2 ) ).';
            Etheta(:,:,el) = Etheta(:,:,el) .* exp(1j*val*pi/180);
            
            doRead = true;
            while doRead
                l = fgets(fid);                                	% Get next line
                p1 = regexp(l,['</',pfx,'EthetaPhase>'],'once');
                if ~isempty( p1 )
                    doRead = false;
                end
            end
            pe = p1 + npfx + 14;
            
        elseif ~isempty( regexp(l,['<',pfx,'EphiMag'], 'once') )
            if c_ElevationGrid || c_AzimuthGrid
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:samling_grid_undefined',...
                    'You must define ElevationGrid and AzimuthGrid before EthetaMag.');
            end
            p1 = regexp(l,'EphiMag el="','once');
            if isempty( p1 )                                    
                el = 1;
            else
                p2 = regexp(l,'">','once');
                el = str2double( l(p1+12:p2-1) );
            end
            if el > NoElements
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:wrong_NoElements',...
                    'NoElements does not match the element index in EthetaMag. You must define NoElements first.');
            end
            c_EphiMag(el) = false;
            
            % Read pattern
            val = fscanf( fid, '%f', size_read );
            val = reshape( val, size( AzimuthGrid,2 ), size( ElevationGrid,2 ) ).';
            Ephi(:,:,el) = sqrt(10.^(0.1*val));
            
            doRead = true;
            while doRead
                l = fgets(fid);                                 % Get next line
                p1 = regexp(l,['</',pfx,'EphiMag>'],'once');
                if ~isempty( p1 )
                    doRead = false;
                end
            end
            pe = p1 + npfx + 10;
            
        elseif ~isempty( regexp(l,['<',pfx,'EphiPhase'], 'once') )
            p1 = regexp(l,'EphiPhase el="','once');
            if isempty( p1 )                                    % Only one element
                el = 1;
            else
                p2 = regexp(l,'">','once');
                el = str2double( l(p1+14:p2-1) );
            end
            if c_EphiMag(el)
                fclose(fid);
                error('QuaDRiGa:qd_arrayant:xml_read:EphiMag_undefined',...
                    'You must define EphiMag before EphiPhase.');
            end
            
            % Read pattern
            val = fscanf( fid, '%f', size_read );
            val = reshape( val, size( AzimuthGrid,2 ), size( ElevationGrid,2 ) ).';
            Ephi(:,:,el) = Ephi(:,:,el) .* exp(1j*val*pi/180);
            
            doRead = true;
            while doRead
                l = fgets(fid);                              	% Get next line
                p1 = regexp(l,['</',pfx,'EphiPhase>'],'once');
                if ~isempty( p1 )
                    doRead = false;
                end
            end
            pe = p1 + npfx + 12;
            
        elseif ~isempty( regexp(l,['</',pfx,'arrayant'], 'once') )
            scan = false;
        end
    end
    
    % Build qd_arrayant object
    a(1,n) = qd_arrayant([]);
    a(1,n).name = name;
    a(1,n).center_frequency = CenterFrequency;
    a(1,n).elevation_grid = ElevationGrid;
    a(1,n).azimuth_grid = AzimuthGrid;
    a(1,n).no_elements = NoElements;
    a(1,n).element_position = ElementPosition;
    a(1,n).Fa = Etheta;
    a(1,n).Fb = Ephi;
    a(1,n).coupling = Coupling;
end

if remember_to_close_file
    fclose(fid);                                               	% Close file
end

% Format output data
h_array = qd_arrayant([]);
for n = 1 : size( a_ind,1 )
    for m = 1 : size( a_ind,2 )
        h_array(n,m) = a(a_ind(n,m));
    end 
end

end
