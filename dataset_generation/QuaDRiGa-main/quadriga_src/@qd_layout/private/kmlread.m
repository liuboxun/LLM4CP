function k = kmlread( kmlfile )
%KMLREAD Reads a KML file into a structure
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

k.name = '';
nPlacemark = 0;                                                 % Placemark Counter
nExtendedData = 0;                                              % ExtendedData Counter

file = fopen( kmlfile ,'r' );                                   % Open file for reading
l = fgets(file);                                                % Get first line

% Check if an XML File is provided, print error message if not
p1 = regexp(l,'<\?xml','once');                                 % Check for XML
if isempty( p1 )
    fclose(fid);                                                % Close file
    error('QuaDRiGa:qd_layout:kmlread:no_xml_found','No XML file.');
end

while ischar(l)                                                 % Do until file ends
    
    if ~isempty( regexp(l,'<Placemark>', 'once') )              % Check for <Placemark>
        nPlacemark = nPlacemark + 1;                            % Increase Placemark counter
        [ k.Placemark(nPlacemark),l ] = parse_Placemark( file,l );
        
    elseif ~isempty( regexp(l,'<description>', 'once') )        % Check for <description>
        [ k.Description,l ] = parse_Description( file,l );
        
    elseif ~isempty( regexp(l,'<ExtendedData>', 'once') )       % Check for <ExtendedData>
        p1 = regexp(l,'<ExtendedData>', 'once');                % Check if we have <ExtendedData>
        pe = p1 + 14;
        if pe+1 < numel(l)
            l = fgets(file);
        else
            l = l(pe:end);
        end
        
        scan = true;
        while scan
            if ~isempty( regexp(l,'<qdant:', 'once') )              % Antenna data
                scan = false;
                [ k.qdant , l ] = qd_arrayant.xml_read([],file,'qdant',l,1);
                
            elseif ~isempty( regexp(l,'<Data name="', 'once') )     % Extended data
                scan = false;
                nExtendedData = nExtendedData + 1;
                [ k.ExtendedData(nExtendedData),l ] = parse_ExtendedData( file,l );
                
            elseif ~isempty( regexp(l,'</ExtendedData>', 'once') )     % Extended data
                scan = false;
                
            else
                l = fgets(file);                                    % Next line
            end
        end
        
    elseif ~isempty( regexp(l,'<name>', 'once') )               % Check for <name>
        p1 = regexp(l,'<name>');
        p2 = regexp(l,'</name>');
        k.name = l(p1+6:p2-1);                                  % Save last <name> outside <Placemark>
        l = l(p2+7:end);                                        % Update l
        
    else
        l = fgets(file);                                        % Get next line
    end
end
fclose(file);

end


%% ---- SUBFUNCTION: parse_ExtendedData ----
function [ ExtendedData,l ] = parse_ExtendedData( file,l )
%PARSE_EXTENDEDDATA Parses the ExtendedData in a KML file

ExtendedData = struct;                                      % Create output structure

iExtendedData = true;
while iExtendedData                                         % Read contents
    
    % Read key
    p1 = regexp(l,'<Data name="','once');                   % Check for next data field
    while iExtendedData && isempty( p1 )
        p2 = regexp(l,'</ExtendedData>','once');            % Check for end of <ExtendedData>
        if isempty( p2 )
            l = fgets(file);                                % Read new line from file
            p1 = regexp(l,'<Data name="','once');           % Check for next data field
        else
            l = l(p2+15:end);
            iExtendedData = false;
        end
    end
    if iExtendedData
        p2 = regexp(l,'">','once');                        	% Check for closing tag
        ExtendedData_name = l( p1+12:p2-1 );               	% Store extended data name
        l = l( p2+2:end );                                	% Shorten line string
        
        % Read value
        ExtendedData_value = '';                           	% Empty string
        p1 = regexp( l,'<value>', 'once');                	% Check for value field
        while isempty( p1 )                                 % If there is none ...
            l = fgets(file);                                % Read new line from file
            p1 = regexp( l,'<value>', 'once');              % Check for value field
        end
        doRead = true;
        while doRead
            p2 = regexp( l,'</value>', 'once');             % Check for closing value field
            if isempty( p2 )
                ExtendedData_value = [ ExtendedData_value, l(p1+7:end) ];
                l = fgets(file);                            % Read new line from file
                p1 = -6;                                    % Adjust p1 so that the next iteration starts at 1
            else
                ExtendedData_value = [ ExtendedData_value, l(p1+7:p2-1)];
                l = l( p2+8:end );
                doRead = false;
            end
        end
        
        % Add data to struct
        ExtendedData.( ExtendedData_name ) = ExtendedData_value;
    end
end

end


%% ---- SUBFUNCTION: parse_Description ----
function [ Description , l ] = parse_Description( file,l )
%PARSE_DESCRIPTION Parses the description element

Description = struct;

p = regexp(l,'<description>', 'once');                            % Check if we have <description>
if ~isempty( p )
    iDescription = true;
    l = l( p+13:end );                                          % Set staring pointer
else
    iDescription = false;
end

while iDescription
    
    p1 = regexp(l,'=','once');                           % Check if there is an equal sign
    p2 = regexp(l,'</description>');
    if ~isempty(p1)                                 % If there is a "=" sign
        Description_name = regexp( l(1:p1-1) ,'[A-Za-z0-9_]+','match');           % Read name
        if isempty( p2 )       % Read value
            Description_value = regexp( l(p1+1:end) ,'[A-Za-z0-9_\-:,. ]+','match');
        else
            Description_value = regexp( l(p1+1:p2-1) ,'[A-Za-z0-9_\-:,. ]+','match');
        end
        
        % Remove spaces at the beginning of the values
        ind_space = 0;
        while Description_value{1}(ind_space+1) == ' '
            ind_space = ind_space + 1;
        end
        if ind_space
            Description_value{1} = Description_value{1}(ind_space+1:end);
        end
        
        try
            Description.( Description_name{1} ) = Description_value{1};   % Save to output variable
        end
    end
    if isempty(p2)
        l = fgets( file );  % Read next line
    else
        iDescription = false;
        l = l(p2+14:end);   % Return rest of line for further processing
    end
end


end

%% ---- SUBFUNCTION: parse_Placemark ----
function [ Placemark,l ] = parse_Placemark( file,l )
%PARSE_PLACEMARK Parses the Placemark element

% Initialize output
Placemark = struct;                                             % Create output structure
Placemark.Name = '';                                            % Initialize Name
Placemark.Type = '';                                            % Initialize Type
Placemark.Coordinates = [];
Placemark.ExtendedData = struct;
Placemark.Description = struct;

p = regexp(l,'<Placemark>', 'once');                            % Check if we have <Placemark>
if ~isempty( p )
    iPlacemark = true;
    l = l( p+14:end );                                          % Set staring pointer
else
    iPlacemark = false;
end

while iPlacemark                                                % Read contents
    
    % Check for <name>
    if ~isempty(l) && ~isempty( regexp(l,'<name>', 'once') )
        p1 = regexp(l,'<name>');
        p2 = regexp(l,'</name>');
        Placemark.Name = l(p1+6:p2-1);                          % Save last <name> outside <Placemark>
        l = l(p2+7:end);                                        % Update l
    
    % Check for <Point> or <LineString>
    elseif ~isempty(l) && (~isempty( regexp(l,'<Point>', 'once') ) || ~isempty( regexp(l,'<LineString>', 'once') ))
        p1 = regexp(l,'<Point>');
        if ~isempty( p1 )
            l = l(p1+7:end);
            Placemark.Type = 'Point';
        else
            p1 = regexp(l,'<LineString>');
            if ~isempty( p1 )
                l = l(p1+12:end);
                Placemark.Type = 'LineString';
            end
        end
        
        Coordinates = '';                                       % Empty string
        p1 = regexp( l,'<coordinates>', 'once');                % Check for coordinates field
        while isempty( p1 )                                     % If there is none ...
            l = fgets(file);                                    % Read new line from file
            p1 = regexp( l,'<coordinates>', 'once');            % Check for coordinates field
        end
        doRead = true;
        while doRead
            p2 = regexp( l,'</coordinates>', 'once');           % Check for closing coordinates field
            if isempty( p2 )
                Coordinates = [ Coordinates, ' ', l(p1+13:end) ];
                l = fgets(file);                              	% Read new line from file
                p1 = -12;                                       % Adjust p1 so that the next iteration starts at 1
            else
                Coordinates = [ Coordinates, ' ', l(p1+13:p2-1)];
                l = l( p2+14:end );
                doRead = false;
            end
        end
        Coordinates = regexp( Coordinates, '[0-9e.-]+', 'match' );
        Coordinates = str2double( Coordinates );
        Placemark.Coordinates = reshape( Coordinates , 3 , [] );
        
    % Check for exit statement
    elseif ~isempty(l) && ~isempty( regexp(l,'</Placemark>', 'once') )
        p1 = regexp(l,'</Placemark>', 'once');
        l = l(p1+12:end);
        iPlacemark = false; 
        
    % Check for <ExtendedData>
    elseif ~isempty(l) && ~isempty( regexp(l,'<ExtendedData>', 'once') )
        [ Placemark.ExtendedData,l ] = parse_ExtendedData( file,l );
        
    % Check for <description>
    elseif ~isempty(l) && ~isempty( regexp(l,'<description>', 'once') )
        [ Placemark.Description,l ] = parse_Description( file,l );
        
    else
        %l = fgetl(file);                                        % Get next line
        l = fgets(file);
    end
    
end

end
