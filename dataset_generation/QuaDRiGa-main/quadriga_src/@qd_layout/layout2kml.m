function layout2kml( h_layout , fn , reference_coord, embed_antennas, use_description, split_segments, absolute_altitude )
%LAYOUT2KML Exports a layout object to a KML file
%
% Calling object:
%   Single object
%
% Description:
%   This function exports a QuaDRiGa layout object to a KML file. KML-Files can be read e.g. by
%   Google(TM) maps. A complete description of QuaDRiGa-KML specification can be found in the
%   documentation.
%
% Input:
%   fn
%   The filename of the KML-File.
%
%   reference_coord
%   A tuple for longitude and latitude (WGS84) at which the origin (0,0,0) of the metric Cartesian
%   coordinate system used by QuaDRiGa is placed (optional). If this value is not given, the origin
%   is placed at QuaDRiGas origin: 13.324947e,52.516319n.
%
%   embed_antennas
%   Boolean value (optional). By default (1), antennas are embedded into the KML file. If disabled
%   (0), antennas are written to an external QDANT file.
%
%   use_description
%   Boolean value (optional). By default (0), additional QuaDRiGa simulation parameters are written
%   to the ExtendedData elements in the KML file. If enabled (1), parameters are written to the
%   description element. Description elements can be edited in Google earth.
%
%   split_segments
%   This parameter controls the splitting of long segments into sub-segments with the same scenario
%   definition. SplitSegments is a tuple of 4 values defining:
%
%    * min. length of a sub-segment (e.g. 10 m)
%    * max. length of the sub-segment; must be >2·min. (e.g. 30 m)
%    * average length of the sub-segment (e.g. 15 m)
%    * standard-deviation of a sub-segment (e.g. 5 m)
%
%   The four values are written to the KML file and applied when the file is loaded by kml2layout.
%   The effect will not be visible when viewing the KML file in Google earth. If SplitSegments is
%   not defined, segment splitting is disabled.
%
%   absolute_altitude
%   Boolean value (optional). By default (0), heights are stored relative to the ground height. If
%   set to true, heights are stored relative to sea level.
%
%
% QuaDRiGa Copyright (C) 2011-2021
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

if ~exist('reference_coord','var') || isempty( reference_coord )
    reference_coord_trx = cat(1,h_layout.tx_track.ReferenceCoord);
    if ~isempty( h_layout.ReferenceCoord )              % Does layout have a reference?
        reference_coord = h_layout.ReferenceCoord;      % Use layout variable
    elseif ~isempty( reference_coord_trx )              % Does Tracks have a reference?
        if all( abs(reference_coord_trx(:,1) - reference_coord_trx(1,1)) < 1e-12 ) && ...
                all( abs(reference_coord_trx(:,2) - reference_coord_trx(1,2)) < 1e-12 )
            reference_coord = reference_coord_trx(1,:); % Use track reference
        else
            error('QuaDRiGa:qd_layout:layout2kml',...
                '"reference_coord" in tx tracks is ambiguous.');
        end
    else
        reference_coord = [ 13.3249472 , 52.5163194 ];  % HHI
    end
end

if ~exist('embed_antennas','var') || isempty( embed_antennas )
    embed_antennas = true;
end

if ~exist('use_description','var') || isempty( use_description )
    use_description = false;
end

if ~exist('absolute_altitude','var') || isempty( absolute_altitude )
    absolute_altitude = false;
end

if ~exist('split_segments','var') || isempty( split_segments )
    split_segments = [];
elseif size( split_segments,2 ) ~= 4
    error('QuaDRiGa:qd_layout:layout2kml',...
        '"split_segments" must have 4 elements.');
end

% Fix for Octave
h_layout = h_layout(1,1);

% Make sure, all variables are double precision
tx_track_all = h_layout.tx_track;
rx_track_all = h_layout.rx_track;
for n = 1 : h_layout.no_tx
    tx_track_all(1,n).initial_position = double( tx_track_all(1,n).initial_position );
    tx_track_all(1,n).positions = double( tx_track_all(1,n).positions );
end
for n = 1 : h_layout.no_rx
    rx_track_all(1,n).initial_position = double( rx_track_all(1,n).initial_position );
    rx_track_all(1,n).positions = double( rx_track_all(1,n).positions );
end
h_layout.tx_track = tx_track_all;
h_layout.rx_track = rx_track_all;
h_layout.simpar(1,1).center_frequency = double( h_layout.simpar.center_frequency );

% Process antennas
a = qd_arrayant([]);                                            % Empty array of qd_arrayant objects
a_cnt = 0;                                                      % Counter
a_txind = zeros( size( h_layout.tx_array) );                    % Index list for the Tx
a_rxind = zeros( size( h_layout.rx_array) );                    % Index list for the Tx
for ifrq = 1 : size( h_layout.tx_array,1 )
    for iue = 1 : size( h_layout.tx_array,2 )
        tx_array = h_layout.tx_array( ifrq,iue );               % Copy handle
        iseq = qf.eqo( tx_array, a );                           % Check if antenna already exists
        if any( iseq )
            a_txind( ifrq,iue ) = find( iseq );                 % Save index
        elseif tx_array.no_elements == 1 && all( tx_array.Fa(:) == 1 ) && all( tx_array.Fb(:) == 0 )
            % Omni-antennas are not stored
        else
            a_cnt = a_cnt + 1;                                  % Increase counter
            a(1,a_cnt) = tx_array;                              % Copy handle
            a_txind( ifrq,iue ) = a_cnt;                        % Save index
        end
    end
end
for ifrq = 1 : size( h_layout.rx_array,1 )
    for iue = 1 : size( h_layout.rx_array,2 )
        rx_array = h_layout.rx_array( ifrq,iue );               % Copy handle
        iseq = qf.eqo( rx_array, a );                           % Check if antenna already exists
        if any( iseq )
            a_rxind( ifrq,iue ) = find( iseq );                 % Save index
        elseif rx_array.no_elements == 1 && all( rx_array.Fa(:) == 1 ) && all( rx_array.Fb(:) == 0 )
            % Omni-antennas are not stored
        else
            a_cnt = a_cnt + 1;                                  % Increase counter
            a(1,a_cnt) = rx_array;                              % Copy handle
            a_rxind( ifrq,iue ) = a_cnt;                        % Save index
        end
    end
end

f = fopen( fn,'w' );                                            % Open KML file for writing

fprintf(f,'<?xml version="1.0" encoding="UTF-8"?>\n');          % XML declaration
if embed_antennas                                               % Define QDANT namespace
    fprintf(f,'<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:qdant="http://www.quadriga-channel-model.de">\n');
else
    fprintf(f,'<kml xmlns="http://www.opengis.net/kml/2.2">\n');
end
fprintf(f,'<Document>\n');                                      % Add Document element
fprintf(f,['<name>',fn,'</name>\n']);                           % Save Filename

if any( [ a_txind(:) ; a_rxind(:) ] ~= 0 )
    if embed_antennas                                           % Save the antennas
        fprintf(f,'<ExtendedData>\n');                          % Embedded in the KML
        xml_write( a, [], 'qdant', [], f );
        fprintf(f,'</ExtendedData>\n');
    else
        xml_write( a, [fn,'.qdant'] );                          % External file
    end
end

fprintf(f,'<Folder>\n');                                        % Add Folder element
fprintf(f,['\t<name>',h_layout.name,'</name>\n']);              % Layout name

% Extended Data for the layout
if use_description
    fprintf(f,'\t<description>\n');
else
    fprintf(f,'\t<ExtendedData>\n');
end

val = sprintf('%1.14g,',h_layout.simpar(1,1).center_frequency);      % CenterFrequency
if use_description
    fprintf(f,['CenterFrequency = ',val(1:end-1),'\n']);
else
    fprintf(f,'\t\t<Data name="CenterFrequency">');
    fprintf(f,['<value>',val(1:end-1),'</value>']);
    fprintf(f,'</Data>\n');
end

% Check if all tracks have the movement profile enabled
use_update_rate = true;
for n = 1 : h_layout.no_tx
    if tx_track_all(1,n).no_snapshots > 1 && isempty( tx_track_all(1,n).movement_profile )
        use_update_rate = false;
    end
end
for n = 1 : h_layout.no_rx
    if rx_track_all(1,n).no_snapshots > 1 && isempty( rx_track_all(1,n).movement_profile )
        use_update_rate = false;
    end
end
if ~isempty( h_layout.update_rate ) && ~use_update_rate
    error('QuaDRiGa:qd_layout:layout2kml',...
        'In order to set an "UpdateRate", you must specify a movement profile for each track.');
elseif ~isempty( h_layout.update_rate )
    val =  sprintf('%1.14g',h_layout.update_rate);
    if use_description
        fprintf(f,['UpdateRate = ',val,'\n']);
    else
        fprintf(f,'\t\t<Data name="UpdateRate">');
        fprintf(f,['<value>',val,'</value>']);
        fprintf(f,'</Data>\n');
    end
end

val = sprintf('%1.14g',h_layout.simpar(1,1).sample_density);
if use_description
    fprintf(f,['SampleDensity = ',val,'\n']);
else
    fprintf(f,'\t\t<Data name="SampleDensity">');
    fprintf(f,['<value>',val,'</value>']);
    fprintf(f,'</Data>\n');
end

val = sprintf('%u',h_layout.simpar(1,1).use_absolute_delays);
if use_description
    fprintf(f,['AbsoluteDelays = ',val,'\n']);
else
    fprintf(f,'\t\t<Data name="AbsoluteDelays">');
    fprintf(f,['<value>',val,'</value>']);
    fprintf(f,'</Data>\n');
end

val = sprintf('%u',h_layout.simpar(1,1).use_random_initial_phase);
if use_description
    fprintf(f,['RandInitPhase = ',val,'\n']);
else
    fprintf(f,'\t\t<Data name="RandInitPhase">');
    fprintf(f,['<value>',val,'</value>']);
    fprintf(f,'</Data>\n');
end

val = sprintf('%u',h_layout.simpar(1,1).use_3GPP_baseline);
if use_description
    fprintf(f,['Baseline3GPP = ',val,'\n']);
else
    fprintf(f,'\t\t<Data name="Baseline3GPP">');
    fprintf(f,['<value>',val,'</value>']);
    fprintf(f,'</Data>\n');
end

val = sprintf('%u',h_layout.simpar(1,1).show_progress_bars);
if use_description
    fprintf(f,['ProgressReport = ',val,'\n']);
else
    fprintf(f,'\t\t<Data name="ProgressReport">');
    fprintf(f,['<value>',val,'</value>']);
    fprintf(f,'</Data>\n');
end

if use_description
    fprintf(f,['AutoCorrFcn = ',h_layout.simpar(1,1).autocorrelation_function,'\n']);
else
    fprintf(f,'\t\t<Data name="AutoCorrFcn">');
    fprintf(f,['<value>',h_layout.simpar(1,1).autocorrelation_function,'</value>']);
    fprintf(f,'</Data>\n');
end

val = sprintf('%u,%u ',h_layout.pairing(:));
if use_description
    fprintf(f,['Pairing = ',val(1:end-1),'\n']);
else
    fprintf(f,'\t\t<Data name="Pairing">');
    fprintf(f,['<value>',val(1:end-1),'</value>']);
    fprintf(f,'</Data>\n');
end

if ~isempty(split_segments)
    val = sprintf('%1.14g,',split_segments);
    if use_description
        fprintf(f,['SplitSegments = ',val(1:end-1),'\n']);
    else
        fprintf(f,'\t\t<Data name="SplitSegments">');
        fprintf(f,['<value>',val(1:end-1),'</value>']);
        fprintf(f,'</Data>\n');
    end
end

val = sprintf('%1.14g,',reference_coord);
if use_description
    fprintf(f,['ReferenceCoord = ',val(1:end-1),'\n']);
else
    fprintf(f,'\t\t<Data name="ReferenceCoord">');
    fprintf(f,['<value>',val(1:end-1),'</value>']);
    fprintf(f,'</Data>\n');
end

if use_description
    fprintf(f,'\t</description>\n');
else
    fprintf(f,'\t</ExtendedData>\n');
end

% Save Tx-positions
for n = 1 : h_layout.no_tx
    fprintf(f,'\t<Placemark>\n');
    
    str = tx_track_all(1,n).name;
    str = regexprep(str,'\_','-');
    str = ['tx_' ,str ];
    fprintf(f,['\t\t<name>',str,'</name>\n']);
    
    ExtendedData = false;
    if any( abs( tx_track_all(1,n).orientation(:) ) > 1e-7 )
        ExtendedData = true;
    end
    if ~isempty( tx_track_all(1,n).movement_profile )
        ExtendedData = true;
    end
    if any( a_txind(:,n) ~= 0 )
        ExtendedData = true;
    end
    
    % Save orientation data (only if different from Default values, in DEG)
    if ExtendedData
        if use_description
            fprintf(f,'\t\t<description>\n');
        else
            fprintf(f,'\t\t<ExtendedData>\n');
        end
        if any( a_txind(:,n) ~= 0 )                             % Is there any antenna that is not "omni"
            if use_description
                fprintf(f,'Antenna = ');
            else
                fprintf(f,'\t\t\t<Data name="Antenna"><value>');% Save antenna data
            end
            for m = 1 : size(a_txind,1)                       	% for each frequency
                if size(a_txind,1) > 1 && a_txind(m,n) == 0
                    fprintf(f,':0');                           	% Omni antenna
                else
                    if ~embed_antennas
                        fprintf(f,[fn,'.qdant']);              	% Antenna file name
                    end
                    fprintf(f,[':',num2str(a_txind(m,n))]);    	% Save index
                end
                if m < size(a_txind,1)
                    fprintf(f,',');                             % Comma to separate antennas for different frequencies
                end
            end
            if use_description
                fprintf(f,'\n');
            else
                fprintf(f,'</value></Data>\n');
            end
        end
        if any( abs( tx_track_all(1,n).orientation(1,:) ) > 1e-7 )
            val = sprintf('%1.8g,',tx_track_all(1,n).orientation(1,:)*180/pi);
            if use_description
                fprintf(f,['Bank = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Bank">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if any( abs( tx_track_all(1,n).orientation(2,:) ) > 1e-7 )
            val = sprintf('%1.8g,',tx_track_all(1,n).orientation(2,:)*180/pi);
            if use_description
                fprintf(f,['Tilt = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Tilt">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if any( abs( tx_track_all(1,n).orientation(3,:) ) > 1e-7 )
            val = sprintf('%1.8g,',tx_track_all(1,n).orientation(3,:)*180/pi);
            if use_description
                fprintf(f,['Heading = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Heading">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if ~isempty( tx_track_all(1,n).movement_profile )
            val = sprintf('%1.8g,',tx_track_all(1,n).movement_profile(1,:));
            if use_description
                fprintf(f,['Time = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Time">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
            
            val = sprintf('%1.12g,',tx_track_all(1,n).movement_profile(2,:));
            if use_description
                fprintf(f,['Distance = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Distance">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if use_description
            fprintf(f,'\t\t</description>\n');
        else
            fprintf(f,'\t\t</ExtendedData>\n');
        end
    end
    
    pos = tx_track_all(1,n).positions_abs;
    [ pos(1,:), pos(2,:), pos(3,:) ] = trans_ue2global( pos, reference_coord );
    coordinates = sprintf('%1.14g,%1.14g,%1.14g ',pos);
    if any(pos(3,:) > 1000)
        absolute_altitude = true;
    end
    
    if tx_track_all(1,n).no_snapshots == 1
        fprintf(f,'\t\t<Point>\n');
        if absolute_altitude
            fprintf(f,'\t\t\t<extrude>1</extrude><altitudeMode>absolute</altitudeMode>\n');
        else
            fprintf(f,'\t\t\t<extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>\n');
        end
        fprintf(f,['\t\t\t<coordinates>',coordinates(1:end-1),'</coordinates>\n']);
        fprintf(f,'\t\t</Point>\n');
    else
        fprintf(f,'\t\t<LineString>\n');
        if absolute_altitude
            fprintf(f,'\t\t\t<extrude>0</extrude><altitudeMode>absolute</altitudeMode>\n');
        else
            fprintf(f,'\t\t\t<extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>\n');
        end
        fprintf(f,['\t\t\t<coordinates>',coordinates(1:end-1),'</coordinates>\n']);
        fprintf(f,'\t\t</LineString>\n');
    end
    fprintf(f,'\t</Placemark>\n');
end

% Save Rx-tracks
for n = 1 : h_layout.no_rx
    fprintf(f,'\t<Placemark>\n');
    
    rx_name_str = rx_track_all(1,n).name;
    rx_name_str = regexprep(rx_name_str,'\_','-');
    rx_name_str = ['rx_' ,rx_name_str ];
    fprintf(f,['\t\t<name>',rx_name_str,'</name>\n']);
    
    ExtendedData = false;
    if any( abs( rx_track_all(1,n).orientation(:) ) > 1e-7 )
        ExtendedData = true;
    end
    if ~isempty( rx_track_all(1,n).movement_profile )
        ExtendedData = true;
    end
    if any( a_rxind(:,n) ~= 0 )
        ExtendedData = true;
    end
    
    % Save orientation data (only if different from Default values, in DEG)
    rx_track = rx_track_all(1,n);
    if ExtendedData
        if use_description
            fprintf(f,'\t\t<description>\n');
        else
            fprintf(f,'\t\t<ExtendedData>\n');
        end
        if any( a_rxind(:,n) ~= 0 )                             % Is there any antenna that is not "omni"
            if use_description
                fprintf(f,'Antenna = ');
            else
                fprintf(f,'\t\t\t<Data name="Antenna"><value>'); % Save antenna data
            end
            for m = 1 : size(a_rxind,1)                         % for each frequency
                if size(a_rxind,1) > 1 && a_rxind(m,n) == 0
                    fprintf(f,':0');                          	% Omni antenna
                else
                    if ~embed_antennas
                        fprintf(f,[fn,'.qdant']);              	% Antenna file name
                    end
                    fprintf(f,[':',num2str(a_rxind(m,n))]);   	% Save index
                end
                if m < size(a_rxind,1)
                    fprintf(f,',');                             % Comma to separate antennas for different frequencies
                end
            end
            if use_description
                fprintf(f,'\n');
            else
                fprintf(f,'</value></Data>\n');
            end
        end
        
        % Save orientation only if it differs from the orientations calculated from the track
        orientation = rx_track.orientation;
        rx_track.calc_orientation;
        if any( abs( rx_track.orientation(1,:) - orientation(1,:) ) > 1e-7 )
            val = sprintf('%1.8g,',orientation(1,:)*180/pi);
            if use_description
                fprintf(f,['Bank = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Bank">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if any( abs( rx_track.orientation(2,:) - orientation(2,:) ) > 1e-7 )
            val = sprintf('%1.8g,',orientation(2,:)*180/pi);
            if use_description
                fprintf(f,['Tilt = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Tilt">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if any( abs( rx_track.orientation(3,:) - orientation(3,:) ) > 1e-7 )
            val = sprintf('%1.8g,',orientation(3,:)*180/pi);
            if use_description
                fprintf(f,['Heading = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Heading">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                
                fprintf(f,'</Data>\n');
            end
        end
        rx_track.orientation = orientation;
        
        if ~isempty( rx_track.movement_profile )
            val = sprintf('%1.8g,',rx_track.movement_profile(1,:));
            if use_description
                fprintf(f,['Time = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Time">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
            
            val = sprintf('%1.12g,',rx_track.movement_profile(2,:));
            if use_description
                fprintf(f,['Distance = ',val(1:end-1),'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Distance">');
                fprintf(f,['<value>',val(1:end-1),'</value>']);
                fprintf(f,'</Data>\n');
            end
        end
        if use_description
            fprintf(f,'\t\t</description>\n');
        else
            fprintf(f,'\t\t</ExtendedData>\n');
        end
    end
    
    pos = rx_track_all(1,n).positions_abs;
    [ pos(1,:), pos(2,:), pos(3,:) ] = trans_ue2global( pos, reference_coord );
    coordinates = sprintf('%1.12g,%1.12g,%1.12g ',pos);
    if any(pos(3,:) > 1000)
        absolute_altitude = true;
    end
    
    if rx_track_all(1,n).no_snapshots == 1
        fprintf(f,'\t\t<Point>\n');
        if absolute_altitude
            fprintf(f,'\t\t\t<extrude>1</extrude><altitudeMode>absolute</altitudeMode>\n');
        else
            fprintf(f,'\t\t\t<extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>\n');
        end
        fprintf(f,['\t\t\t<coordinates>',coordinates(1:end-1),'</coordinates>\n']);
        fprintf(f,'\t\t</Point>\n');
    else
        fprintf(f,'\t\t<LineString>\n');
        if absolute_altitude
            fprintf(f,'\t\t\t<extrude>0</extrude><altitudeMode>absolute</altitudeMode>\n');
        else
            fprintf(f,'\t\t\t<extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>\n');
        end
        fprintf(f,['\t\t\t<coordinates>',coordinates(1:end-1),'</coordinates>\n']);
        fprintf(f,'\t\t</LineString>\n');
    end
    fprintf(f,'\t</Placemark>\n');
    
    % Save segments and scenarios
    ntx = size(rx_track.scenario,1);
    if ~( rx_track.no_segments == 1 && numel( rx_track.scenario ) == 1 && isempty( rx_track.scenario{1,1} ) )
        for m = 1 : rx_track.no_segments
            fprintf(f,'\t<Placemark>\n');
            
            % Set name string
            if ntx == 1
                str = ['seg_',rx_track.scenario{1,m}];
            else
                str = 'seg_';
                for o = 1 : ntx
                    if sum( h_layout.pairing(2,:) == n & h_layout.pairing(1,:) == o) == 1
                        str = [str , rx_track.scenario{o,m} , ':' ];
                    else
                        str = [str , '-' , ':' ];
                    end
                end
                str = str(1:end-1);
            end
            fprintf(f,['\t\t<name>',str,'</name>\n']);
            
            if use_description
                fprintf(f,'\t\t<description>\n');
            else
                fprintf(f,'\t\t<ExtendedData>\n');
            end
            
            if use_description
                fprintf(f,['Track = ',rx_name_str,'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Track">');
                fprintf(f,['<value>',rx_name_str,'</value>']);
                fprintf(f,'</Data>\n');
            end
            val = sprintf('%d',rx_track.segment_index(m));
            if use_description
                fprintf(f,['Index = ',val,'\n']);
            else
                fprintf(f,'\t\t\t<Data name="Index">');
                fprintf(f,['<value>',val,'</value>']);
                fprintf(f,'</Data>\n');
            end
            
            if use_description
                fprintf(f,'\t\t</description>\n');
            else
                fprintf(f,'\t\t</ExtendedData>\n');
            end
            
            pos = rx_track.positions( :,rx_track.segment_index(m) ) + rx_track.initial_position;
            [ pos(1,:), pos(2,:), pos(3,:) ] = trans_ue2global( pos, reference_coord );
            coordinates = sprintf('%1.14g,%1.14g,%1.14g ',pos);
            if any(pos(3,:) > 1000)
                absolute_altitude = true;
            end
            
            fprintf(f,'\t\t<Point>\n');
            if absolute_altitude
                fprintf(f,'\t\t\t<altitudeMode>absolute</altitudeMode>\n');
            else
                fprintf(f,'\t\t\t<altitudeMode>relativeToGround</altitudeMode>\n');
            end
            fprintf(f,['\t\t\t<coordinates>',coordinates(1:end-1),'</coordinates>\n']);
            fprintf(f,'\t\t</Point>\n');
            fprintf(f,'\t</Placemark>\n');
        end
    end
end

fprintf(f,'</Folder>\n');
fprintf(f,'</Document>\n');
fprintf(f,'</kml>\n');

fclose(f);

end

