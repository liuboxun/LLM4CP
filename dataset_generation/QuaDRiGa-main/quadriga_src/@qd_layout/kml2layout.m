function [ h_layout, ReferenceCoord ] = kml2layout( fn, split_seg, ForceRefCoord )
%KML2LAYOUT Imports a layout object from a KML file 
%
% Calling object:
%   None (static method)
%
% Description:
%   This function loads a QuaDRiGa layout from a KML-File. KML-Files are created e.g. by Google(TM)
%   maps or a GPS device. In the KML-File, the user can specify the positions of the transmitters
%   (Tx), the receiver tracks (Rx) and segments for the receiver track. In order to work properly,
%   the KML-File needs to meet some specific formatting requirements: Tx-positions are represented
%   by a "Placemark" in the KML-file. The name must be "tx_TxName" where 'TxName' has to be
%   replaced by a unique name for each transmitter. Rx-Tracks are specified by paths. Each path
%   found in the KML-File is interpreted as a Rx-track. As for the Tx, all paths must contain a
%   unique name. Segments of a Rx-Track are determined by placemarks in close proximity to the
%   track. The name of the placemark contains the scenario. The naming convention for segments is
%   "seg_Scen" where 'Scen' determines the scenario of the segment. For example, you can create a
%   path with the name "rx_GPS1". In order so assign the scenario "WINNER_UMa_C2_LOS" to the path,
%   you need to add a placemark at the beginning of the track with the name:
%   "seg_WINNER_UMa_C2_LOS". It also possible to specify a different scenario for each transmitter
%   in the layout. The naming convention then is "seg_ScenTx1:ScenTx2:...:ScenTxN" where the
%   scenarios for each transmitter are separated by a ":". A complete description of QuaDRiGa-KML
%   specification can be found in the documentation.
%
% Input:
%   fn
%   The filename of the KML-File (string)
%
%   split_segment
%   It set to true (1, default), tracks are split into segment as indicated by the parameter
%   "SplitSegments" in the KML file. If set to false (0), "SplitSegments" is ignored. 
%
%   ForceRefCoord
%   A tuple for longitude and latitude (WGS84) at which the origin (0,0,0) of the metric Cartesian
%   coordinate system used by QuaDRiGa is placed. If this value is given, any value provided in the
%   KML file is ignored.
%
% Output:
%   h_layout
%   The 'qd_layout' object
%
%   ReferenceCoord
%   A tuple for longitude and latitude (WGS84) at which the origin (0,0,0) of the metric Cartesian
%   coordinate system used by QuaDRiGa is placed.
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

if ~exist( 'split_seg', 'var' ) || isempty( split_seg )
   split_seg = true; 
end

if ~exist( 'ForceRefCoord', 'var' ) || isempty( ForceRefCoord )
   ForceRefCoord = []; 
end

% Parse KML file
p = kmlread( fn );

% Create New Layout
h_layout = qd_layout;

if ~isempty( p.name )
    h_layout.name = p.name;
end

% Parse the Extended data element
for n = 1 : 2
    switch n
        case 1
            fld = 'ExtendedData';
        case 2
            fld = 'Description';
    end
    
    if isfield( p , fld )
        if isfield( p.(fld) , 'CenterFrequency' )
            h_layout.simpar(1,1).center_frequency = ...
                str2double( regexp( p.(fld).CenterFrequency,'[0-9e.-]+','match' ) );
        end
        if isfield( p.(fld) , 'SampleDensity' )
            h_layout.simpar(1,1).sample_density = ...
                str2double( regexp( p.(fld).SampleDensity,'[0-9e.-]+','once','match' ) );
        end
        if isfield( p.(fld) , 'AbsoluteDelays' )
            h_layout.simpar(1,1).use_absolute_delays = ...
                str2double( regexp( p.(fld).AbsoluteDelays,'[0,1]','once','match' ) );
        end
        if isfield( p.(fld) , 'RandInitPhase' )
            h_layout.simpar(1,1).use_random_initial_phase = ...
                str2double( regexp( p.(fld).RandInitPhase,'[0,1]','once','match' ) );
        end
        if isfield( p.(fld) , 'Baseline3GPP' )
            h_layout.simpar(1,1).use_3GPP_baseline = ...
                str2double( regexp( p.(fld).Baseline3GPP,'[0,1]','once','match' ) );
        end
        if isfield( p.(fld) , 'AutoCorrFcn' )
            h_layout.simpar(1,1).autocorrelation_function = regexp( p.(fld).AutoCorrFcn,'[a-zA-Z0-9]+','once','match');
        end
        if isfield( p.(fld) , 'ProgressReport' )
            h_layout.simpar(1,1).show_progress_bars = ...
                str2double( regexp( p.(fld).ProgressReport,'[0,1]','once','match' ) );
        end
        if isfield( p.(fld) , 'UpdateRate' )
            h_layout.update_rate = str2double( regexp( p.(fld).UpdateRate,'[0-9e.-]+','once','match' ) );
        end
        if isfield( p.(fld) , 'Pairing' )
            Pairing = str2double( regexp( p.(fld).Pairing,'[0-9]+','match' ) );
        end
        if isfield( p.(fld) , 'ReferenceCoord' )
            ReferenceCoord = str2double( regexp( p.(fld).ReferenceCoord,'[0-9.-]+','match' ) );
        end
        if isfield( p.(fld) , 'SplitSegments' )
            SplitSegments = str2double( regexp( p.(fld).SplitSegments,'[0-9e.-]+','match' ) );
        end
    end
end

if h_layout.simpar(1,1).show_progress_bars
    fprintf('Processing KML file ... ')
end

if ~isfield( p , 'Placemark' )
    error('QuaDRiGa:qd_layout:kml2layout:no_Placemark_found',...
        'There are no Placemark elements in the XML file.');
end

% Parse_Placemarks
tx_cnt = 0;
rx_cnt = 0;
seg_cnt = 0;
antenna_ind = [];       % Antenna index : [ file, file-id, tx, rx, freq, object-id ]
antenna_file = {};      % List of QDANT filenames
tx_track = qd_track([]);
rx_track = qd_track([]);
clear Segment
for n = 1 : numel( p.Placemark )
    
    % Parse the Placemark name (tx_,rx_,seg_)
    isTx = false;
    isRx = false;
    if numel( p.Placemark(n).Name ) > 3
        if strcmp( p.Placemark(n).Name(1:3),'tx_' )
            isTx = true;
            tx_cnt = tx_cnt + 1;
        elseif strcmp( p.Placemark(n).Name(1:3),'rx_' )
            isRx = true;
            rx_cnt = rx_cnt + 1;
        elseif strcmp( p.Placemark(n).Name(1:4),'seg_' ) && strcmp( p.Placemark(n).Type,'Point' )
            seg_cnt = seg_cnt + 1;
            Segment(seg_cnt) = p.Placemark(n);
        end
    end
    
    if isTx || isRx
        t = qd_track([]);
        t.name = p.Placemark(n).Name(4:end);
        t.positions = p.Placemark(n).Coordinates;
        
        % Set orientation to NaN, then read values from KML
        t.orientation = NaN(3,t.no_snapshots);
        
        % Antenna
        if isfield( p.Placemark(n).Description , 'Antenna' )
            val = p.Placemark(n).Description.Antenna;
        elseif isfield( p.Placemark(n).ExtendedData , 'Antenna' )
            val = p.Placemark(n).ExtendedData.Antenna;
        else
            val = [];
        end
        if ~isempty( val )
            ii = [ 0, regexp(val,','), numel(val)+1 ];      % Find ,
            for m = 1 : numel( ii )-1
                val1 = val(  ii(m)+1:ii(m+1)-1 );           % Extract single antenna
                ij = regexp(val1,':');                      % Find :
                if isempty( ij )
                    ant_fn = regexp( val1,'[a-zA-Z0-9._-]+','match','once' );
                    id = 1;
                else
                    ant_fn = regexp( val1(1:ij-1),'[a-zA-Z0-9._-]+','match','once' );
                    id = str2double( regexp( val1(ij+1:end),'[0-9]+','match','once' ) );
                end
                
                if isempty( ant_fn )                        % Antenna embedded
                    antenna_ind(end+1,:) = [0,id, tx_cnt*double(isTx), rx_cnt*double(isRx),m,0 ];
                else                                        % Not embedded
                    ik = strcmp( ant_fn, antenna_file );    % File already stored ?
                    if isempty( ik ) || ~any( ik )
                        antenna_file{end+1} = ant_fn;
                        antenna_ind(end+1,:) = [ numel( antenna_file ),id,...
                            tx_cnt*double(isTx), rx_cnt*double(isRx),m,0  ];
                    else                                    % already stored
                        antenna_ind(end+1,:) = [find(ik,1),id,...
                            tx_cnt*double(isTx), rx_cnt*double(isRx),m,0 ];
                    end
                end
            end
        end
        
        % Bank
        if isfield( p.Placemark(n).Description , 'Bank' )
            val = str2double( regexp( p.Placemark(n).Description.Bank,'[0-9e.-]+','match' ) );
        elseif isfield( p.Placemark(n).ExtendedData , 'Bank' )
            val = str2double( regexp( p.Placemark(n).ExtendedData.Bank,'[0-9e.-]+','match' ) );
        else
            val = [];
        end
        if ~isempty( val )
            t.orientation(1,:) = val*pi/180;
        end
        
        % Tilt
        if isfield( p.Placemark(n).Description , 'Tilt' )
            val = str2double( regexp( p.Placemark(n).Description.Tilt,'[0-9e.-]+','match' ) );
        elseif isfield( p.Placemark(n).ExtendedData , 'Tilt' )
            val = str2double( regexp( p.Placemark(n).ExtendedData.Tilt,'[0-9e.-]+','match' ) );
        else
            val = [];
        end
        if ~isempty( val )
            t.orientation(2,:) = val*pi/180;
        end
        
        % Heading
        if isfield( p.Placemark(n).Description , 'Heading' )
            val = str2double( regexp( p.Placemark(n).Description.Heading,'[0-9e.-]+','match' ) );
        elseif isfield( p.Placemark(n).ExtendedData , 'Heading' )
            val = str2double( regexp( p.Placemark(n).ExtendedData.Heading,'[0-9e.-]+','match' ) );
        else
            val = [];
        end
        if ~isempty( val )
            t.orientation(3,:) = val*pi/180;
        end
        
        % Time
        if isfield( p.Placemark(n).Description , 'Time' )
            val = str2double( regexp( p.Placemark(n).Description.Time,'[0-9e.-]+','match' ) );
        elseif isfield( p.Placemark(n).ExtendedData , 'Time' )
            val = str2double( regexp( p.Placemark(n).ExtendedData.Time,'[0-9e.-]+','match' ) );
        else
            val = [];
        end
        if ~isempty( val )
            val(2,:) = zeros( 1,numel(val ) );
        end
        
        % Distance
        if ~isempty( val ) && size( val,2 ) > 1
            if isfield( p.Placemark(n).Description , 'Distance' )
                val(2,:) = str2double( regexp( p.Placemark(n).Description.Distance,'[0-9e.-]+','match' ) );
            elseif isfield( p.Placemark(n).ExtendedData , 'Distance' )
                val(2,:) = str2double( regexp( p.Placemark(n).ExtendedData.Distance,'[0-9e.-]+','match' ) );
            end
        end
        if ~isempty( val )
            t.movement_profile = val;
        end
        
    end
    if isTx
        tx_track(1,tx_cnt) = t(1,1);
    elseif isRx
        rx_track(1,rx_cnt) = t(1,1);
    end
end
if numel( tx_track ) == 1
    h_layout.Ptx_track = tx_track(1,1);
else
    h_layout.Ptx_track = tx_track;
end
if numel( rx_track ) == 1
    h_layout.Prx_track = rx_track(1,1);
else
    h_layout.Prx_track = rx_track;
end

if h_layout.simpar(1,1).show_progress_bars
    fprintf([num2str(h_layout.no_tx),' Tx, '])
    fprintf([num2str(h_layout.no_rx),' Rx, '])
    fprintf([num2str(seg_cnt),' Segments\n'])
end

% Get the reference coordinate if it is not given in the extended data
if ~isempty( ForceRefCoord )
    ReferenceCoord = ForceRefCoord;
elseif ~exist('ReferenceCoord','var') || isempty( ReferenceCoord )
    coord = [];
    for n = 1 : h_layout.no_tx
        coord = [coord,tx_track(1,n).positions(1:2,:)];
    end
    for n = 1 : h_layout.no_rx
        coord = [coord,rx_track(1,n).positions(1:2,:)];
    end
    ReferenceCoord = mean(coord,2)';
end
h_layout.ReferenceCoord = ReferenceCoord;       % Hidden variable

if h_layout.simpar(1,1).show_progress_bars
    fprintf('Reference Coordinates: ');
    fprintf( '%1.6f',abs( ReferenceCoord(1) ) );
    if ReferenceCoord(1) >= 0
        fprintf( 'e, ' );
    else
        fprintf( 'w, ' );
    end
    fprintf( '%1.6f',abs( ReferenceCoord(2) ) );
    if ReferenceCoord(2) >= 0
        fprintf( 'n' );
    else
        fprintf( 's' );
    end
    if numel(ReferenceCoord) == 3 && ReferenceCoord(3) ~= 0
        fprintf( ', %1.6g m',abs( ReferenceCoord(3) ) );
    end
    fprintf('\n')
end

% Copy all track handles in the layout to a temporary variable
trk = h_layout.tx_track;
trk( 1, end+1 : end+numel( h_layout.rx_track ) ) = h_layout.rx_track;

% Process track contents
for n = 1 : numel( trk )
    % Copy the orientations. If they were not specified, the variable will contain NaNs
    orientation = trk(1,n).orientation;
    
    % Convert the positions from WGS84 to Cartesian
    lon = trk(1,n).positions(1,:);
    lat = trk(1,n).positions(2,:);
    hnn = trk(1,n).positions(3,:);
    pos = trans_global2ue( lon, lat, hnn, ReferenceCoord );
    
    trk(1,n).initial_position = pos(:,1);
    trk(1,n).positions = pos - pos(:,ones(1,size(pos,2)));
    
    % Calculate the orientations using the track headings
    trk(1,n).calc_orientation;
    
    % Overwrite the orientations from the track with the ones from the file (if specified)
    trk(1,n).orientation( ~isnan( orientation ) ) = orientation( ~isnan( orientation ) );
    
    % Fix numeric precision issue with closed tracks
    if trk(1,n).no_snapshots > 1 &&...
            get_length( trk(1,n) ) > 1e-6 &&...
            all( abs(trk(1,n).positions(:,1) - trk(1,n).positions(:,end)) < 1e-6 ) &&...
            all( abs(trk(1,n).orientation(:,1) - trk(1,n).orientation(:,end)) < 1e-6 )
        trk(1,n).positions(:,end) = trk(1,n).positions(:,1);
        trk(1,n).orientation(:,end) = trk(1,n).orientation(:,1);
    end
    
    if ~isempty( trk(1,n).movement_profile )
        len = get_length( trk(1,n) );
        if size( trk(1,n).movement_profile,2 ) == 1      % Single time value from track
            trk(1,n).movement_profile = [ 0,trk(1,n).movement_profile(1,1) ; 0,len ];
        elseif any( trk(1,n).movement_profile(2,:) > len )
            if trk(1,n).movement_profile(2,1) ~= 1 && trk(1,n).movement_profile(2,end) ~= trk(1,n).no_snapshots
                ind = trk(1,n).movement_profile(2,:) > len;
                if any( trk(1,n).movement_profile(2,ind) - len > 0.001 )
                    warning('QuaDRiGa:qd_layout:kml2layout:distance_exceeds_track_length',...
                        ['Distance exceeds track length for "',trk(1,n).name,'".']);
                end
                trk(1,n).movement_profile(2, trk(1,n).movement_profile(2,:) > len ) = len;
            end
        end
    end
end

% Process segments
trk = h_layout.rx_track;
for n = 1 : seg_cnt
    % Extract segment position in local coordinates
    pos = Segment(n).Coordinates;
    pos = trans_global2ue( pos(1), pos(2), pos(3), ReferenceCoord );
        
    % Extract scenario name string
    scenario = {};
    ii = [ 4, regexp( Segment(n).Name,':' ), numel(Segment(n).Name)+1 ];
    for m = 1 : numel( ii )-1
        scenario{m,1} = Segment(n).Name( ii(m)+1 : ii(m+1)-1 );
    end
    
    % Extract Track assignment
    trk_name = '';
    if isfield( Segment(n).Description , 'Track' )
        trk_name = Segment(n).Description.Track(4:end);
    elseif isfield( Segment(n).ExtendedData , 'Track' )
        trk_name = Segment(n).ExtendedData.Track(4:end);
    end
    
    % Find the track index
    trk_ind = 1 : numel( trk );
    if ~isempty( trk_name )
         for iT = 1 : numel( trk )
            if strcmp( trk(1,iT).name, trk_name )
                trk_ind = iT;
                break
            end
         end
    end
    
    % Extract Segment index assignment
    segment_index = [];
    if isfield( Segment(n).Description , 'Index' )
        segment_index = str2double(Segment(n).Description.Index);
    elseif isfield( Segment(n).ExtendedData , 'Index' )
        segment_index = str2double(Segment(n).ExtendedData.Index);
    end
    
    % Add the segment to the track(s)
    if ~isempty( segment_index ) && ~isempty( trk_name )
        si = trk(1,trk_ind).segment_index;                 % Copy the existing segment index
        sc = trk(1,trk_ind).scenario;                      % Copy the scenario definition
        if ~any(si==segment_index)                         % Scenario does not exist
            trk(1,trk_ind).segment_index = [ si( si<segment_index ), segment_index, si( si>segment_index )+1 ];
        end
        trk(1,trk_ind).scenario = [ sc( :,si<segment_index ), scenario , sc( :,si>segment_index ) ];
    else
        add_segment( trk(1,trk_ind), pos, scenario );
    end
end

% Check segments
for n = 1 : h_layout.no_rx
    if isempty( h_layout.rx_track(1,n).scenario{1,1} ) && h_layout.rx_track(1,n).no_segments ~= 1
        sc = h_layout.rx_track(1,n).scenario(:,2:end);
        si = h_layout.rx_track(1,n).segment_index;
        h_layout.rx_track(1,n).segment_index = si([1,3:end]);
        h_layout.rx_track(1,n).scenario = sc;
    end
end

% Split segments
if exist('SplitSegments','var') && split_seg
    if h_layout.simpar(1,1).show_progress_bars
        fprintf('Creating sub-segments ... ')
    end
    if numel( SplitSegments ) == 4
        if h_layout.simpar(1,1).show_progress_bars
            fprintf([num2str(SplitSegments(3)),' +/- ',...
                num2str(SplitSegments(4)),' m, min. ',num2str(SplitSegments(1)),' m, max. ',...
                num2str(SplitSegments(2)),' m\n'])
        end
        split_segment( h_layout.rx_track, SplitSegments(1), SplitSegments(2), SplitSegments(3), SplitSegments(4) );
        correct_overlap( h_layout.rx_track );
    else
        error('QuaDRiGa:qd_layout:kml2layout:SplitSegments','"SplitSegments" must have 4 elements.');
    end
end

% Fix track lengths
trk = h_layout.rx_track;
for n = 1 : h_layout.no_rx
    if ~isempty( trk(1,n).movement_profile )
        len = get_length( trk(1,n) );
        trk(1,n).movement_profile(2, trk(1,n).movement_profile(2,:) > len ) = len;
    end
end

% Process antennas
if h_layout.simpar(1,1).show_progress_bars
    fprintf('Processing antennas ... ')
end
if isempty( antenna_ind ) && h_layout.simpar(1,1).show_progress_bars
    fprintf('no antennas found, using omni\n')
else
    
    % Placeholder for the antennas
    a = [];
    
    % Process embedded antennas
    if isfield( p,'qdant' )
        ii = antenna_ind(:,1) == 0;
        antenna_ind(ii,6) = antenna_ind(ii,2);
        a = p.qdant;
        if h_layout.simpar(1,1).show_progress_bars
            fprintf([num2str(numel(a)),' embedded antennas, ']);
        end
    end
    if h_layout.simpar(1,1).show_progress_bars
        fprintf([num2str(numel(antenna_file)),' QDANT files\n']);
    end
    
    % Loading antennas
    for n = 1 : numel( antenna_file )
        if h_layout.simpar(1,1).show_progress_bars
            fprintf(['Loading "',antenna_file{n},'" ... ']);
        end
        a_tmp = qd_arrayant.xml_read( antenna_file{n}, [], [], [], 1 );
        if h_layout.simpar(1,1).show_progress_bars
            fprintf([num2str(numel(a_tmp)),' antenna']);
            if numel(a_tmp) == 1
                fprintf('\n');
            else
                fprintf('s\n');
            end
        end
        ii = antenna_ind(:,1) == n;
        if isempty( a )
            antenna_ind(ii,6) = antenna_ind(ii,2);
            a = a_tmp;
        else
            antenna_ind(ii,6) = antenna_ind(ii,2)+numel(a);
            a( 1, end+1 : end+numel( a_tmp ) ) = a_tmp;
        end
    end
end

% Assigning antennas to the tx and rx
if ~isempty( antenna_ind )
    if h_layout.simpar(1,1).show_progress_bars
        fprintf('Assigning antennas to transmitters and receivers\n');
    end
    n_freq = numel( h_layout.simpar(1,1).center_frequency );
    for n = 1 : h_layout.no_tx
        % Duplicate handles in existing rx_array to match the number of frequencies
        ii = find( antenna_ind(:,3) == n );
        if n_freq > 1 && numel( ii ) == n_freq && size( h_layout.tx_array,1 ) == 1
            a_omni = qd_arrayant('omni');
            h_layout.tx_array(2:n_freq,:) = a_omni( 1,ones(1,h_layout.no_tx) );
        end
        
        % Assign antennas to rx
        for m = 1 : numel( ii )
            ij = find( antenna_ind(:,3) == n & antenna_ind(:,5) == m );
            h_layout.tx_array(m,n) = a( 1 , antenna_ind(ij,6) );
        end
    end
    for n = 1 : h_layout.no_rx
        % Duplicate handles in existing rx_array to match the number of frequencies
        ii = find( antenna_ind(:,4) == n );
        if n_freq > 1 && numel( ii ) == n_freq && size( h_layout.rx_array,1 ) == 1
            a_omni = qd_arrayant('omni');
            h_layout.rx_array(2:n_freq,:) = a_omni( 1,ones(1,h_layout.no_rx) );
        end
        
        % Assign antennas to rx
        for m = 1 : numel( ii )
            ij = find( antenna_ind(:,4) == n & antenna_ind(:,5) == m );
            h_layout.rx_array(m,n) = a( 1 , antenna_ind(ij,6) );
        end
    end
    if numel( h_layout.tx_array ) == 1  % Fix for octave
        h_layout.tx_array = h_layout.tx_array(1,1);
    end
    if numel( h_layout.rx_array ) == 1
        h_layout.rx_array = h_layout.rx_array(1,1);
    end
end

% Apply the pairing
if ~exist('Pairing','var') || isempty( Pairing )
    h_layout.set_pairing;
else
    h_layout.pairing = reshape( Pairing,2,[] );
end

end
