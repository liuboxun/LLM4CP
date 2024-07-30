function h_qd_satellite = read_tle( filename )
%READ_TLE Reads orbital elements from two-line element sets
%
% Calling object:
%   None (static method)
%
% Description:
%   A two-line element set (TLE) is a data format encoding a list of orbital elements of an Earth-
%   orbiting object for a given point in time, the epoch. The format was originally intended for
%   punch cards, encoding a set of elements on two standard 80-column cards. This format was
%   eventually replaced by text files with each set of elements written to two 70-column ASCII
%   lines. The United States Air Force tracks all detectable objects in Earth orbit, creating a
%   corresponding TLE for each object, and makes publicly available TLEs for many of the space
%   objects on the website Space Track (http://celestrak.com). The TLE format is a de facto
%   standard for distribution of an Earth-orbiting object's orbital elements.
%
% Input:
%   filename
%   Filename of the text-file containing the TLE data.
%
% Output:
%   h_qd_satellite
%   Handle to the created 'qd_satellite' object.
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

fid = fopen(filename);

L0 = fgetl(fid);
L1 = fgetl(fid);
L2 = fgetl(fid);

n = 0;
while ischar(L2)
    n = n+1;
    
    % Check checksum
    tmp = regexp(L1(1:68), {'[1-9]', '[-]'});
    assert(mod(sum(L1(tmp{1})-48)+numel(tmp{2}), 10) == L1(69)-48, 'L1 checksum failed');
    tmp = regexp(L2(1:68), {'[1-9]', '[-]'});
    assert(mod(sum(L2(tmp{1})-48)+numel(tmp{2}), 10) == L2(69)-48, 'L2 checksum failed');
    
    % parse title line
    % Field	Columns	Content	Example
    % 1	01–24	Satellite name	ISS (ZARYA)
    satellite_name{n} = strtrim(L0);
    
    % parse line 1
    % Field	Columns	Content	Example
    % 1	01–01	Line number	1
    % 2	03–07	Satellite number	25544
    satellite_number(n) = str2double(L1(3:7));
    
    % 3	08–08	Classification (U=Unclassified)	U
    classification{n} = L1(8);
    
    % 4	10–11	International Designator (Last two digits of launch year)	98
    ID_launch_year(n) = str2double(L1(10:11));
    
    % 5	12–14	International Designator (Launch number of the year)	067
    ID_launch_number(n) = str2double(L1(12:14));
    
    % 6	15–17	International Designator (piece of the launch)	A
    ID_launch_piece{n} = L1(15:17);
    
    % 7	19–20	UTC Epoch Year (last two digits of year)	08
    epoch_year(n) = str2double(L1(19:20));
    if epoch_year(n) >= 57
        epoch_year(n) = epoch_year(n) + 1900;
    else
        epoch_year(n) = epoch_year(n) + 2000;
    end
    
    % 8	21–32	UTC Epoch (day of the year and fractional portion of the day)	264.51782528
    epoch_day(n) = str2double(L1(21:32));
    
    % 9	34–43	First Time Derivative of the Mean Motion divided by two [11]	−.00002182
    mean_motion_FTD(n) = str2double(L1(34:43))*2;
    
    % 10	45–52	Second Time Derivative of Mean Motion divided by six (decimal point assumed)	00000-0
    mean_motion_STD(n) = (str2double(L1(45:50))*1e-5*10^str2double(L1(51:52)))*6;
    
    % 11	54–61	BSTAR drag term (decimal point assumed) [11]	-11606-4
    BSTAR_drag(n) = str2double(L1(54:59))*1e-5*10^str2double(L1(60:61));
    
    % 12	63–63	The number 0 (originally this should have been "Ephemeris type")	0
        
    % 13	65–68	Element set number. Incremented when a new TLE is generated for this object.[11]	292
    TLE_number(n) = str2double(L1(65:68));
    
    % parse line 2
    % Field	Columns	Content	Example
    % 1	01–01	Line number	2
    % 2	03–07	Satellite number	25544
    satellite_number(n) = str2double(L2(3:7));
    
    % 3	09–16	Inclination (degrees)	51.6416
    inclination(n) = str2double(L2(9:16));
    
    % 4	18–25	Right ascension of the ascending node (degrees)	247.4627
    right_ascending_node(n) = str2double(L2(18:25));
    
    % 5	27–33	Eccentricity (decimal point assumed)	0006703
    eccentricity(n) = str2double(L2(27:33))*1e-7;
    
    % 6	35–42	Argument of perigee (degrees)	130.5360
    argument_of_perigee(n) = str2double(L2(35:42));
    
    % 7	44–51	Mean Anomaly (degrees)	325.0288
    mean_anomaly(n) = str2double(L2(44:51));
    
    % 8	53–63	Mean Motion (revolutions per day)	15.72125391
    mean_motion(n) = str2double(L2(53:63));
    
    % 9	64–68	Revolution number at epoch (revolutions)	56353
    revolutions(n) = str2double(L2(64:68));
    
    %fprintf('%s\n', L0);
    L0 = fgetl(fid);
    L1 = fgetl(fid);
    L2 = fgetl(fid);
end
fclose(fid);

% (86400 sec/day) / (revolutions/day)
period = 86400./mean_motion;

% semi-major axis in km
semimajoraxis = ((period./(2*pi)).^2.*qd_satellite.mu).^(1/3);

% From M calculate the eccentric anomaly E using an iteration for elliptical orbits
e = eccentricity;                                       % The orbital eccentricity
M = mean_anomaly;                                       % Mean Anomaly (degrees)
E = M;                                                  % Eccentric anomaly for circular orbits at timepoint t in [degrees]
upd = e > 1e-12;                                        % Determine which values must be solved by iteration

% Solve "M = E - e * sind( E )" for elliptical orbits
En = E;                                                 % Initial En
Mn = En - e .* sind( En ) .* 180/pi;                    % Updated M
dM = mod( M - Mn, 360 );                                % The angle difference between Mn and M in range [0,180]
dM( dM > 180 ) = 360 - dM( dM > 180 );
dMn = dM;                                               % The initial difference
dE = 10*ones(1,n);                                      % Initial step size in degrees

while any( upd(:) )                                     % Iterate until M == Mn
    En(upd) = E(upd) + dE(upd);                         % Updated En
    Mn(upd) = En(upd) - e(upd) .* sind( En(upd) ) .* 180/pi;  % Updated Mn
    
    dM(upd) = M(upd) - Mn(upd);
    dM(upd) = mod( M(upd) - Mn(upd), 360 );             % The angle difference
    dM( dM > 180 & upd ) = 360 - dM( dM > 180 & upd );
    
    ii = dM < dMn & upd;                                % Estimate improved
    E(ii) = En(ii);                                     % Update E
    dMn(ii) = dM(ii);                                   % Update dMn

    ii = dM > dMn & upd;                                % Estimate got worse
    dE(ii) = -0.382 * dE(ii);                           % Change step size and direction

    upd(upd) = dMn(upd) > 1e-7;                         % Continue until M ~ Mn
end
E = mod( E +180,360)-180;     
% From E calculate the true anomaly v
v = 2*atand(sqrt((1+e)./(1-e)).*tand(E/2));
true_anomaly = mod( v +180,360)-180;       

% Calculate the for each satellite
epoch = zeros(1,numel(epoch_year) );
for n = 1 : numel(epoch_year)
    epoch(n) = datenum( [num2str(epoch_year(n),'%04d'),'-01-01'],'yyyy-mm-dd' ); % Days since 1/1/0000
end
epoch = epoch + epoch_day - 1;

common_epoch = max(epoch);                              % Common epoch for all satellites
upd = common_epoch - epoch > 0.05;                      % Satellite orbits that must be updated

% Update satellite orbital position to match common epoch
if any( upd )
    for n = find( upd )
        h_qd_satellite = qd_satellite('custom', semimajoraxis(n), eccentricity(n), inclination(n),...
            right_ascending_node(n), argument_of_perigee(n), true_anomaly(n));
        h_qd_satellite.epoch = epoch(n);
        [ ~, ~, ~, ~, ~, ~, ~, right_ascending_node(n), argument_of_perigee(n), true_anomaly(n) ] =...
            h_qd_satellite.orbit_predictor( (common_epoch - epoch(n))*86400 );
    end
end

% Create satellite constellation object
h_qd_satellite = qd_satellite('custom', semimajoraxis, eccentricity, inclination, right_ascending_node, argument_of_perigee, true_anomaly);
h_qd_satellite.epoch = common_epoch;

ii = regexp(filename,'\.');
if numel( ii ) == 1
    h_qd_satellite.name = filename(1:ii-1);
else
    h_qd_satellite.name = filename;
end

% Set satellite name
for i_sat = 1 : h_qd_satellite.n_satellites
    tmp = satellite_name{i_sat};
    tmp = regexprep( tmp,'\W',' ' );
    tmp = regexprep( tmp,'[\s]*',' ' );
    if strcmp( tmp(1),' ' ); tmp = tmp(2:end); end
    if strcmp( tmp(end),' ' ); tmp = tmp(1:end-1); end
    tmp = regexprep( tmp,' ','-' );
    h_qd_satellite.sat_name{1,i_sat} = tmp;
end

end
