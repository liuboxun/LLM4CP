function h_builder = gen_cdl_model( cdl_model, center_frequency, mobile_speed, duration,...
    ds, kf, asd, asa, esd, esa, cas_factor, sample_density )
%GEN_CDL_MODEL Generates channel models for link-level evaluations (TDL and CDL)
%
% Calling object:
%   None (static method)
%
% Description:
%   This method creates a 'qd_builder' object that can be used to generate 3GPP compliant
%   standardized channels for link-level evaluations. The models are defined for the full frequency
%   range from 0.5 GHz to 100 GHz with a maximum bandwidth of 2 GHz. All models can be scaled in
%   delay. 
%
%   Clustered Delay Line (CDL) models define arrival and departure directions per cluster.
%   The angle values of CDL models are fixed, which is not very suitable for MIMO simulations for
%   several reasons; The PMI statistics can become biased, and a fixed precoder may perform better
%   than open-loop and on par with closed-loop or reciprocity beamforming. Furthermore, a CDL only
%   represents a single channel realization. The predefined angle values in the CDL models can be
%   generalized by introducing angular translation and scaling. Scaling can be achieved by
%   providing custom angular spread values 'asd', 'asa', 'esd', 'esa'. Translation can be achieved
%   by changing the position of the terminal in the generated 'qd_builder' object. Spatial filter
%   (antenna patterns) can be applied on a CDL to derive a TDL for evaluating directional
%   algorithms. This is done by providing either a 'qd_builder.tx_array' or 'qd_builder.rx_array'
%   object or both. Array broadside direction must face east. 
%
%   Tapped Delay Line (TDL) models are
%   used tor simplified evaluations, e.g., for non-MIMO evaluations. The Doppler spectrum for each
%   tap is characterized by a classical (Jakes) spectrum shape and a maximum Doppler shift fD =
%   v/lambda. Due to the presence of a LOS path, the first tap in NR-TDL-D and NR-TDL-E follows a
%   Ricean fading distribution. For those taps the Doppler spectrum additionally contains a peak at
%   the Doppler shift fS = -0.7 fD with an amplitude such that the resulting fading distribution has
%   the specified K-factor. This is equivalent to a MT moving away from the BS at a 45 degree
%   angle.  The following list of models is currently supported:
%
% Models:
%   EPA-TDL
%   3GPP TS 36.104 V16.5.0 p250, Table B.2-1 Extended Pedestrian A model
%   
%   EVA-TDL
%   3GPP TS 36.104 V16.5.0 p250, Table B.2-2 Extended Vehicular A model
%   
%   ETU-TDL
%   3GPP TS 36.104 V16.5.0 p250, Table B.2-3 Extended Typical Urban model
%   
%   NR-CDL-A
%   3GPP TR 38.901 V16.1.0 (2019-12) p74, Table 7.7.1-1. CDL-Model for NLOS
%
%   NR-CDL-B
%   3GPP TR 38.901 V16.1.0 (2019-12) p75, Table 7.7.1-2. CDL-Model for NLOS
%
%   NR-CDL-C
%   3GPP TR 38.901 V16.1.0 (2019-12) p75, Table 7.7.1-3. CDL-Model for NLOS
%
%   NR-CDL-D
%   3GPP TR 38.901 V16.1.0 (2019-12) p76, Table 7.7.1-4. CDL-Model for LOS
%
%   NR-CDL-E
%   3GPP TR 38.901 V16.1.0 (2019-12) p76, Table 7.7.1-5. CDL-Model for LOS
%
%   NR-TDL-A
%   3GPP TR 38.901 V16.1.0 (2019-12) p77, Table 7.7.2-1. TDL-Model for NLOS
%
%   NR-TDL-B
%   3GPP TR 38.901 V16.1.0 (2019-12) p77, Table 7.7.2-2. TDL-Model for NLOS
%
%   NR-TDL-C
%   3GPP TR 38.901 V16.1.0 (2019-12) p78, Table 7.7.2-3. TDL-Model for NLOS
%
%   NR-TDL-D
%   3GPP TR 38.901 V16.1.0 (2019-12) p78, Table 7.7.2-4. TDL-Model for LOS
%
%   NR-TDL-E
%   3GPP TR 38.901 V16.1.0 (2019-12) p79, Table 7.7.2-5. TDL-Model for LOS
%
%   V2X-CDL-UrbanLOS
%   3GPP TR 37.885 V15.3.0 (2019-06) p32, Table 6.2.3.1-1. CDL model for Urban LOS V2X channel
%
%   V2X-CDL-UrbanNLOS
%   3GPP TR 37.885 V15.3.0 (2019-06) p32, Table 6.2.3.1-2. CDL model for Urban NLOS V2X channel
%   
%   V2X-CDL-UrbanNLOSv
%   3GPP TR 37.885 V15.3.0 (2019-06) p33, Table 6.2.3.1-3. CDL model for Urban NLOSv V2X channel
%   
%   V2X-CDL-HighwayLOS
%   3GPP TR 37.885 V15.3.0 (2019-06) p33, Table 6.2.3.1-4. CDL model for Highway LOS V2X channel
%   
%   V2X-CDL-HighwayNLOSv
%   3GPP TR 37.885 V15.3.0 (2019-06) p34, Table 6.2.3.1-5. CDL model for Highway NLOSv V2X channel
%   
%
% Input:
%   cdl_model
%   String selecting the TDL/CDL model
%
%   center_frequency
%   The center frequency in [Hz]
%
%   mobile_speed
%   The mobile speed in [m/s]
%
%   duration
%   The channel observation time in [s]
%
%   ds
%   The desired RMS delay spread in [ns] (optional)
%
%   kf
%   The Ricean K-Factor in [dB] (optional)
%
%   asd
%   The azimuth-of-departure angular spread in [deg] (optional)
%
%   asa
%   The azimuth-of-arrival angular spread in [deg] (optional)
%
%   esd
%   The elevation-of-departure angular spread in [deg] (optional)
%
%   esa
%   The elevation-of-arrival angular spread in [deg] (optional)
%
%   cas_factor
%   Scaling factor for the per-cluster angular spreads (optional, default = 1)
%
% Output:
%   h_builder
%   Handle to the created 'qd_builder' object
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

% Check cdl model
supported_types = {'EPA-TDL','EVA-TDL','ETU-TDL',...
    'NR-CDL-A','NR-CDL-B','NR-CDL-C','NR-CDL-D','NR-CDL-E',...
    'NR-TDL-A','NR-TDL-B','NR-TDL-C','NR-TDL-D','NR-TDL-E',...
    'V2X-CDL-UrbanLOS','V2X-CDL-UrbanNLOS','V2X-CDL-UrbanNLOSv','V2X-CDL-HighwayLOS','V2X-CDL-HighwayNLOSv'};
if ~exist( 'cdl_model' , 'var' ) || isempty( cdl_model )
    error('QuaDRiGa:qd_builder:gen_cdl_model','CDL model type not given.');
else
    cdl_model = upper( cdl_model );
    if ~( ischar(cdl_model) && any( strcmpi(cdl_model,supported_types)) )
        str = ['CDL model not found. Supported are: ',sprintf('%s, ',supported_types{:})];
        error('QuaDRiGa:qd_builder:gen_cdl_model',str(1:end-2))
    end
end

% Check center frequency
if ~exist( 'center_frequency', 'var' ) || isempty( center_frequency )
    center_frequency = qd_simulation_parameters.speed_of_light;  % 300 MHz, wavelength = 1 m
end

% Check mobile speed and duration
if ~exist( 'mobile_speed', 'var' ) || isempty( mobile_speed )
    mobile_speed = 0;  % Static channel
end
if ~exist( 'duration', 'var' ) || isempty( duration )
    duration = 0;  % Static channel
end

% Check parameter scaling variables
as_desired = NaN(4,1);
if ~exist( 'ds', 'var' ) || isempty(ds)
    ds = []; % Do not scale
end
if exist( 'asd', 'var' ) && ~isempty(asd)
    as_desired(1,1) = asd*pi/180;
end
if exist( 'asa', 'var' ) && ~isempty(asa)
    as_desired(2,1) = asa*pi/180;
end
if exist( 'esd', 'var' ) && ~isempty(esd)
    as_desired(3,1) = esd*pi/180;
end
if exist( 'esa', 'var' ) && ~isempty(esa)
    as_desired(4) = esa*pi/180;
end
if ~exist( 'kf', 'var' ) || isempty( kf )
    kf = []; % Do not scale
end
if ~exist( 'cas_factor', 'var' ) || isempty( cas_factor )
    cas_factor = 1; % Do not scale
end
if ~exist( 'sample_density', 'var' ) || isempty( sample_density )
    sample_density = 2.5; % Default value
end

% Set path parameters
switch cdl_model
    case {'EPA-TDL'}
        % See 3GPP TS 36.104 V16.5.0 p250, Table B.2-1 Extended Pedestrian A model
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -Inf, 0, -1, -2, -3, -8, -17.2, -20.8 ];
        taus = [ 0, 0, 30, 70, 90, 110, 190, 410 ];
        
    case {'EVA-TDL'}
        % See 3GPP TS 36.104 V16.5.0 p250, Table B.2-2 Extended Vehicular A model
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -Inf, 0, -1.5, -1.4, -3.6, -0.6, -9.1, -7, -12, -16.9 ];
        taus = [ 0, 0, 30, 150, 310, 370, 710, 1090, 1730, 2510 ];
        
    case {'ETU-TDL'}
        % See 3GPP TS 36.104 V16.5.0 p250, Table B.2-3 Extended Typical Urban model
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -1, -1, -1, 0, 0, 0, -3, -5, -7 ];
        taus = [ 0, 50, 120, 200, 230, 500, 1600, 2300, 5000  ];
        
    case {'NR-CDL-A','NR-TDL-A'}
        % See 3GPP TR 38.901 V16.1.0 (2019-12) p74, Table 7.7.1-1
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -Inf, -13.4 , 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4,...
            -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7 ];
        taus = [ 0, 0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375,...
            1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, ...
            5.3043, 9.6586 ];
        aod = [ 0, -178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83, 134.8, -153,...
            -172, -129.9, -136, 165.4, 148.4, 132.7, -118.6, -154.1, 126.5, -56.2 ];
        aoa = [ -180, 51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.8, -41.9, 94.2, 51.9, -115.9, ...
            26.6, 76.6, -7, -23, -47.2, 110.4, 144.5, 155.3, 102, -151.8, 55.2 ];
        zod = [ 98.5, 50.2, 93.2, 93.2, 93.2, 122, 122, 122, 150.2, 55.2, 26.4, 126.4, 171.6, 151.4, 157.2,...
            47.2, 40.4, 43.3, 161.8, 10.8, 16.7, 171.7, 22.7, 144.9 ];
        zoa = [ 81.5, 125.4, 91.3, 91.3, 91.3, 94, 94, 94, 47.1, 56, 30.1, 58.8, 26, 49.2, 143.1, 117.4,...
            122.7, 123.2, 32.6, 27.2, 15.2, 146, 150.7, 156.1 ];
        xpr = 10;
        cAS = [ 5, 11, 3, 3 ];  % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'NR-CDL-B','NR-TDL-B'}
        % See 3GPP TR 38.901 V16.1.0 (2019-12) p75, Table 7.7.1-2
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -Inf, 0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3, -8.9, -9, -4.8, -5.7, ...
            -7.5, -1.9, -7.6, -12.2, -9.8, -11.4, -14.9, -9.2, -11.3 ];
        taus = [ 0, 0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697,...
            0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187,...
            4.1067, 4.2790, 4.7834 ];
        aod = [ 0, 9.3, 9.3, 9.3, -34.1, -65.4, -11.4, -11.4, -11.4, -67.2, 52.5, -72, 74.3, -52.2, ...
            -50.5, 61.4, 30.6, -72.5, -90.6, -77.6, -82.6, -103.6, 75.6, -77.6 ];
        aoa = [ -180, -173.3, -173.3, -173.3, 125.5, -88.0, 155.1, 155.1, 155.1, -89.8, 132.1, ...
            -83.6, 95.3, 103.7, -87.8, -92.5, -139.1, -90.6, 58.6, -79.0, 65.8, 52.7, 88.7, -60.4 ];
        zod = [ 98.5, 105.8, 105.8, 105.8, 115.3, 119.3, 103.2, 103.2, 103.2, 118.2, 102.0, 100.4, ...
            98.3, 103.4, 102.5, 101.4, 103.0, 100.0, 115.2, 100.5, 119.6, 118.7, 117.8, 115.7 ];
        zoa = [ 81.5, 78.9, 78.9, 78.9, 63.3, 59.9, 67.5, 67.5, 67.5, 82.6, 66.3, 61.6, 58.0, 78.2, ...
            82.0, 62.4, 78.0, 60.9, 82.9, 60.8, 57.3, 59.9, 60.1, 62.3 ];
        xpr = 8;
        cAS = [ 10, 22, 3, 7 ];  % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'NR-CDL-C','NR-TDL-C'}
        % See 3GPP TR 38.901 V16.1.0 (2019-12) p75, Table 7.7.1-3
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -Inf, -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, ...
            -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8 ];
        taus = [ 0, 0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, ...
            0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, ...
            7.0427, 8.6523 ];
        aod = [ 0, -46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3, 73.1, -64.5, 80.2, -97.1, -55.3, ...
            -64.3, -78.5, 102.7, 99.2, 88.8, -101.9, 92.2, 93.3, 106.6, 119.5, -123.8 ];
        aoa = [ -180, -101, 120, 120, 120, -127.5, 170.4, 170.4, 170.4, 55.4, 66.5, -48.1, 46.9, ...
            68.1, -68.7, 81.5, 30.7, -16.4, 3.8, -13.7, 9.7, 5.6, 0.7, -21.9, 33.6 ];
        zod = [ 98.5, 97.2, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 105.2, 95.3, 106.1, 93.5, 103.7, ...
            104.2, 93.0, 104.2, 94.9, 93.1, 92.2, 106.7, 93.0, 92.9, 105.2, 107.8 ];
        zoa = [ 81.5, 87.6, 72.1, 72.1, 72.1, 70.1, 75.3, 75.3, 75.3, 67.4, 63.8, 71.4, 60.5, 90.6, ...
            60.1, 61.0, 100.7, 62.3, 66.7, 52.9, 61.8, 51.9, 61.7, 58, 57 ];
        xpr = 7;
        cAS = [ 2, 15, 3, 7 ];  % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'NR-CDL-D','NR-TDL-D'}
        % See 3GPP TR 38.901 V16.1.0 (2019-12) p76, Table 7.7.1-4
        pow = [ -0.2, -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30.0, -27.7 ];
        taus = [ 0, 0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525 ];
        aod = [ 0, 0, 89.2, 89.2, 89.2, 13, 13, 13, 34.6, -64.5, -32.9, 52.6, -132.1, 77.2 ];
        aoa = [ -180, -180, 89.2, 89.2, 89.2, 163, 163, 163, -137, 74.5, 127.7, -119.6, -9.1, -83.8 ];
        zod = [ 98.5, 98.5, 85.5, 85.5, 85.5, 97.5, 97.5, 97.5, 98.5, 88.4, 91.3, 103.8, 80.3, 86.5 ];
        zoa = [ 81.5, 81.5, 86.9, 86.9, 86.9, 79.4, 79.4, 79.4, 78.2, 73.6, 78.3, 87, 70.6, 72.9 ];
        xpr = 11;
        cAS = [ 5, 8, 3, 3 ];  % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'Laplacian';
        
    case {'NR-CDL-E','NR-TDL-E'}
        % See 3GPP TR 38.901 V16.1.0 (2019-12) p76, Table 7.7.1-5
        pow = [ -0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6, -22.3,...
            -25.6, -20.2, -29.8, -29.2 ];
        taus = [ 0, 0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589, 2.6426, ...
            3.7136, 5.4524, 12.0034, 20.6419 ];
        aod = [ 0, 0, 57.5, 57.5, 57.5, -20.1, 16.2, 9.3, 9.3, 9.3, 19, 32.7, 0.5, 55.9, 57.6 ];
        aoa = [ -180, -180, 18.2, 18.2, 18.2, 101.8, 112.9, -155.5, -155.5, -155.5, -143.3, -94.7,...
            147, -36.2, -26 ];
        zod = [ 99.6, 99.6, 104.2, 104.2, 104.2, 99.4, 100.8, 98.8, 98.8, 98.8, 100.8, 96.4, 98.9,...
            95.6, 104.6 ];
        zoa = [ 80.4, 80.4, 80.4, 80.4, 80.4, 80.8, 86.3, 82.7, 82.7, 82.7, 82.9, 88, 81, 88.6, 78.3 ];
        xpr = 8;
        cAS = [ 5, 11, 3, 7 ];  % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'Laplacian';
        
    case {'V2X-CDL-URBANLOS'}
        % 3GPP TR 37.885 V15.3.0 (2019-06) p32, Table 6.2.3.1-1. CDL model for Urban LOS V2X channel
        pow = [ -0.12, -15.52, -17.7, -19.5, -15.9, -14.6, -9.1, -11.3, -13.1, -19.3, -20.0, -16.3,...
            -17.9, -25.4, -26.9, -22.9, -25.3 ];
        taus = [ 0, 0, 6.4, 12.8, 11.0793, 21.9085, 29.6768, 36.0768, 42.4768, 68.4085, 82.2944, ...
            115.4173, 143.2963, 146.4136, 183.1925, 214.1501, 326.7825 ];
        aod = [ 0, 0, 0, 0, 93.2, 85.4, -49.9, -49.9, -49.9, -97.1, 108.5, -90.7, 105.5, 127.1, ...
            127, -101.5, 125.2 ];
        aoa = [ -180, -180, -180, -180, -51.8, -51.9, 96.8, 96.8, 96.8, -37.5, -45.7, 57.3, -42.1, ...
            -17.6, 18.6, 20.6, -19.1 ];
        zod = [ 90, 90, 90, 90, 75.1, 76.6, 84.8, 84.8, 84.8, 107.3, 107.7, 104, 107, 68.4, 67.2, 69.4, 112 ];
        zoa = [ 90, 90, 90, 90, 57.8, 119.7, 100, 100, 100, 130.3, 130.3, 58.1, 52.5, 36.9, 145.1, 42.8, 141.6 ];
        xpr = 9;
        cAS = [ 3, 17, 7, 7 ]; % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'V2X-CDL-URBANNLOS'}
        % 3GPP TR 37.885 V15.3.0 (2019-06) p32, Table 6.2.3.1-2. CDL model for Urban NLOS V2X channel
        % Additional zero-power (virtual) LOS component added for QuaDRiGa
        pow = [ -Inf, -4.8, -0.8, -3, -4.8, 0, -0.8, -0.9, -0.8, -3, -4.8, -6.3, -4, -8.1, -8, -7, -8.3,...
            -1.7, -7.6, -16.2, -4.2, -18.2, -21.8, -19.9 ];
        taus = [ 0, 0, 6.46631, 11.6926, 16.9189, 19.4978, 20.6484, 38.7458, 48.7547, 53.981, 59.2073,...
            62.1898, 68.7158, 70.3389, 74.461, 105.954, 117.904, 137.072, 210.522, 218.823, 232.216,...
            289.654, 357.791, 380.239 ];
        aod = [ 0, -53, -2.7, -2.7, -2.7, -30.3, -28, 28.3, 0.5, 0.5, 0.5, -80, 60, -75.7, -76.8, 59.4,...
            72.6, 42.3, 57.3, -93.9, -37.8, 106.7, 107.5, -95 ];
        aoa = [ -180, -36, -162.5, -162.5, -162.5, -87, -79.7, -88.6, 143.6, 143.6, 143.6, 6.3, -58.2, -19.9,...
            23, -28.7, -5.4, -82.8, -22.4, 56.2, 32.9, -57, -103.3, 68.4 ];
        zod = [ 90, 79.7, 89.3, 89.3, 89.3, 93.6, 85.3, 96, 91.3, 91.3, 91.3, 79.8, 81.1, 102.9, 103.2,...
            77.5, 77.2, 95.5, 100.5, 111.2, 80, 64.3, 119.9, 117 ];
        zoa = [ 90, 12.5, 73.3, 73.3, 73.3, 73.7, 121.2, 119.6, 92.9, 92.9, 92.9, 175.1, 29.4, 159, 168.3,...
            6.9, 168, 134.6, 21.4, 84.2, 171, 110.2, 39.6, 127.9 ];
        xpr = 8;
        cAS = [ 10, 22, 7, 7 ]; % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'V2X-CDL-URBANNLOSV'}
        % 3GPP TR 37.885 V15.3.0 (2019-06) p33, Table 6.2.3.1-3. CDL model for Urban NLOSv V2X channel
        pow = [ -0.14, -14.93, -8.9, -11.2, -12.9, -17.9, -14.8, -11.9, -10.2, -12.5, -14.2, -11.1,...
            -15.5, -13.8, -12.5, -20.2, -11.7, -19, -17.1, -17.5, -18.1, -22.2, -16.4, -19.8 ];
        taus = [ 0, 0, 20.1752, 34.2552, 48.3352, 34.3633, 37.1866, 52.1209, 52.7982, 66.8782,...
            80.9582, 53.2168, 53.2285, 55.2847, 65.8409, 79.0272, 90.9391, 91.0347, 105.476, 118.795,...
            166.128, 253.705, 293.544, 471.377 ];
        aod = [ 0, 0, 36, 36, 36, -45.7, 60.7, 53.6, -34.5, -34.5, -34.5, 48.4, -45.8, 56, 55.7,...
            -48.9, 51.1, 62.7, -43, 62.4, -50.6, -57, -43.1, -50.1 ];
        aoa = [ -180, -180, 138.4, 138.4, 138.4, -79.9, -85.1, -100.6, -119.5, -119.5, -119.5,...
            -103.5, 92.5, 80.7, 100.7, -69.4, 101.2, 69, 86.5, 91.5, -76.6, -68.1, 82.7, -61.8 ];
        zod = [ 90, 90, 84.1, 84.1, 84.1, 74.2, 76.4, 77.3, 97.4, 97.4, 97.4, 99.7, 105.6, 76.6,...
            76.9, 71.3, 77.9, 71.6, 73.9, 72.4, 72.7, 110.7, 104.6, 108.6 ];
        zoa = [ 90, 90, 81.1, 81.1, 81.1, 118.1, 117.3, 71.3, 103, 103, 103, 108.7, 63.7, 67,...
            109.3, 125.9, 108.3, 58.4, 119.8, 119.9, 120.3, 54.1, 62.1, 56.4 ];
        xpr = 8;
        cAS = [ 10, 22, 7, 7 ]; % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'V2X-CDL-HIGHWAYLOS'}
        % 3GPP TR 37.885 V15.3.0 (2019-06) p33, Table 6.2.3.1-4. CDL model for Highway LOS V2X channel
        pow = [ -0.07, -18.08, -19.9, -13.9, -16.2, -17.9, -14.5, -21.3, -18.7, -14.9, -16.2, -18.5,...
            -20.2, -17.1, -13.8, -28.4, -27.4 ];
        taus = [ 0, 0, 2.1109, 2.9528, 17.0328, 31.1128, 9.1629, 10.6761, 11.0257, 18.5723, 19.8875,...
            33.9675, 48.0475, 25.737, 36.2683, 66.7093, 139.97 ];
        aod = [ 0, 0, 63.4, 50, 50, 50, 55.2, -62.6, 56, 53.3, -51.1, -51.1, -51.1, -56.1, 58.4, 74.7, -71.5 ];
        aoa = [ -180, -180, -80.2, 98.6, 98.6, 98.6, 73.1, -64.3, 65.7, -90.9, 84.5, 84.5, 84.5, 71.3, -81.5, 41.4, -42.6 ];
        zod = [ 90, 90, 83.8, 86.9, 86.9, 86.9, 85.1, 97.3, 96.3, 85.3, 94.4, 94.4, 94.4, 95.5, 86.2, 81.1, 99.4 ];
        zoa = [ 90, 90, 75, 98.4, 98.4, 98.4, 78.1, 73.7, 105.4, 79.1, 79.4, 79.4, 79.4, 77.4, 80.4, 68.1, 111.8 ];
        xpr = 9;
        cAS = [ 3, 17, 7, 7 ]; % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    case {'V2X-CDL-HIGHWAYNLOSV'}
        % 3GPP TR 37.885 V15.3.0 (2019-06) p34, Table 6.2.3.1-5. CDL model for Highway NLOSv V2X channel
        pow = [  -0.2927, -11.8594, -12.509, -14.7274, -16.4884, -11.8681, -11.3289, -17.8834, -9.9943,...
            -12.7302, -13.912, -16.8781, -12.9647, -15.1832, -16.9441, -10.7858, -12.3875, -17.3827,...
            -14.7254, -13.5863, -20.908, -15.5653, -19.7098, -24.7824 ];
        taus = [ 0, 0, 5.5956, 19.6756, 33.7556, 21.7591, 21.8113, 27.2207, 39.3242, 51.0232, 51.4828,...
            53.3659, 65.1775, 79.2575, 93.3375, 67.9841, 70.7561, 73.998, 75.8665, 84.3678, 90.1654,...
            91.6154, 142.931, 158.434 ];
        aod = [ 0, 0, -52.3, -52.3, -52.3, 66.4, -46.7, 89.8, -56.8, 75.9, 85.4, 88.9, -46.2, -46.2,...
            -46.2, -50.9, -54.3, 88.3, 78, 73.5, -69.7, -62.1, -70.3, -84.5 ];
        aoa = [ -180, -180, -120, -120, -120, -114.6, 96.5, 77, -124.7, 93.1, 88.2, 80.7, 94.4, 94.4,...
            94.4, 98.1, -99.3, 66.8, 91.5, -108.4, -89.6, 84.4, -81.8, -69.6 ];
        zod = [ 90, 90, 99.2, 99.2, 99.2, 77.2, 78.6, 106.8, 81.2, 102.1, 103.7, 74, 100, 100, 100,...
            100.2, 77.7, 73.8, 103.8, 104.8, 69.4, 103.2, 109.1, 113.8 ];
        zoa = [ 90, 90, 61.5, 61.5, 61.5, 54.8, 56.1, 34.8, 120.7, 119.9, 48.7, 136.9, 56.9, 56.9,...
            56.9, 121.1, 121.6, 141.6, 131.1, 51.1, 147.1, 46.7, 32.2, 157.3 ];
        xpr = 8;
        cAS = [ 10, 22, 7, 7 ]; % [ cASD, cASA, cZSD, cZSA ]
        SubpathMethod = 'legacy';
        
    otherwise
        error('QuaDRiGa:qd_builder:gen_cdl_model','CDL model parameters not defined.');
end

if any( strcmpi(cdl_model,{'EPA-TDL','EVA-TDL','ETU-TDL','NR-TDL-A','NR-TDL-B','NR-TDL-C','NR-TDL-D','NR-TDL-E'}))
    use_tdl_model = true;
    if ~exist('aod','var')
        aod = [0, 360*rand(1,numel(pow)-1)-180];
    end
    if ~exist('aoa','var')
        aoa = [-180, 360*rand(1,numel(pow)-1)-180];
    end
    if ~exist('zod','var')
        zod = [90, 5*rand(1,numel(pow)-1)+90];
    end
    if ~exist('zoa','var')
        zoa = [90, 5*rand(1,numel(pow)-1)+90];
    end
    if ~exist('xpr','var')
        xpr = 100;
    end
    cAS = ones(1,4);
else
    use_tdl_model = false;
end

% Transform all angles to [rad] and ZOA/D to EOA/D
ang = [ aod; aoa; 90-zod; 90-zoa ]*pi/180;

% Apply scaling factor to cluster-angular-spread
cAS = cAS * cas_factor;

% Calculate linear power values
powL = 10.^(0.1*pow);

% Adjust the KF
if ~isempty( kf )
    p_total = sum( powL );
    powL(1) = sum(powL(2:end)) * 10^(0.1*kf);
    powL = powL .* p_total/sum(powL);
    taus = taus ./ qf.calc_delay_spread( taus, powL );
end

% Adjust the delays to match the given delay spread
if isempty( ds )
    taus = taus * 1e-9;     % Delays are in [ns] 
else
    taus = ds * 1e-9 * taus ./ qf.calc_delay_spread( taus, powL );
end

% Adjust the angles to match the given angular spread
[ as_model, mean_angle ] = qf.calc_angular_spreads( ang, powL );  % Values in [rad]
for n = 1:4
    if ~isnan(as_desired(n))
        tmp = angle( exp( 1j*( ang(n,:)-mean_angle(n) )) );     % Rotate angles
        tmp = as_desired(n)/as_model(n) * tmp;                  % Scale angles
        if powL(1) > 1e-14                                      % Align LOS direction
            rot_angle = ang(n,1)-tmp(1);
            tmp = angle( exp( 1j*( tmp+rot_angle )) );          % Rotate angles
        else
            tmp = angle( exp( 1j*( tmp+mean_angle(n) )) );      % Rotate angles
            tmp(1) = ang(n,1);                                  % Set original LOS angle
        end
        ang(n,:) = tmp;
    else
        as_desired(n) = as_model(n);                            % Used later
    end
end

% Sanity check
tmp = qf.calc_angular_spreads( ang, powL );
if any( abs( as_desired - tmp ) > 1e-12 )
    error('QuaDRiGa:qd_builder:gen_cdl_model','Failed to set angle spread.')
end

% Configure channel builder
h_builder = qd_builder('LOSonly');
h_builder.simpar.center_frequency = center_frequency;
h_builder.simpar.sample_density = sample_density;
h_builder.simpar.use_3GPP_baseline = true;
h_builder.simpar.show_progress_bars = false;
h_builder.simpar.autocorrelation_function = 'Disable';

% Default omni-antennas, vertical polarization
h_builder.tx_array = qd_arrayant('omni');
h_builder.rx_array = qd_arrayant('omni');

% Set virtual TX and RX positions
h_builder.tx_position = [0;0;15];
if use_tdl_model
    h_builder.rx_track = qd_track('linear',mobile_speed*duration,45*pi/180);
else
    h_builder.rx_track = qd_track('linear',mobile_speed*duration,0);
end
if h_builder.rx_track.no_snapshots > 1.5
    h_builder.rx_track.interpolate_positions( h_builder.simpar(1,1).samples_per_meter );
end
h_builder.rx_track.initial_position(3,1) = 1.5;     % RX-Height = 1.5 Meters
if abs( ang(3,1) ) < 1e-3 && abs( ang(4,1) ) < 1e-3
    d2d = 100;
    h_builder.tx_position(3,1) = 1.5;   % Same tx-rx-height
else
    d2d = (h_builder.tx_position(3,1)-1.5) ./ tan(ang(4,1));
end
h_builder.rx_track.initial_position(1:2,1) = [cos(ang(1,1));sin(ang(1,1))]*d2d;

if duration > 0
    h_builder.rx_track.movement_profile = [0,duration;0, get_length( h_builder.rx_track )];
end
h_builder.check_dual_mobility;


% Set the LSPs to match the input variables
scenpar                 = h_builder.scenpar;
scenpar.NumClusters     = numel(powL);
scenpar.DS_mu           = log10( qf.calc_delay_spread( taus, powL ) );
scenpar.DS_sigma        = 0;
scenpar.AS_D_mu         = log10( as_desired(1)*180/pi );
scenpar.AS_D_sigma      = 0;
scenpar.AS_A_mu         = log10( as_desired(2)*180/pi );
scenpar.AS_A_sigma      = 0;
scenpar.ES_D_mu         = log10( as_desired(3)*180/pi );
scenpar.ES_D_sigma      = 0;
scenpar.ES_A_mu         = log10( as_desired(4)*180/pi );
scenpar.ES_A_sigma      = 0;
scenpar.XPR_mu          = xpr;
scenpar.XPR_sigma       = 0;
scenpar.KF_mu           = 10*log10( powL(1)/sum(powL(2:end)) );
scenpar.SF_sigma        = -100;
if use_tdl_model
    % Emulate Jakes Spectrum
    scenpar.PerClusterAS_D  = 180;
    scenpar.PerClusterAS_A  = 360;
    scenpar.PerClusterES_D  = 5;
    scenpar.PerClusterES_A  = 5;
    scenpar.NumSubPaths     = 200;
else
    scenpar.PerClusterAS_D  = cAS(1);
    scenpar.PerClusterAS_A  = cAS(2);
    scenpar.PerClusterES_D  = cAS(3);
    scenpar.PerClusterES_A  = cAS(4);
    scenpar.SubpathMethod   = SubpathMethod;
end
plpar.model             = 'constant';
plpar.A                 = -10*log10(sum(powL));
h_builder.scenpar       = scenpar;
h_builder.plpar         = plpar;
h_builder.gen_parameters;
h_builder.sos           = [];

% Write cluster parameters to the file
h_builder.taus      = taus;
h_builder.pow       = powL/sum(powL);
h_builder.AoD       = ang(1,:);
h_builder.AoA       = ang(2,:);
h_builder.EoD       = ang(3,:);
h_builder.EoA       = ang(4,:);

% Set FBS and LBS positions (for visualization only)
if ~use_tdl_model
    gen_fbs_lbs( h_builder );
end

% Set name-tags
h_builder.name = ['ch_',cdl_model];
h_builder.rx_track.name = 'RX';
h_builder.tx_track.name = 'TX';

% Fix for Octave
h_builder = h_builder(1,1);

end

