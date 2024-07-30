function h_qd_satellite = generate( constellation, Ain, Bin, Cin, Din, Ein, Fin )
%GENERATE Generates predefined satellite constellations
%
% Calling object:
%   None (constructor)
%
% Description:
%   The constructor creates new satellite constellations with a number of S satellites. The input
%   parameter 'constellation' defines the constellation type. Additional parameters are specific to
%   the constellation. They are defined by the following list. If no input is specified, an empty
%   object is created.
%
% Constellation:
%   custom
%   A custom constellation. All input parameters must be given as vectors having the dimensions [ 1
%   x S ].
%    * Ain - Semimajor axis in [km]; Default: 42164 km, GEO orbit
%    * Bin - Orbital eccentricity [0-1]; Default: 0
%    * Cin - Orbital inclination in [degree]; Default: 0
%    * Din - Longitude of the ascending node in [degree]; Default: 0
%    * Ein - Argument of periapsis in [degree]; Default: 0
%    * Fin - True anomaly in [degree]; Default: 0
%
%
%   gso
%   A constellation of equally spaced geostationary satellites.
%    * Ain - Number of satellites S; Scalar variable; Default: 3
%    * Bin - Phase offset of the first satellite in [degree]; Scalar variable
%
%
%   walker-delta
%   Walker-Delta pattern constellation is used for a global coverage of the Earth's surface by a
%   minimum number of satellites in circular orbits. A Walker-Delta pattern contains of total of
%   'S' satellites in 'p' orbital planes with 't=S/p' satellites in each orbital plane. All orbital
%   planes are assumed to be in same inclination with reference to the equator. The phase
%   difference between satellites in adjacent plane is defined as the angle in the direction of
%   motion from the ascending node to the nearest satellite at a time when a satellite in the next
%   most westerly plane is at its ascending node. In order for all of the orbit planes to have the
%   same phase difference with each other, the phase difference between adjacent satellites must be
%   a multiple of 'f*360/S', where 'f' can be an integer between 0 to p-1.
%    * Ain - Semimajor axis for all satellites in [km]; Scalar variable
%    * Bin - Orbital inclination in [degree]; Scalar variable
%    * Cin - Number of orbital planes 'p'; Scalar variable
%    * Din - Number of satellites per plane 't'; Scalar variable
%    * Ein - Phase difference in [degree]; Scalar variable
%
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

var_names = {'Ain','Bin','Cin','Din','Ein','Fin'};
for n = 1:numel( var_names )
    if ~exist( var_names{n},'var' )
        eval([ var_names{n},' = [];' ]);
    end
end

switch lower( constellation )
    case 'custom'
        h_qd_satellite = gen_constellation_custom( Ain, Bin, Cin, Din, Ein, Fin );
    case 'gso'
        h_qd_satellite = gen_constellation_gso( Ain, Bin );
    case 'walker-delta'
        h_qd_satellite = gen_constellation_walker_delta( Ain, Bin, Cin, Din, Ein );
    otherwise
        error('QuaDRiGa:qd_satellite:generate',['??? Constellation type "',constellation,'" is not supported.']);
end

end
