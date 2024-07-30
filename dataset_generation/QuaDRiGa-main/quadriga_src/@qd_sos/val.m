function s = val( h_sos, ca, cb, ph )
%VAL Returns correlated values at given coordinates
%
% Calling object:
%   Object array
%
% Description:
%   This method generates spatially correlated random variables at the given coordinates. The
%   function allows two inputs for the coordinates ("ca" and "cb"). This can be used for dual-
%   mobility scenarios where both ends of the link are moving. In this case, "ca" contains the
%   coordinates for the first mobile device and "cb" contains the coordinates for the second mobile
%   device. For each point in "ca" there must be one point in "cb". For single-mobility scenarios,
%   where only one side is mobile, you may only provide the input for "ca".
%
% Input:
%   ca
%   Coordinates for the first mobile device in [m] given as [3 x N] matrix. The rows correspond to
%   the x,y and z coordinate.
%
%   cb
%   Coordinates for the corresponding second mobile device in [m] given as [3 x N] matrix. The rows
%   correspond to the x,y and z coordinate. This variable must either be empty or have the same
%   size as "ca".
%
%   ph
%   Different phase offsets for the SOS generators (optional)
%
% Output:
%   s
%   Vector ([M x N] elements) of spatially correlated random variables. If the method was called on
%   an object arrays, the rows contain the outputs of the individual SOS objects.  
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

% Number of output values
no_val = size(ca,2);

if ~exist( 'cb','var' ) || isempty( cb )
    cb = [];
elseif any( size(cb) ~= size(ca) )
    error('QuaDRiGa:qd_sos:val','Size of "cb" mut match the size of "ca".' );
end

if ~exist( 'ph','var' ) || isempty( ph )
    ph = [];
else
    ph = double( ph );
end


if numel( h_sos ) > 1
    
    % Recursive call for all objects in the array
    sic = size( h_sos );
    s = zeros( numel( h_sos ), no_val );
    for i_sos = 1 : numel( h_sos )
        [ i1,i2,i3,i4 ] = qf.qind2sub( sic, i_sos );
        s(i_sos,:) = val( h_sos( i1,i2,i3,i4 ), ca, cb, ph );
    end

else
    
    % Fix for octave 4.0 (conversion from object-array to single object)
    h_sos = h_sos(1,1);

    % Read phases from SOS object
    if isempty( ph )
        ph = double( h_sos.sos_phase );
    end

    % Read local variables to increase speed
    al = double( h_sos.sos_amp );
    fr = 2*pi*double( h_sos.sos_freq );
    L  = size(fr,1);
    dims = size(fr,2);
    
    if size( ca, 1 ) > dims
        error('QuaDRiGa:qd_sos:val','Number of requested dimensions is not supported.' );
    end
    
    if size( ca, 1 ) < dims
        ca = [ ca; zeros( dims - size(ca,1), no_val ) ];
    end
    
    if ~isempty(cb) && size( cb, 1 ) < dims
        cb = [ cb; zeros( dims - size(cb,1), no_val ) ];
    end
    
    % Calculate the random variables for the first input coordinate
    if no_val < 1e4
        sa = al .* sum( cos( fr * ca + repmat( ph(:,1),1,no_val ) ),1);
    else % Better memory-efficiency, but slower
        sa = zeros( 1,no_val );
        for l = 1:L
            sa = sa + cos( fr(l,:) * ca  + ph(l,1) );
        end
        sa = sa .* al;
    end
    
    if ~isempty( cb )
        % Adjust SOS frequencies if there is a different decorrelation distance for the second terminal
        if numel( h_sos.Pdist_decorr ) == 2
            fr = fr .* ( h_sos.Pdist_decorr(1) / h_sos.Pdist_decorr(2) );
        end
        
        % Calculate the random variables for the second input coordinate
        if no_val < 1e4
            sb = al .* sum( cos( fr * cb + repmat( ph(:,2),1,no_val ) ),1);
        else % Better memory-efficiency, but slower
            sb = zeros( 1,no_val );
            for l = 1:L
                sb = sb + cos( fr(l,:) * cb  + ph(l,2) );
            end
            sb = sb .* al;
        end
        
        % If the same phases are used for both terminals, they are placed on the same radio map.
        % In this case, closely spaces terminals have correlated values which changes the variance of
        % the distributions. This is fixed here.
        
        % Check if the phases are the same
        if all( abs( ph(:,1) - ph(:,2) ) < 1e-6 )
            % Calculate the distances between the first and second coordinate
            D = sqrt( sum( (cb-ca).^2,1 ) );
            
            % Interpolate the ACF
            D( D>h_sos.dist(end) ) = h_sos.dist(end);
            A = qf.interp( h_sos.dist, 0, h_sos.acf, D );
            
            % Combine the two values
            s = (sa+sb) ./ sqrt(2+2*A);
        else
            % Combine the two values
            s = (sa+sb) ./ sqrt(2);
        end
        
    else
        % There is only one input variable
        s = sa;
    end
    
    % Set the distribution type
    switch h_sos.distribution
        case 'Uniform'
            % Transform Normal distribution to Uniform distribution
            s = 0.5 * erfc( -s/sqrt(2) );
            
        case 'UniformForced'
            % Transform Normal distribution to Uniform distribution
            s = 0.5 * erfc( -s/sqrt(2) );
            
            % Renormalize the output to approximate a Uniform distribution
            if numel(s) > 1
                % Calculate the CDF in the output values
                bins = 0:0.01:1;
                cdf = qf.acdf(s,bins);
                
                % Obtain unique values for the interpolation
                [cdf,ii] = unique(cdf);
                bins = bins(ii);
                bins(end) = 1;
                
                % Interpolate output such that the values match the given distribution
                s = qf.interp( bins, [], cdf.', s );
            end
    end
end

end
