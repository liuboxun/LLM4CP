function h_figures = visualize( h_qd_arrayant, i_element )
%VISUALIZE Create a plot showing the element configurations
%
% Calling object:
%   Single object
%
% Input:
%   i_element
%   The element indices for which the plot os created. If no element index are given, a plot is
%   created for each element in the array. 
%
% Output:
%   h_figures
%   The figure handles for further processing of the images.
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
if numel( h_qd_arrayant ) > 1 
   error('QuaDRiGa:qd_arrayant:visualize','visualize not definded for object arrays.');
else
    h_qd_arrayant = h_qd_arrayant(1,1); % workaround for octave
end

if nargin == 1
    i_element = 1:h_qd_arrayant.no_elements;
end

if ~(any(size(i_element) == 1) && isnumeric(i_element) ...
        && isreal(i_element) && all(mod(i_element, 1) == 0) && all(i_element > 0))
    error('??? "element" must be integer and > 0')
elseif any(i_element > h_qd_arrayant.no_elements)
    error('??? "element" exceed "no_elements"')
end

az = double(h_qd_arrayant.azimuth_grid);
if numel(az) == 361
    indt = 1:5:361;
else
    indt = 1:numel(az);
end
az = az(indt);

elev = double(h_qd_arrayant.elevation_grid');
if numel(elev) == 181
    indp = 1:5:181;
else
    indp = 1:numel(elev);
end
elev = elev(indp);

if min(diff(az)) < 3 || min(diff(el)) < 3
    use_grid = 0;
else
    use_grid = 1;
end

[az_grid, elev_grid] = meshgrid(az, elev);
[Xi, Yi, Zi] = sph2cart(az_grid, elev_grid, 1);

min_value = -20;
scaling = 1;

h_figures = zeros(1, numel(i_element));

for n = 1:numel(i_element)
    
    % Calculate the gain using the calc-gain method
    [directivity_dbi,P_max] = calc_gain( h_qd_arrayant, i_element(n) );
    directivity_lin = 10.^(0.1*directivity_dbi);
    P_max = 10.^(0.1*P_max);
        
    % Read the element patterns
    Fa = h_qd_arrayant.Fa(indp, indt, i_element(n));
    Fb = h_qd_arrayant.Fb(indp, indt, i_element(n));
    
    % Normalize patterns by directivity
    tmp = sqrt(directivity_lin./P_max);
    Fa = Fa .* tmp;
    Fb = Fb .* tmp;
    
    h_figures(n) = figure('Position', [50, 400, 1200, 500],...
        'Name', [h_qd_arrayant.name ' element ', num2str(i_element(n))]);
    
    axes('position',[0 0 0.92 0.9]);%,'Visible','Off');
    axis off
    
    title(['Array Antenna Element ', num2str(i_element(n))] );
    
    text(0.02,1,'D^{[\theta]}(\theta, \phi)');
    text(0.9 ,1,'D^{[\phi]}(\theta, \phi)');
    
    for m = 1:2
        
        axes('position',[-0.4+0.45*m 0.12 0.38 0.82]);
        switch m
            case 1
                Po = 10*log10( abs(Fa).^2 );
            case 2
                Po = 10*log10( abs(Fb).^2 );
        end
        Po = double( Po );
        
        P = Po;
        P(P < min_value) = min_value;
        P = (P - min_value) ./ (directivity_dbi - min_value) .* scaling;
        
        X = P .* Xi;
        Y = P .* Yi;
        Z = P .* Zi;
        
        if use_grid
            surf(X, Y, Z, Po,'Linewidth',0.1)
        else
            surf(X, Y, Z, Po,'Edgecolor','none')
        end
        
        axis('square');
        axis(scaling.*[-1 1 -1 1 -1 1]);
        caxis([min_value, directivity_dbi]);
        set(gca, 'xtick', (-1:1).*scaling/2);
        set(gca, 'ytick', (-1:1).*scaling/2);
        set(gca, 'ztick', (-1:1).*scaling/2);
        xlabel('x')
        ylabel('y')
        zlabel('z')
       
        view(45, 33)
    end
    
    axes('position', [0.08 0.08 0.08 0.08],'Visible','Off');
    caxis([min_value, directivity_dbi]);
    %han = colorbar( 'EastOutside','XTick', min_value:3:floor(directivity_dbi));
    han = colorbar( 'XTick', min_value:3:floor(directivity_dbi));
    set(han, 'position', [0.92 0.06 0.02 0.90])
    zlab = get(han, 'ylabel');
    set(zlab, 'String', 'Partial Directivity in dBi');
    
end


end
