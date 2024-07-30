function element_position = calc_element_position( h_arrayant, verbose )
%CALC_ELEMENT_POSITION Calculates the element positions from the antenna patterns
%
% Calling object:
%   Single object
%
% Description:
%   When an antenna pattern is measured in an anechoic chamber, the phase center, i.e. the point
%   from which the electromagnetic radiation spreads spherically outward with the phase of the
%   signal being equal at any point on the sphere, is relative to the rotation center of the
%   positioner. This method calculates the element positions relative to the phase center of the
%   array and adjusts the phases of the individual elements such that their phase center is
%   centered at the element position. The maximum radius of the array antenna cannot exceed 20
%   wavelengths and the resolution of the position estimation is set to 0.02 wavelengths.
%
% Input:
%   verbose
%   A value of 1 (default) shows a progress bar for the calculations. A value of 2 shows a plot
%   visualizing the estimation process.
%
% Output:
%   element_position
%   Position of the antenna elements in local cartesian coordinates (using units of [m])
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

if numel( h_arrayant ) > 1 
   error('QuaDRiGa:qd_arrayant:calc_element_position','calc_element_position not definded for object arrays.');
else
    h_arrayant = h_arrayant(1,1); % workaround for octave
end

if ~exist('verbose','var') || isempty(verbose)
    verbose = true;
end

nEl = h_arrayant.no_elements;
if verbose
    fprintf('Element Pos. [');
    vb_dots = 50;
    tStart = clock;
    m0=0;
end
i_bar = 0;

if any( abs( h_arrayant.element_position(:) ) > 1e-7 )
    h_arrayant.combine_pattern;
end

% Determine the wavelength
s = qd_simulation_parameters;
s.center_frequency = h_arrayant.center_frequency;
lambda = s.wavelength;

% Search positions
pos = -20*lambda : lambda/2 : 20.1*lambda;
res = 0.0375;

for iEl = 1 : nEl
    
    x = 0;
    y = 0;
    z = 0;
    
    for n = 1 : 6
        % Generate probe antenna opject
        a = h_arrayant.sub_array(iEl);  % Copy pattern
        a.center_frequency = h_arrayant.center_frequency;
        
        switch n
            case 1 % Search x-pos
                a.set_grid( (-180:5:180)*pi/180, (-80:5:80)*pi/180 );
                a.no_elements = numel(pos);
                a.element_position(1,:) = pos;
                str = 'Coarse x position';
                pp = pos;
                
            case 2 % Search y-pos
                a.set_grid( (-180:5:180)*pi/180, (-80:5:80)*pi/180 );
                x = pos(ndx);
                a.element_position = [x;y;z];
                a.combine_pattern;
                a.no_elements = numel(pos);
                a.element_position(2,:) = pos;
                str = 'Coarse y position';
                pp = pos;
                
            case 3 % Search z-pos
                a.set_grid( (-180:3:180)*pi/180, (-81:3:81)*pi/180 );
                y = pos(ndx);
                a.element_position = [x;y;z];
                a.combine_pattern;
                a.no_elements = numel(pos);
                a.element_position(3,:) = pos;
                str = 'Coarse z position';
                pp = pos;
                
            case 4 % Refine x-pos
                a.set_grid( (-180:10:180)*pi/180, (-80:10:80)*pi/180 );
                z = pos(ndx);
                a.element_position = [x;y;z];
                a.combine_pattern;
                a.no_elements = numel(pos);
                a.element_position(1,:) = pos*res;
                str = 'Refined x position';
                pp = x+pos*res;
                
            case 5 % Refine y-pos
                a.set_grid( (-180:10:180)*pi/180, (-80:10:80)*pi/180 );
                x = x + pos(ndx)*res;
                a.element_position = [x;y;z];
                a.combine_pattern;
                a.no_elements = numel(pos);
                a.element_position(2,:) = pos*res;
                str = 'Refined y position';
                pp = y+pos*res;
                
            case 6
                a.set_grid( (-180:10:180)*pi/180, (-80:10:80)*pi/180 );
                y = y + pos(ndx)*res;
                a.element_position = [x;y;z];
                a.combine_pattern;
                a.no_elements = numel(pos);
                a.element_position(3,:) = pos*res;
                str = 'Refined z position';
                pp = z+pos*res;
        end
        
        a.combine_pattern;
        
        W = angle(a.Fa);
        X = unwrap( W,[],2 );
        X = abs(diff( X,1,2 ));
        Y = abs(a.Fa(:,1:end-1,:)).^2 + abs(a.Fa(:,2:end,:)).^2;
        Z = sum(sum(X.*Y,1),2);
        cst = Z(:);
        
        X = unwrap( W,[],1 );
        X = abs(diff( X,1,1 ));
        Y = abs(a.Fa(1:end-1,:,:)).^2 + abs(a.Fa(2:end,:,:)).^2;
        Z = sum(sum(X.*Y,1),2);
        cst = cst + Z(:);
        
        W = angle(a.Fb);
        X = unwrap( W,[],2 );
        X = abs(diff( X,1,2 ));
        Y = abs(a.Fb(:,1:end-1,:)).^2 + abs(a.Fb(:,2:end,:)).^2;
        Z = sum(sum(X.*Y,1),2);
        cst = cst + Z(:);
        
        X = unwrap( W,[],1 );
        X = abs(diff( X,1,1 ));
        Y = abs(a.Fb(1:end-1,:,:)).^2 + abs(a.Fb(2:end,:,:)).^2;
        Z = sum(sum(X.*Y,1),2);
        cst = cst + Z(:);
        
        [~,ndx] = min(cst);
        
        % Update progress bar
        i_bar = i_bar + 1;
        if verbose; m1=ceil(i_bar/(nEl*6)*vb_dots); if m1>m0
                for m2=1:m1-m0; fprintf('o'); end; m0=m1; end; end
        
        if verbose == 2
              figure(1)
              plot(pp,cst(:),'-ob');
              hold on
              plot(pp(ndx),cst(ndx),'ob','Markerfacecolor','r');
              hold off
              text(pp(ndx),cst(ndx),['  ',num2str(pp(ndx))])
              title(['Element ',num2str(iEl),' - ',str]);
              ylabel('Cost Function')
              xlabel([str,' [m]'])
              drawnow
        end
        
    end
    z = z + pos(ndx)*res;
    
    h_arrayant.element_position(:,iEl) = [x;y;z];
end

element_position = -h_arrayant.element_position;
h_arrayant.combine_pattern;
h_arrayant.element_position = element_position;

if verbose
    fprintf('] %5.0f seconds\n',round( etime(clock, tStart) ));
end

end

