function [ mu, sigma, R, str ] = regression( V, X, margin, round_digits, Xname, fit_sigma )
%REGRESSION Linear regression fitting with N variables
%
% Calling object:
%   None (static method)
%
% Description:
%   This function fits the data "V" sampled at points "X":
%
%   V = R + Q * S 
%   R = mu(1) + mu(2) * X(1,:) + mu(3) * x(2,:) ... 
%   S = sigma(1) + sigma(2) * X(1,:) + sigma(3) * x(2,:) ... 
%
%   V is assumed to be a random, normal-distributed variable with a mean value R and standard
%   deviation S. The reference value R depends on the support X. Q is a normal-distributed random
%   variable with zero-mean and unit-variance. The scaling of the STD of V is done by S, which also
%   depends on the support X.
%
% Input:
%   V
%   Vector of M data values. Dimension: [1 x M]
%
%   X
%   Support vectors for the data values (the values where the data was sampled). Dimension: [N x M]
%
%   margin
%   Absolute values of fitted parameters below this margin are assumed to be zero. The default
%   value is obtained from round_digits, e.g. for round_digits of 2, margin is 0.01.
%
%   round_digits
%   Rounds the output to this number of decimal digits after the coma.
%
%   Xname
%   Cell array with N elements containing the names of the variables (optional).
%
%   fit_sigma
%   Logical vector with N elements indicating for which variable the STD should be fitted
%   (optional, default: all variables).
%
% Output:
%   mu
%   Fitted parameters (mean values). Dimension: [1 x N+1]
%
%   sigma
%   Fitted standard deviation. Dimension: [1 x N+1]
%
%   R
%   The reference value R for the input sample points. Dimension: [1 x M]
%
%   str
%   A human readable text string of the fitted function.
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


if size(V,1) ~= 1
   V = V(:).';
end
M = numel(V);

if ~exist( 'X','var' ) || isempty( X )
    X = [];
end
if numel(X) >= M && size(X,1) == M
    X = X.';
end
if size( X,2 ) ~= size( V,2 ) && ~isempty(X)
    error('Size of X does not match size of V');
else
    noX = size(X,1);
end

if ~exist( 'round_digits','var' ) || isempty( round_digits )
    round_digits = [];
    num_format = '%g';
    accuracy_target = 1e-13;
else
    num_format = ['%1.',num2str(ceil(round_digits)),'f'];
    accuracy_target = 10^(-ceil(round_digits)-2);
    if abs( rem(round_digits,1)) - 0.5 < 1e-6   % Round to 0.5
        round_digits = log10(10^floor(round_digits)*2);
    elseif abs( rem(round_digits,1)) - 0.2 < 1e-6   % Round to 0.2
        round_digits = log10(10^floor(round_digits)*5);
    end
end

if ~exist( 'margin','var' ) || isempty( margin )
    if isempty( round_digits )
        margin = 0;
    else
        margin = 10^-round_digits;
    end
end

if ~exist( 'Xname','var' ) || isempty( Xname )
    disp_str = false;
    Xname = {};
    for n = 1:noX
        Xname{n} = ['x',num2str(n)]; %#ok
    end
else
    disp_str = true;
end

if ~exist( 'fit_sigma','var' ) || isempty( fit_sigma )
    fit_sigma = true( 1,noX );
elseif numel( fit_sigma ) == 1
    fit_sigma = logical( fit_sigma );
    fit_sigma = fit_sigma( 1,ones(1,noX ));
else
    fit_sigma = logical( fit_sigma );
end

V = double(V);                                          % Always use double precision
X = double(X);
O = ones( numel(V),1 );

ii = all( ~isinf([V;X]) & ~isnan([V;X]) ,1);            % Sort out errors
V = V(ii);
if noX; X = X(:,ii); end

% Find parameters
if noX == 0
    mu = mean(V);
    
elseif noX == 1
    [Q,R] = qr( [ X',O ] ,0);                        	% Solve linear least squares problem
    mu = (R\(Q'*V.')).';                                % Calculate coefficients
    if abs(mu(1)) < margin                              % Update mu
        mu(1) = mean(V);
        mu(2) = 0;
    else
        mu = mu([2,1]);
    end
    
else % noX > 1

    mu = zeros( 2, noX+1 );                         	% Start values
    accuracy = Inf;                                 	% Initial accuracy
    lp = 0;                                         	% Loop counter
    while lp < 1000 && accuracy > accuracy_target
        for iX = 1 : noX
            iN = true(1,noX); iN(iX) = false;           % Select which values to keep fixed
            Vn = V - mu(1,[false,iN]) * X(iN,:);        % Update values
            [Q,R] = qr( [ X(~iN,:)',O ] ,0);            % Solve linear least squares problem
            p = R\(Q'*Vn.');                            % Calculate coefficients
            mu(2,[true,~iN]) = p([2,1])';               % Update mu
        end
        accuracy = max(abs(mu(1,:)-mu(2,:)));       	% Calculate accuracy
        mu(1,:) = mu(2,:);                              % Update mu
        lp = lp + 1;                                   	% Increase loop counter
    end
    mu = mu(1,:);                                      	% Use last update
    
    ii = abs(mu) < margin; ii(1) = false;               % Set values below margin to 0
    if any( ii(2:end) )                                 % Recursive update of other values
        mu(~ii) = qf.regression( V, X( ~ii(2:end),: ), margin, round_digits );
        mu(ii) = 0;
    end
end

if ~isempty( round_digits )                             % Round output
    mu = round( mu * 10^round_digits )/10^round_digits;
end

if nargout > 1                                        	% Fit STD
    R = mu(1,1) + mu(1,2:end) * X;                     	% Reminder after fitting
    sigma = zeros(1,noX+1);                             % Placeholder for STD
    
    % Fit the variance
    sigma([true,fit_sigma]) = qf.regression( abs(V-R).^2, X(fit_sigma,:) );
    
    % Transform variance to STD
    Y = rand(noX,10000);
    for iX = 1 : noX
        tmp = min(X(iX,:));
        Y(iX,:) = Y(iX,:) .* (max(X(iX,:))-tmp) + tmp;
    end
    U = sigma(1,1) + sigma(1,2:end) * Y;
    U(U<0) = 0;
    U = sqrt(U);
    
    % Fit the STD
    sigma([true,fit_sigma]) = qf.regression( U, Y(fit_sigma,:), margin, round_digits );
end

if nargout > 3 || nargin > 4                            % Format output string
    str ={};
    str{1} = ['avg = ',num2str(mu(1),num_format),' '];
    str{2} = ['std = ',num2str(sigma(1),num_format),' '];
    for iX = 1 : noX
        if mu(iX+1) > 0.001
            str{1} = [str{1},'+ ',num2str(mu(iX+1),num_format)];
        elseif mu(iX+1) < -0.001
            str{1} = [str{1},'- ',num2str(-mu(iX+1),num_format)];
        end
        if abs( mu(iX+1) ) > 0.001
            str{1} = [str{1},' * ',Xname{iX},' '];
        end
        if sigma(iX+1) > 0.001
            str{2} = [str{2},'+ ',num2str(sigma(iX+1),num_format)];
        elseif sigma(iX+1) < -0.001
            str{2} = [str{2},'- ',num2str(-sigma(iX+1),num_format)];
        end
        if abs( sigma(iX+1) ) > 0.001
            str{2} = [str{2},' * ',Xname{iX},' '];
        end
    end
    str{1} = str{1}(1:end-1);
    str{2} = str{2}(1:end-1);
    if nargout < 4 && disp_str
        disp(str{1});disp(str{2});
    end
end

end
