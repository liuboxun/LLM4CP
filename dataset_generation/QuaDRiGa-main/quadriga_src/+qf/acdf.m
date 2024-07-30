function [ Sh , bins , Sc , mu , sig ] = acdf( data , bins , dim , cdim  )
%ACDF Calculate the cumulative distribution function of a given data set
%
% Description:
%   This function calculates the empirical cumulative distribution function from the given data. It
%   is possible to analyze multi-dimensional data-sets and average the results. For example, you
%   may collect 10,000 data samples of an experiment and repeat the experiment 5 times. Hence, your
%   data variable may have the size [ 5,10000 ]. Then, you want to calculate 5 independent CDFs,
%   one for each experiment run, and evaluate how much the results differ for each run. Calling 
%
%   [ Sh , bins, Sc , mu , sig ] = qf.acdf( data, [], 2,1  ); 
%
%   will produce the desired results. The input parameter dim = 2 describes which dimension of the
%   data set contains the samples, cdim = 1 describes on which dimension the repetitions are stored.
%   The output variable Sh contains the individual probabilities (y-axis), bins contains the sample
%   values (x-axis), Sc the average probabilities where the averaging was done over the quantiles.
%
% Input:
%   data
%   The data samples can be either given as a multi-dimensional array of size [ nA, nB, nC ...] or
%   as a cell array. The latter can be useful if there are different numbers of results per
%   experimental run.
%
%   bins
%   The center values of each bin on the x-axis of the (non-cumulative) distribution function. By
%   default, 201 bins are equally spaced over the data range.
%
%   dim
%   The dimension on which the analysis is done, Default: 1
%
%   cdim
%   The dimension for which the resulting CDFs are averaged, Default: 2
%
% Output:
%   Sh
%   The individual CDFs. Default size: [ no_bins, nB, nC, ...]
%
%   bins
%   The center values of each bin on the x-axis of the (non-cumulative) distribution function.
%
%   Sc
%   The averaged CDFs; Default size: [ no_bins, nC, ...]
%
%   mu
%   The average 0.1 ... 0.9 quantiles; Default size: [ 9, nC, ...]
%
%   sig
%   The standard deviation of the 0.1 ... 0.9 quantiles; Default size: [ 9, nC, ...]
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

% Read the dimension of the input array
ns = size(data);

if ~exist('dim','var')
    dim = find(ns>1,1);
end

if ~exist('cdim','var')
    cdim = dim + 1;
end

if isempty( data )
    Sh   = NaN;
    bins = NaN;
    Sc   = NaN;
    mu   = NaN(9,1);
    sig  = NaN(9,1);
    return
end

if iscell( data )
    dim = 1;
    cdim = 2;
end

if nargin == 1 || isempty( bins )
    if iscell( data )
        temp = cat( 1,data{:} );
    else
        temp = reshape( data , [] , 1 );
    end
    mi = min( temp( temp>-Inf ) );
    ma = max( temp( temp<Inf ) );
    if mi ~= ma
        bins = mi : ( ma-mi ) / 200 : ma;
    else
        mi = 0.9*mi;
        ma = 1.1*ma;
        bins = mi : ( ma-mi ) / 200 : ma;
    end
end

if cdim == dim
    error('Input "cdim" must be different from "dim"') ;
end

% For easy access
no_bins = numel( bins );

if iscell( data )
    
    no_seed =  numel( data );
    
    % Calculate the histograms
    Sh = zeros( no_bins , no_seed );
    for n = 1:no_seed
        tmp = data{n}(:);
        tmp = tmp(~isinf(tmp));
        tmp = tmp(~isnan(tmp));
        Sh(:,n) = hist( tmp , bins );
        Sh(:,n)  = cumsum( Sh(:,n)  ) ./  numel( tmp );
    end
    
    D = Sh;
    
    vals = 0:0.005:0.995;
    
    Sc  = zeros( numel(vals) , 1 );
    Scc = zeros( numel(vals) , no_seed );
    mu = zeros( 9 , 1 );
    sig = zeros( 9 ,1 );
    
    for i_val = 1:numel(vals)
        temp = zeros( no_seed,1 );
        for i_seed = 1:no_seed
            temp( i_seed ) = ...
                bins( find( D(:,i_seed) > vals(i_val) , 1 ) );
        end
        Sc(i_val)    = mean(temp,1);
        Scc(i_val,:) = temp;
    end
    
    for i_mu = 1:9
        ii = i_mu*20-4 : i_mu*20+6;
        tmp = Scc( ii,: );
        mu(i_mu) = mean( tmp(:) );
        sig(i_mu) = std( tmp(:)) ;
    end
    
    % Map to bins
    Sd = zeros(no_bins,1);
    for i_bins = 1:no_bins
        Sd(i_bins) = sum( Sc < bins(i_bins) );
    end
    Sc = Sd./ numel(vals);
    
else
    no_values = ns(dim);
    
    % Reshape the input data
    order = [ dim , setdiff( 1:numel(ns) , dim ) ];
    D = permute( data , order );
    D = reshape(  D , no_values , [] );
    
    % Calculate the histograms
    
    % hist dies not work with Inf values
    ii = find( isinf(D) );
    sgn = sign( D(ii) );
    D( ii(sgn<0) ) = -3.4e38;
    D( ii(sgn>0) ) = 3.4e38;
    
    Sh = hist( D , bins );
    if numel( size(D) ) == 2 && size(D,2) == 1
        Sh = Sh.';
    end
    Sh = cumsum( Sh ) ./ no_values;
    
    % Reshape the output to match input and remove singletons
    nsn = ns( order( 2:end ) );
    if ~(numel(nsn) == 1 && nsn == 1)
        nsn = nsn( nsn~=1 );
    end
    
    % Calculate mu
    if cdim == 0
        vals = 0.1:0.1:0.9;
        mu = zeros( 9 , size(Sh,2) );
        for i_mu = 1:9
            for i_data  = 1:size(Sh,2)
                mu( i_mu,i_data ) = bins( find( Sh(:,i_data) >= vals(i_mu) , 1) );
            end
        end
        mu  = reshape( mu  , [ 9 , nsn ] );
    end
    
    Sh = reshape( Sh , [ no_bins , nsn ] );
    
    if cdim
        % Reorder to accout for singletons
        temp = find( ns==1 );
        if isempty( temp )
            temp = order;
        else
            temp = order( order ~= temp );
        end
        ncdim = find( temp == cdim );
        
        if isempty( ncdim )
            % Here, cdim is a singleton dimension
            Sc = Sh;
            vals = 0.1:0.1:0.9;
            for i_mu = 1:9
                mu( i_mu,1 ) = bins( find( Sh(:,1) >= vals(i_mu) , 1) );
            end
            sig = zeros( size(mu) );
            
        else
            ns = size(Sh);
            order = [ 1 , ncdim , setdiff( 2:numel(ns) , ncdim ) ];
            
            D = permute( Sh , order );
            D = reshape( D , ns(1) , ns(ncdim) , [] );
            
            no_data = size(D,3);
            no_seed = size(D,2);
            
            vals = 0:0.001:0.999;
            
            Sc = zeros( numel(vals) , size(D,3) );
            mu = zeros( 9 , size(D,3) );
            sig = zeros( 9 , size(D,3) );
            
            i_mu = 1;
            for i_val = 1:numel(vals)
                temp = zeros( no_seed, no_data );
                for i_seed = 1:no_seed
                    for i_data = 1:no_data
                        temp( i_seed,i_data ) = ...
                            bins( find( D(:,i_seed,i_data) > vals(i_val) , 1 ) );
                    end
                end
                Sc(i_val,:) = mean(temp,1);
                
                if any( i_val == 101:100:901 )
                    mu(i_mu,:) = mean(temp,1);
                    sig(i_mu,:) = std( temp) ;
                    i_mu = i_mu + 1;
                end
            end
            
            % Map to bins
            Sd = zeros(no_bins,no_data);
            for i_bins = 1 : no_bins
                Sd(i_bins,:) = sum( Sc < bins(i_bins) );
            end
            Sc = Sd./ numel(vals);
            
            % Reorder to match initial grid
            if numel( order )>2
                Sc = reshape( Sc  , [ no_bins , ns( order(3:end) ) ] );
                mu  = reshape( mu  , [ 9 , ns( order(3:end) ) ] );
                sig = reshape( sig  , [ 9 , ns( order(3:end) ) ] );
            end
        end
    else
        Sc = Sh;
        sig = zeros( size(mu) );
    end
    
end

end



