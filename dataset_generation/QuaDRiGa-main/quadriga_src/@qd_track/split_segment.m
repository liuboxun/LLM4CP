function split_segment( h_track , mi , ma , mu , sig , no_check )
%SPLIT_SEGMENT Splits long segments in subsegments of the same type.
%
% Calling object:
%   Object array
%
% Input:
%   mi
%   Minimum length of the subsegment in [m], default: 10m
%
%   ma
%   Maximum length of the subsegment in [m], must be > 2*mi, default: 30m
%
%   mu
%   Mean length of the subsegment (mi < mu < ma), default: 15m
%
%   sig
%   Std of the length of the subsegment, default: 5m
%
%   no_check
%   Disable parsing of input variables, default: false
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

% Parse input variables
if ~exist( 'no_check' , 'var' ) 
    no_check = false;
else
    no_check = logical( no_check );
end
    
if ~no_check
    if exist( 'mi' , 'var' ) && ~isempty( mi )
        if ~( isnumeric(mi) && all(size(mi) == [1 1]) && isreal(mi) && mi>0 )
            error('??? mi must be scalar, and > 0')
        end
    else
        mi = 10;
    end
    
    if ~exist( 'ma' , 'var' ) || isempty( ma )
        ma = 30;
    end
    if ~( isnumeric(ma) && all(size(ma) == [1 1]) && isreal(ma) && ma>2*mi )
        error('??? ma must be scalar, and > 2*mi')
    end
    
    if ~exist( 'mu' , 'var' ) || isempty( ma )
        mu = 15;
    end
    if ~( isnumeric(mu) && all(size(mu) == [1 1]) && isreal(mu) && mu>mi && mu<ma)
        error('??? mu must be scalar, and mu > mi , mu < ma')
    end
    
    if exist( 'sig' , 'var' ) && ~isempty( mi )
        if ~( isnumeric(sig) && all(size(sig) == [1 1]) && isreal(sig) )
            error('??? sig must be scalar')
        end
    else
        sig = 5;
    end
end

if numel(h_track) > 1
    
    sic = size( h_track );
    prc = false( sic );
    for n = 1 : prod( sic )
        if ~prc( n )
            [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
            split_segment( h_track(i1,i2,i3,i4) , mi , ma , mu , sig , 1 );
            prc( qf.eqo( h_track(i1,i2,i3,i4), h_track ) ) = true;
        end
    end
    
else
    h_track = h_track(1,1);
    
    seg_ind = [h_track.segment_index , h_track.no_snapshots];
    [~,dist] = get_length( h_track );
    
    seg = [];
    scen = {};
    i_seg_new = 1;
    
    % Read the parameters from the track and determine, which parameters to
    % process.
    opar = h_track.par;
    if isempty( opar )
        n_par = 0;
        process_par = [];
        par = [];
    else
        names = fieldnames( opar );
        n_par = numel( names );
        process_par = false( 1,n_par );
        for n = 1:n_par
            if ~isempty( opar.( names{n} ) ) && ...
                    ~strcmp( names{n} , 'kf' ) && ...
                    ~strcmp( names{n} , 'pg' )
                process_par(n) = true;
            end
        end
        par = opar;
    end    
    
    % Process each segment
    for i_seg = 1 : h_track.no_segments
        
        seg_start  = seg_ind( i_seg );
        seg_end    = seg_ind( i_seg+1 );
        
        seg_length = dist( seg_end ) - dist( seg_start );
        dist_seg = dist(seg_start:seg_end) - dist(seg_start);
        
        seg(i_seg_new) = seg_start;
        scen(:,i_seg_new) = h_track.scenario(:,i_seg);
        for i_par = 1:n_par
            if process_par( i_par )
                par.( names{i_par} )(:,i_seg_new,:) = opar.( names{i_par} )(:,i_seg,:);
            end
        end
        i_seg_new = i_seg_new+1;
        
        if seg_length > ma
            
            % Get a list of sub-segments
            dd = [];
            split_me = true;
            while split_me
                % Get a random length
                d = randn(1)*sig + mu;
                while d<mi || d>ma
                    d = randn(1)*sig + mu;
                end
                
                % We get the remaining length
                rem = seg_length - sum(dd) - d;
                if rem >= ma
                    dd = [dd,d];
                else
                    dd = [ dd , d , rem ];
                    split_me = false;
                end
            end
            dd = dd( randperm( numel(dd) ) );
            dd = cumsum(dd);
            
            for o = 2:numel(dd)
                ind = find( dist_seg > dd(o-1) , 1);
                
                if ( dist_seg(ind) - dd(o-1) ) > 1 && n_par == 0
                    
                    % Creae new snapshot
                    PT = h_track.positions;
                    ii = seg_start+ind-1;                      	% Start Index
                    P = PT(:,ii);                               % Start point
                    U = PT(:,ii-1)-P;                           % Direction
                    r = (dist_seg(ind)-dd(o-1)) ./ sqrt(sum(U.^2));             % Distance
                    PG =  P + r*U;                              % Position of new point
                    
                    or = h_track.orientation;                   % Copy the orientations
                    si = h_track.segment_index;                 % Copy the existing segment index
                    
                    h_track.positions = [ PT(:,1:ii-1), PG, PT(:,ii:end) ];
                    h_track.orientation = [ or(:,1:ii), or(:,ii:end) ];
                    h_track.segment_index = [ si( si<ii ), si( si>=ii )+1 ];
                    
                    % Update segment parameters
                    seg_ind = [h_track.segment_index , h_track.no_snapshots];
                    [~,dist] = get_length( h_track );
                    seg_start  = seg_ind( i_seg );
                    seg_end    = seg_ind( i_seg+1 );
                    dist_seg = dist(seg_start:seg_end) - dist(seg_start);
                    
                    seg(i_seg_new) = seg_start + ind - 1;
                    scen(:,i_seg_new) = h_track.scenario(:,i_seg);
                    i_seg_new = i_seg_new+1;
                    
                else
                    seg(i_seg_new) = seg_start + ind - 1;
                    scen(:,i_seg_new) = h_track.scenario(:,i_seg);
                    
                    for i_par = 1:n_par
                        if process_par( i_par )
                            par.( names{i_par} )(:,i_seg_new,:) = opar.( names{i_par} )(:,i_seg,:);
                        end
                    end
                    i_seg_new = i_seg_new+1;
                end
            end
        end
    end
    
    [seg,scenind] = unique(seg);
    scen = scen(:,scenind);
    
    h_track.segment_index = seg;
    h_track.scenario = scen;
    h_track.par = par;
end
end

