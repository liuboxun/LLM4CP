function [ h_channel, dims ] = mat_load( varargin )
%MAT_LOAD Load data from specially formatted MAT-file. 
%
% Calling object:
%   None (static method)
%
% Description:
%   QuaDRiGa can store the generated channels in a specially formatted MAT file. The static method
%   mat_load allows access to these stored channels. In MATLAB, it is possible to speed up the
%   loading process by only loading part of the file. However, in Octave, this this is not
%   recommended, because Octave always loads the entire file each time the method is called. There
%   are several options for using this method:
%
% Examples:
%   h_channel = qd_channel.mat_load( filename ) 
%   "rowInd" (and also "colInd", "d3Ind" and "d4Ind") are not given. Hence, only the filename is
%   provided. In this case, mat_load reads the complete file with all stored channels. If the file
%   is very big, this will require significant loading time and memory since all stored objects are
%   decompressed. 
% 
%   [∼, dims] = qd_channel.mat_load( filename, 0 )
%   "rowInd" is set to "0" and no further parameters are given. In this case, no channels will be
%   loaded and h_channel will returned as an empty array. However, the output variable dims will
%   contain the dimensions of the stored channel objects. (Not recommended in Octave!)
% 
%   h_channel = qd_channel.mat_load( filename, rowInd ) 
%   "rowInd" is a scalar integer number between 1 to 65535 or a vector of integer numbers and no
%   further parameters are given. In this case, the selected channels are returned. The numbering
%   scheme is the same as in MATLAB, so you can use the MATLAB function "sub2ind" to determine which
%   objects to load. (Not recommended in Octave!) 
%
%   h_channel = qd_channel.mat_load( filename, rowInd, colInd, ... )
%   "rowInd" and "colInd" (and optional "d3Ind" and "d4Ind") are given. Provided that the channel
%   objects are stored as a matrix in the HDF5 file, you can select the rows and columns to load.
%   For example, if the rows represent the receiver index and the columns represent the transmitter
%   index, you can use  "h_channel = channel.mat_load( filename, 2:3, 4 )" to load the channels for
%   Tx4-Rx2 and Tx4-Rx3. A 2x1 array of channel objects will be returned. (Not recommended in
%   Octave!) 
% 
%   h_channel = qd_channel.mat_load( filename, [], colInd, ... )
%   "rowInd" is empty (i.e. "[]") and "colInd" (and optional "d3Ind" and "d4Ind") are given. In this
%   case, all rows for the given "colInd" are returned. For example, if the rows represent the
%   receiver index and the columns represent the transmitter index, you can use "h_channel =
%   channel.mat_load( filename, [], 4 )" to load the channels for all receivers belonging to Tx4.
%   (Not recommended in Octave!) 
% 
%   h_channel = qd_channel.mat_load( filename, [], [], [], [], 'par' )
%   Only the values stored in the "par"-field are loaded. No channel coefficients are loaded from
%   the file. This speeds up the loading procedure, if only results are needed. (Not recommended
%   in Octave!)
%
% Input:
%   filename
%   The path to the MAT file which contains the stored channel data.
%
%   rowInd
%   This is an optional parameter which can be used for loading only parts of the file. See
%   description for details.
%
%   colInd
%   The index of the columns of the channel objects which are loaded from the file. This variable
%   is only used if "rowInd" is given. "colInd" can also be empty. In this case, all columns are
%   returned (same as case 5 above).
%
%   d3Ind
%   The index of the 3rd dimension of the channel objects which are loaded from the file. This
%   variable is only used if "rowInd" and "colInd" are given.
%
%   d4Ind
%   The index of the 4th dimension of the channel objects which are loaded from the file. This
%   variable is only used if "rowInd", "colInd" and "d3Ind" are given.
%
%   usage
%   It set to "par", only evaluation results are loaded and the channel coefficients are skipped.
%
% Output:
%   h_channel
%   An array of 'qd_channel' objects
%
%   dims
%   The dimensions of the stored channel objects
%
%
% QuaDRiGa Copyright (C) 2011-2021
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

% Disable warning
warning('off','MATLAB:load:variableNotFound');

% Parse filename
if nargin == 0
    error('QuaDRiGa:qd_channel:wrongInputValue','??? Filename is missing')
else
    filename = varargin{1};
end

% Get the file size
file = dir( filename );
if numel( file ) ~=1
    error('QuaDRiGa:qd_channel:mat_load',['??? File ''',filename,''' does not exist.'])
end

% Load entire file into memory (required for octave)
if nargin == 1 || isempty(strfind(version,'R20')) %#ok
    load(filename); %#ok
    has_all_data = true;
else
    has_all_data = false;
end

% Parse optional usage parameter
load_only_par = false;
load_selected_par = 0;
if nargin >= 6
    if strcmp( varargin{6} ,'par' )
        load_only_par = true;
        if nargin > 6
            load_selected_par = 7;
        end
    end
    nargin_hack = 5;
else
    nargin_hack = nargin;
end

% Determine the number of channels in the mat file
test_next_channel = true;
max_channel = 0;
while test_next_channel
    var_name = ['channel_',num2str(max_channel+1,'%05.0f'),'_Name'];
    if ~has_all_data
        load(filename,var_name);
    end
    if exist(var_name,'var')
        max_channel = max_channel+1;
    else
        test_next_channel = false;
    end
end
if max_channel == 0
    error('QuaDRiGa:qd_channel:mat_load',['??? File ''',filename,''' has no channel objects.']);
else
    groups = 1:max_channel;
end

% Dimension of Channel object
try
    if ~has_all_data
        load(filename,'ChannelDims');
    end
    channel_dims = double( ChannelDims );
catch
    channel_dims = [ size(groups),1,1];
end

% Parse dimensions
dims = channel_dims;
if nargin > 1
    for n = 2:nargin_hack
        if isempty( varargin{n} )
            if n-1>numel(dims)
                varargin{n} = 1;
                dims(n-1) = 1;
                channel_dims(n-1) = 1;
            else
                varargin{n} = 1:dims(n-1);
            end
        else
            value = varargin{n};
            if ~( numel(value) > 0 && isnumeric(value) && isreal(value) && all(mod(value,1)==0) )
                error('QuaDRiGa:qd_channel:mat_load','??? "dimension" must be integer and > 0')
            end
        end
    end
else
    nargin_hack = 1;
    load_only_par = false;
end

% Get the order of objects in the file
no_existing_channels = prod( channel_dims );
try
    if ~has_all_data
        load(filename,'Order');
    end
    order = Order( 1:no_existing_channels ).';
catch
    order = 1:no_existing_channels;
end

% Determine which channels to load
if nargin_hack > 1
    
    if numel( varargin{2} ) == 1 && varargin{2} == 0
        % If the second input is 0 then we dont load any channels but return
        % the size of the channel array.
        load_groups = [];
        h_channel = [];
        channel_dims = [0 0 0 0];
        
    elseif nargin_hack == 2 && max( varargin{2} ) <= no_existing_channels
        % Only one dimension is given --> linear index
        load_groups = varargin{2};
        channel_dims = [ 1,numel(load_groups) ];
        
    elseif nargin_hack == 3 && ...
            min( varargin{2} )>0 && max( varargin{2} ) <= channel_dims(1) && ...
            min( varargin{3} )>0 && max( varargin{3} ) <= channel_dims(2)
        % Two input arguments
        
        load_groups = [];
        for n = 1 : numel(varargin{3})
            load_groups = [ load_groups ,...
                (varargin{3}(n)-1)*channel_dims(1) + varargin{2} ];
        end
        channel_dims = [ numel( varargin{2} ) , numel( varargin{3}  ) , 1 , 1 ];
        
    elseif nargin_hack == 4 && ...
            min( varargin{2} )>0 && max( varargin{2} ) <= channel_dims(1) && ...
            min( varargin{3} )>0 && max( varargin{3} ) <= channel_dims(2) && ...
            min( varargin{4} )>0 && max( varargin{4} ) <= channel_dims(3)
        % Three input arguments
        
        load_groups = [];
        for m = 1 : numel(varargin{4})
            for n = 1 : numel(varargin{3})
                load_groups = [ load_groups ,...
                    (varargin{4}(m)-1)*channel_dims(1)*channel_dims(2) + ...
                    (varargin{3}(n)-1)*channel_dims(1) + varargin{2} ];
            end
        end
        channel_dims = [ numel( varargin{2} ) , numel( varargin{3} ) ,...
            numel( varargin{4} ) , 1 ];
        
    elseif nargin_hack == 5 && ...
            min( varargin{2} )>0 && max( varargin{2} ) <= channel_dims(1) && ...
            min( varargin{3} )>0 && max( varargin{3} ) <= channel_dims(2) && ...
            min( varargin{4} )>0 && max( varargin{4} ) <= channel_dims(3) && ...
            min( varargin{5} )>0 && max( varargin{5} ) <= channel_dims(4)
        % Four input arguments
        
        load_groups = [];
        for o = 1 : numel(varargin{5})
            for m = 1 : numel(varargin{4})
                for n = 1 : numel(varargin{3})
                    load_groups = [ load_groups ,...
                        (varargin{5}(o)-1)*channel_dims(1)*channel_dims(2)*channel_dims(3) + ...
                        (varargin{4}(m)-1)*channel_dims(1)*channel_dims(2) + ...
                        (varargin{3}(n)-1)*channel_dims(1) + varargin{2} ];
                end
            end
        end
        channel_dims = [ numel( varargin{2} ) , numel( varargin{3} ) ,...
            numel( varargin{4} ), numel(varargin{5}) ];
        
    else
        error('Out of range subscript.')
    end
else
    load_groups = 1:no_existing_channels;
end

% Reorder groups to match dimensions
tmp = order( load_groups );
tmp = tmp( tmp~=0 );
groups = groups( tmp );

% Load selected channels
for n = 1 : numel(groups)
    h_channel(1,n) = qd_channel;
    
    % Name
    var_name = ['channel_',num2str(groups(n),'%05.0f'),'_Name'];
    if ~has_all_data
        load(filename,var_name);
    end
    h_channel(1,n).name = eval(var_name);
    
    % Version
    var_name = ['channel_',num2str(groups(n),'%05.0f'),'_Version'];
    if ~has_all_data
        load(filename,var_name);
    end
    h_channel(1,n).version = eval(var_name);
        
    % Center Frequency
    var_name = ['channel_',num2str(groups(n),'%05.0f'),'_CenterFrequency'];
    try
        if ~has_all_data
            load(filename,var_name);
        end
        h_channel(1,n).center_frequency = eval(var_name);
    end
        
    if ~load_only_par
        % Coefficients
        var_name = ['channel_',num2str(groups(n),'%05.0f'),'_Coeffs'];
        try
            if ~has_all_data
                load(filename,var_name);
            end
            h_channel(1,n).coeff = eval(var_name);
        end
                
        % Delays
        var_name = ['channel_',num2str(groups(n),'%05.0f'),'_Delays'];
        try
            if ~has_all_data
                load(filename,var_name);
            end
            h_channel(1,n).delay = eval(var_name);
        end
                
        % Initial Position
        var_name = ['channel_',num2str(groups(n),'%05.0f'),'_Initial_position'];
        try
            if ~has_all_data
                load(filename,var_name);
            end
            h_channel(1,n).initial_position = eval(var_name);
        end
    end
    
    % RX Positions
    var_name = ['channel_',num2str(groups(n),'%05.0f'),'_rx_position'];
    try
        if ~has_all_data
            load(filename,var_name);
        end
        h_channel(1,n).rx_position = eval(var_name);
    end
        
    % TX Positions
    var_name = ['channel_',num2str(groups(n),'%05.0f'),'_tx_position'];
    try
        if ~has_all_data
            load(filename,var_name);
        end
        h_channel(1,n).tx_position = eval(var_name);
    end
        
    % Read "par" structure
    par = struct;
    selected_ds = {};
    if load_selected_par > 0  % Load only selected items
        for i_selds = 0 : nargin-load_selected_par
            selected_ds{i_selds+1} = varargin{load_selected_par+i_selds};
        end
    else
        try
            var_name = ['channel_',num2str(groups(n),'%05.0f'),'_PARnames'];
            if ~has_all_data
                load(filename,var_name);
            end
            selected_ds = eval(var_name);
        end
    end
    for i_ds = 1 : numel( selected_ds )
        var_name = ['channel_',num2str(groups(n),'%05.0f'),'_par_',selected_ds{i_ds}];
        try
            if ~has_all_data
                load(filename,var_name);
            end
            par.(selected_ds{i_ds}) = eval(var_name);
        end
    end
    if ~isempty( selected_ds )
        h_channel(1,n).par = par;
    end
end

if ~exist('h_channel','var') || isempty( h_channel )
    h_channel = qd_channel([]);
elseif numel(h_channel) == prod(channel_dims)
    h_channel = qf.reshapeo( h_channel , channel_dims );
end

% Enable warning
warning('on','MATLAB:load:variableNotFound');

end
