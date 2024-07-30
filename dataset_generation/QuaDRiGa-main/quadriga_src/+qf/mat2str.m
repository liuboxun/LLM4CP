function str = mat2str( dat, name, line, format )
%MAT2STR Converts numeric data into M-code
%
% Description:
%   This function converts numeric data into M-code that can be used in MATLAB/Octave to be
%   executed on the command line or embedded into scripts and functions.
%
% Input:
%   dat
%   A multi-dimensional array containing the real-valued data. The maximum number of dimension is 3.
%
%   name
%   The variable name (string) to be written to the output.
%
%   line
%   Number of characters per line. Default: 100
%
%   format
%   Format of the output fields, specified as a string. See "num2str" for reference.
%
% Output:
%   str
%   Formatted output string.
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

if ~exist( 'name','var' ) || isempty( name )
    name = 'data';
end
if ~exist( 'format','var' ) || isempty( format )
    format = '%1.1f';
end
if ~exist( 'line','var' ) || isempty( line )
    line = 100;
end

sic = size(dat);

if any( sic == 1 )
    if sic(1) == 1
        brk = ',';
    else
        brk = ';';
    end
    
    if all( dat < 0 )
        str = [ name , ' = -['];
        dat = -dat;
    else
        str = [ name , ' = ['];
    end
    new_line = '    ';
    cnt = numel( str );
    for n = 1 : numel( dat)
        num = sprintf( format, dat(n) );
        
        compress = true;
        while compress
            ii = regexp(num,'\.', 'once');
            if ~isempty( ii ) && ( num(end) == '0' || num(end) == '.' )
                num = num(1:end-1);
            else
                compress = false;
            end
        end
        
        strn = [num , brk ];
        str = [ str, strn ];
        cnt = cnt + numel( strn );
        if cnt > line && n ~= numel( dat)
            str = [ str,'...\n',new_line ];
            cnt = numel( new_line );
        end
    end
    str = [ str(1:end-1) , '];\n' ];
    
else
    % Recursive call
    str = '';
    if numel( sic ) > 2 && sic(3) > 1
        for m = 1:sic(3)
            for n = 1:sic(1)
                
                name_new = [ name,'(',num2str(n),',:,',num2str(m),')' ];
                str = [str, qf.mat2str( dat(n,:,m), name_new, line, format ) ];
            end
        end
    else
        for m = 1 : sic(1);
            name_new = [ name,'(',num2str(m),',:)' ];
            str = [str, qf.mat2str( dat(m,:), name_new, line, format ) ];
        end
    end
end

if nargout ~= 1
    fprintf( str );
end

end

