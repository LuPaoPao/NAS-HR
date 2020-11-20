function [ Pos_signal ] = POS(RGB,fps)
%   此处显示详细说明
WinSec=1.6;%(based on refrence's 32 frame window with a 20 fps camera)
%lines and comments correspond to pseudo code algorithm on reference page 7       
N = size(RGB,1);%line 0 - RGB is of length N frames
H = zeros(1,N);%line 1 - initialize to zeros of length of video sequence
l = ceil(WinSec*fps);%line 1 - window length equivalent to reference: 32 samples of 20 fps camera (default 1.6s)
C = zeros(length(l),3);
for n = 1:N-1%line 2 - loop from first to last frame in video sequence
    %line 3 - spatial averaging was performed when video was read
    m = n - l + 1;%line 4 condition
    if(m > 0)%line 4
        Cn(1,:) = ( RGB(m:n,1)./ mean(RGB(m:n,1))' )';%line 5 - temporal normalization
        Cn(2,:) = ( RGB(m:n,2)./ mean(RGB(m:n,2))' )';
        Cn(3,:) = ( RGB(m:n,3)./ mean(RGB(m:n,3))' )';
        S = [0, 1, -1; -2, 1, 1] * Cn;%line 6 - projection
        h = S(1,:) + ((std(S(1,:)) / std(S(2,:))) * S(2,:));%line 7 - tuning
        H(m:n) = H(m:n) + (h - mean(h));%line 8 - overlap-adding
    end%line 9 - end if
end%line 10 - end for
Pos_signal=H;
end

