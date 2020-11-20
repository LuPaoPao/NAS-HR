function wave2Pic(wave,numFrames,dst_path)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
dst_path = cell2mat(dst_path);
if ~exist(dst_path)
    mkdir(dst_path);
end
img_path = strcat(dst_path, '\wave.png');
waveSize = size(wave);
w_x = linspace(0,100,waveSize(1));
d_x = linspace(0,100,numFrames);
wave_spline = spline(w_x,wave,d_x);
wave_spline = movmean(wave_spline,5);
wave_final(1,:,1) = wave_spline;
wave_final(1,:,2) = wave_spline;
wave_final(1,:,3) = wave_spline;

for c = 1:3
    temp = wave_final(1,:,c);
    temp = movmean(temp,3);
    wave_final(1,:,c) = (temp - min(temp))/(max(temp) - min(temp))*255;
end

img = wave_final;
imwrite(uint8(img), img_path);


end

