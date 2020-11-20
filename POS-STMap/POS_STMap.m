function POS_STMap(video_path, landmark_path, dst_path, fps)

fn=fps;
ap=0.3;
as=10;
wp=0.6;
ws=4; 
lmk_num = 81;
wpp=wp/(fn/2);
wss=ws/(fn/2);
[n,wn]=buttord(wpp,wss,ap,as); 
[b,a]=butter(n,wn); 
dst_path = cell2mat(dst_path);
if ~exist(dst_path)
    mkdir(dst_path);
end
obj = VideoReader(video_path);
numFrames = obj.NumberOfFrames;
signal = [];
% get signal
for k = 1:numFrames
    fid = fopen(strcat(landmark_path,'/', 'landmarks', num2str(k), '.dat'), 'r');
    if fid > 0
        landmarks = fread(fid,inf,'int');
        fclose(fid);
    else
        landmarks = zeros(lmk_num*2, 1);    
    end
    frame = read(obj,k); 
    s = getROI_signal(frame,landmarks);
    signal = [signal;s];
end
a = size(signal);
Combine_channel=zeros(15,a(1),3);
Combine_channel(:,:,1) = signal(:,1:15)';
Combine_channel(:,:,2) = signal(:,16:30)';
Combine_channel(:,:,3) = signal(:,31:45)';
for n=1:15
Combine_channel(n,:,3) = POS(squeeze(Combine_channel(n,:,:)),fps);
end
save_map_fullTime(dst_path, Combine_channel)
% fps_path = strcat(dst_path, '/fps.mat');
% eval(['save ', fps_path, ' fps']);
end