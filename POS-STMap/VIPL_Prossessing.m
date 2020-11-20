clc
clear;
warning off
source1 = {'s1','s2','s3'};
source2 = {'source1','source2','source3'};
% landmark path seeta face
Lmk_root_path = 'D:\HR_landmarks\SDK_smooth';
% vedio path
video_root_path = 'H:\VIPL_HR';
save_path ='D:\Data\VIPL_my15_YUV_LB\';
% person index
p_path = dir(Lmk_root_path);
p_path = sort_nat({p_path.name});
p_path = p_path(3:end);
lmk_num = 81;
count = 0;
dizhen = 0;
for p_index = 1:length(p_path)
    p_now = strcat(Lmk_root_path,'/', cell2mat(p_path(p_index)));
    v_path = dir(p_now);
    v_path = sort_nat({v_path.name});
	v_path = v_path(3:end);
    % vedio
    p_now_vedio = strcat(video_root_path,'/', cell2mat(p_path(p_index)));
    v_path_vedio = dir(p_now_vedio);
    v_path_vedio = sort_nat({v_path_vedio.name});
	v_path_vedio = v_path_vedio(3:end);  
    v_length= min(length(v_path),length(v_path_vedio));
    
    for v_index = 1:v_length
        v_now = strcat(p_now,'/', cell2mat(v_path(v_index)));
        s_path = dir(v_now);
        s_path = sort_nat({s_path.name});
        s_path = s_path(3:end);
        % vedio
        v_now_vedio = strcat(p_now_vedio,'/', cell2mat(v_path_vedio(v_index)));
        s_path_vedio = dir(v_now_vedio);
        s_path_vedio = sort_nat({s_path_vedio.name});
        s_path_vedio = s_path_vedio(3:end);
        % 保证s相同
        falg1 = zeros(3,1);
        falg2 = zeros(3,1);
        for source_kind = 1:3
           for pat = 1:length(s_path)
               if(cell2mat(source1(source_kind))==cell2mat(s_path(pat)))
                  falg1(source_kind) = 1;
               end
           end
           for pat = 1:length(s_path_vedio)
               if(cell2mat(source2(source_kind))==cell2mat(s_path_vedio(pat)))
                  falg2(source_kind) = 1;
               end
           end 
        end
        s_real = find(falg1==falg2&falg1==1);
        for s_index = 1:length(s_real)
            s_now = strcat(v_now,'/', cell2mat(source1(s_real(s_index))));
            % vedio
            s_now_vedio = strcat(v_now_vedio,'/', cell2mat(source2(s_real(s_index))));
            % now path
            landmark_path = strcat(s_now,'/', 'face_landmarks');
            vidio_path = strcat(s_now_vedio,'/', 'video.avi');
            HR_path = strcat(s_now_vedio,'/', 'gt_HR.csv');
            SpO2_path = strcat(s_now_vedio,'/', 'gt_SpO2.csv');
            wave_path = strcat(s_now_vedio,'/', 'wave.csv');
            % vidio
            obj = VideoReader(vidio_path);
            numFrames = obj.NumberOfFrames;
            numlandmarks = length(dir(landmark_path))-2;
            if numFrames==numlandmarks && numFrames>520
                count = count+1
                signal = [];
                % get HR/SpO2/
                HR = csvread(HR_path,1,0);
                SpO2 = csvread(SpO2_path,1,0);
                wave = csvread(wave_path,1,0);
                % 帧率
                fps = (numFrames/length(HR));
                if (fps > 15)
                    % save path
                    dst_path = strcat(save_path,p_path(p_index),v_path(v_index),s_path(s_index));
                    % save wave as pic
                    wave2Pic(wave,numFrames,dst_path);
                    % save STmap as pic
                    POS_STMap(vidio_path, landmark_path, dst_path, fps);
                    % save HR,SpO2,fps
                    % spline HR/SPO2
                    HR_size = size(HR);
                    SpO2_size = size(SpO2);
                    w_x = linspace(0,100,HR_size(1));
                    w_x1 = linspace(0,100,SpO2_size(1));
                    d_x = linspace(0,100,numFrames-1);
                    HR = spline(w_x,HR,d_x);
                    SpO2 = spline(w_x1,SpO2,d_x);
                    fps_path = strcat(cell2mat(dst_path), '/fps.mat');    
                    HR_path = strcat(cell2mat(dst_path), '/HR.mat'); 
                    SpO2_path = strcat(cell2mat(dst_path), '/SpO2.mat');
                    eval(['save ', fps_path, ' fps']);
                    eval(['save ', HR_path, ' HR']);
                    eval(['save ', SpO2_path, ' SpO2']);                 
                end
            else
%                 numFrames
%                 numlandmarks
            end
        end
    end
end
