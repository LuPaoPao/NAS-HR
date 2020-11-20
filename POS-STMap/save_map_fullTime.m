function save_map_fullTime(HR_train_path, SignalMap)

final_signal = SignalMap;
img_path = strcat(HR_train_path, '\img_mvavg_full.png');
channel_num = size(SignalMap,3);
judge = mean(final_signal,1);   
if ~isempty(find(judge(1,:,2) == 0))
     a = 0;
else 
    final_signal1 = final_signal;
    for idx = 1:size(final_signal,1)
        for c = 1:channel_num
            temp = final_signal(idx,:,c);
            % temp = movmean(temp,3);
            final_signal1(idx,:,c) = (temp - min(temp))/(max(temp) - min(temp))*255;
        end
    end

    final_signal1 = final_signal1(:,:,[1 2 3]);
    img1 = final_signal1;
    imwrite(uint8(img1), img_path);
end