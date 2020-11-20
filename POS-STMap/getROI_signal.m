function [ signals ] = getROI_signal(frame,landmarks)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
R = frame(:,:,1);
G = frame(:,:,2);
B = frame(:,:,3);
%Conversion Formula
Y = 0.299 * R + 0.587 * G + 0.114 * B;
U =128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
V =128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
frame(:,:,1) = Y;
frame(:,:,2) = U;
frame(:,:,3) = V;

ROI = zeros(14,3);
signals = zeros(1,45);
d = landmarks(61) - landmarks(49);
BW=uint8(roipoly(frame,[landmarks(49),landmarks(61),landmarks(61),landmarks(48)],[landmarks(50),landmarks(62),landmarks(62)-round(d*0.5),landmarks(50)-round(d*0.5)]));
numpix = sum(sum(BW));
for i =1:3
    imgROI=squeeze(frame(:,:,i)).*BW;%将Image0图像提取ROI，其余部分归零
    signals(1,i*15) = sum(sum(imgROI))/numpix;
end
for i=1:2
    ROI(1,(i-1)*3+1:(i-1)*3+3)=[landmarks(3+i-1),landmarks(121+i-1),landmarks(131+i-1)];
    ROI(2,(i-1)*3+1:(i-1)*3+3)=[landmarks(23+i-1),landmarks(123+i-1),landmarks(147+i-1)];
    ROI(3,(i-1)*3+1:(i-1)*3+3)=[landmarks(81+i-1),landmarks(131+i-1),landmarks(133+i-1)];
    ROI(4,(i-1)*3+1:(i-1)*3+3)=[landmarks(81+i-1),landmarks(133+i-1),landmarks(135+i-1)];
    ROI(5,(i-1)*3+1:(i-1)*3+3)=[landmarks(81+i-1),landmarks(135+i-1),landmarks(137+i-1)];
    ROI(6,(i-1)*3+1:(i-1)*3+3)=[landmarks(81+i-1),landmarks(137+i-1),landmarks(139+i-1)];
    ROI(7,(i-1)*3+1:(i-1)*3+3)=[landmarks(83+i-1),landmarks(147+i-1),landmarks(149+i-1)];
    ROI(8,(i-1)*3+1:(i-1)*3+3)=[landmarks(83+i-1),landmarks(149+i-1),landmarks(151+i-1)];
    ROI(9,(i-1)*3+1:(i-1)*3+3)=[landmarks(83+i-1),landmarks(151+i-1),landmarks(153+i-1)];
    ROI(10,(i-1)*3+1:(i-1)*3+3)=[landmarks(83+i-1),landmarks(153+i-1),landmarks(155+i-1)];
    ROI(11,(i-1)*3+1:(i-1)*3+3)=[landmarks(93+i-1),landmarks(139+i-1),landmarks(143+i-1)];
    ROI(12,(i-1)*3+1:(i-1)*3+3)=[landmarks(95+i-1),landmarks(155+i-1),landmarks(159+i-1)];
    ROI(13,(i-1)*3+1:(i-1)*3+3)=[landmarks(111+i-1),landmarks(129+i-1),landmarks(143+i-1)];
    ROI(14,(i-1)*3+1:(i-1)*3+3)=[landmarks(111+i-1),landmarks(129+i-1),landmarks(159+i-1)];
end
for ROI_Index = 1:14
    BW=uint8(roipoly(frame,ROI(ROI_Index,1:3),ROI(ROI_Index,4:6)));%生成掩膜图像BW1，使得BW1格式与Image0一致。
%     subplot(3,2,1),imshow(BW*255); 
    numpix = sum(sum(BW));
    R = frame(:,:,1);
    G = frame(:,:,2);
    B = frame(:,:,3);
    %Conversion Formula
    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    U =128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
    V =128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
    frame(:,:,1)=Y;
    frame(:,:,2)=U;
    frame(:,:,3)=V;
    for i =1:3
       imgROI=squeeze(frame(:,:,i)).*BW;%将Image0图像提取ROI，其余部分归零
%        subplot(3,2,i+2),imshow(imgROI); 
       signals(1,ROI_Index + (i-1)*15) = sum(sum(imgROI))/numpix;
    end
end
end

