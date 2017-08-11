img_root = '/data2/ynli/age/sfa_age/';

% 
% f = fopen('../datasets/youtube_face/Meta/image.lst', 'r');
% 
% n = int32(str2num(fgetl(f)));
% 
% img_lst = cell(n, 1);
% score = zeros(n,1);
% 
% for i = 1:n
%     img_lst{i} = fgetl(f);
% end
% 
% fclose(f);
% 
% for i = 1:length(img_lst)
%     fn = [img_root , img_lst{i}];
%     img = imread(fn);
%     s = detect_artifact_radon(img, 50);
%     score(i) = s;
%     fprintf('[%.2f%%] %f, %s\n',(100.0 *i/n), s, fn);
% end

% save('score.mat', 'score')

f = fopen('artificial_score.txt', 'w');
for i = 1:n
    fprintf(f, '%s %f\n', img_lst{i}, score(i));
end
fclose(f);


% img_lst = ['video_pose_10000/10120/1718/51.jpg',
%  'video_pose_10000/10048/1684/11.jpg',
%  'video_pose_10000/10040/1475/253.jpg',
%  'video_pose_10000/10108/1149/208.jpg',
%  'video_pose_10000/10185/4203/197.jpg',
%  'video_pose_10000/10079/259/86.jpg',
%  'video_pose_10000/10127/1913/233.jpg',
%  'video_pose_10000/10089/730/316.jpg',
%  'video_pose_10000/10040/1475/261.jpg',
%  'video_pose_10000/10350/9476/440.jpg'
% ]
