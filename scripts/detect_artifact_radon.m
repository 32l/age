function score = detect_artifact_radon(I, psz)
    I = I(1:psz,: ,:);
    g = double(rgb2gray(I));
    
%     g = g/255;
    g = (g - mean(g(:)))/std(g(:));
    
    % edge
    
    
    theta = 1:180;
    R = radon(g, theta);
    
%     r = max(R, [], 1) - min(R, [], 1);
    % r = std(R, [], 1);
    
    r = mean(abs(R(1:end-1,:) - R(2:end,:)));
    
%     figure();
%     subplot(1,3,1);
%     imshow(I);
%     subplot(1,3,2);
%     plot(r);
%     subplot(1,3,3);
%     imshow(R/max(R(:)));
    
    score = min(r);
end
