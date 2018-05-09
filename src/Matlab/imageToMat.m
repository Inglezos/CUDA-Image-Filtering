k = double(imread('images.png'));
k = k./256;

imageLength = 120;
for i=1:imageLength
    for j=1:imageLength
        image(i,j) = k(i,j);
    end
end

save image.mat image

clear all
