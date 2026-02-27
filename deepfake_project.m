clc;
clear;
close all;

threshold = 23.034051; %calculated based on uploaded image data in calculate_threshold file

% input image
img_path = '/MATLAB Drive/deepfake_project/ex_deepfakes.png';

score = process_image(img_path);

fprintf("\nFinal Score: %.6f\n", score);

if score > threshold
    fprintf("Result: REAL IMAGE\n");
else
    fprintf("Result: FAKE IMAGE\n");
end


% main function

function score = process_image(image_path)

    img = imread(image_path);

    % Face detection
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, img);

    if ~isempty(bbox)
        face = imcrop(img, bbox(1,:));
    else
        face = img;
    end

    % Eye detection
    eyeDetector = vision.CascadeObjectDetector('EyePairBig');
    eyeBox = step(eyeDetector, face);

    if ~isempty(eyeBox)
        eyes = imcrop(face, eyeBox(1,:));
    else
        eyes = face;
    end

    % Convert to grayscale
    gray_face = rgb2gray(face);
    gray_eyes = rgb2gray(eyes);

    gray_face = im2double(gray_face);
    gray_eyes = im2double(gray_eyes);

    % spn analysis
    smooth_face = imgaussfilt(gray_face, 1);
    spn_face = imsubtract(gray_face, smooth_face);

    smooth_eyes = imgaussfilt(gray_eyes, 1);
    spn_eyes = imsubtract(gray_eyes, smooth_eyes);

    var_face = var(spn_face(:));
    var_eyes = var(spn_eyes(:));

    spn_difference = abs(var_face - var_eyes);

    % edge analysis (blur)
    edge_map = edge(gray_face, 'Canny');
    edge_density = sum(edge_map(:)) / numel(edge_map);

    % final score
    score = (spn_difference * 100000) + edge_density;

    % results:
    figure;

    subplot(2,2,1);
    imshow(face);
    title('Detected Face');

    subplot(2,2,2);
    imshow(eyes);
    title('Detected Eyes');

    subplot(2,2,3);
    imshow(mat2gray(spn_face));
    title('SPN (Face Noise Pattern)');

    subplot(2,2,4);
    imshow(edge_map);
    title('Edge Map (Canny)');

    fprintf("SPN Variance (Face): %.8f\n", var_face);
    fprintf("SPN Variance (Eyes): %.8f\n", var_eyes);
    fprintf("SPN Difference: %.8f\n", spn_difference);
    fprintf("Edge Density: %.6f\n", edge_density);

end