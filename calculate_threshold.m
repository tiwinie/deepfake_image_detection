clc;
clear;
close all;

cd(fileparts(mfilename('fullpath')));

real_folder = 'real';
fake_folder = 'fake';

real_files = dir(fullfile(real_folder, '*.jpg'));
fake_files = dir(fullfile(fake_folder, '*.jpg'));

real_scores = zeros(1, length(real_files));
fake_scores = zeros(1, length(fake_files));

fprintf("REAL IMAGES\n");

% real images
for i = 1:length(real_files)

    img_path = fullfile(real_folder, real_files(i).name);
    score = process_image(img_path);

    real_scores(i) = score;

    fprintf("Image: %s | Score: %.6f\n", real_files(i).name, score);
end


fprintf("\nFAKE IMAGES\n");

% fake images
for i = 1:length(fake_files)

    img_path = fullfile(fake_folder, fake_files(i).name);
    score = process_image(img_path);

    fake_scores(i) = score;

    fprintf("Image: %s | Score: %.6f\n", fake_files(i).name, score);
end


% threshold calculation
real_mean = mean(real_scores);
fake_mean = mean(fake_scores);

threshold = (real_mean + fake_mean) / 2;

fprintf("\nSuggested Threshold: %.6f\n", threshold);



function score = process_image(image_path)

    img = imread(image_path);

    % face detection
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, img);

    if ~isempty(bbox)
        face = imcrop(img, bbox(1,:));
    else
        face = img;
    end

    % eye detection
    eyeDetector = vision.CascadeObjectDetector('EyePairBig');
    eyeBox = step(eyeDetector, face);

    if ~isempty(eyeBox)
        eyes = imcrop(face, eyeBox(1,:));
    else
        eyes = face;
    end

    % convert to grayscale
    gray_face = im2double(rgb2gray(face));
    gray_eyes = im2double(rgb2gray(eyes));

    % SPN calculation
    smooth_face = imgaussfilt(gray_face, 1);
    spn_face = gray_face - smooth_face;

    smooth_eyes = imgaussfilt(gray_eyes, 1);
    spn_eyes = gray_eyes - smooth_eyes;

    var_face = var(spn_face(:));
    var_eyes = var(spn_eyes(:));

    spn_difference = abs(var_face - var_eyes);

    % edge density
    edge_map = edge(gray_face, 'Canny');
    edge_density = sum(edge_map(:)) / numel(edge_map);

    % final score
    score = (spn_difference * 100000) + edge_density;

end