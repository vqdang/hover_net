% original file, as provided by MoNuSeg organizers (Kumar et. al.)

%%%%% Comment from reposit author
% The aggregated_jaccard_index.m has a bug in the following portion 
% >    % Find best matched nuclei
% >    [maxji, indmax] = max(JI);
% >    predicted_nuclei = 1*(predicted_map == indmax);
% >    % update intersection and union pixel counts
% >    overall_correct_count = overall_correct_count + nnz(and(gt,predicted_nuclei));
% >    union_pixel_count = union_pixel_count + nnz(or(gt,predicted_nuclei));
% >    
% >    % remove used predicted nuclei from predicted map
% >    predicted_map = (1 - predicted_nuclei).*predicted_map;   
% 
% In the case with no intersection between all predicted segments against current
% ground truth segment under consideration
% >    % Find best matched nuclei
% >    [maxji, indmax] = max(JI);
% >    predicted_nuclei = 1*(predicted_map == indmax);
% will default to first location in the the predicted segmented array 
% (index 1 in matlab and 0 in python)
% 
% As a result, the nuclei with ID 1 will be accidentally included togther with the
% current groundtruth to the overall union pixel count (more penalty)
% >    union_pixel_count = union_pixel_count + nnz(or(gt,predicted_nuclei));        
% and also be removed from later calculation (prevent later correct intersection)
% >    predicted_map = (1 - predicted_nuclei).*predicted_map;   

clear all
clc

% Read the ground truth matrix where each nuclei is indexed by a
% unique integer
load('TCGA-18-5592-01Z-00-DX1_gt_map.mat');
% maximum number of predicted nuclei
ngt = nnz(unique(gt_map)); % pixels labeled 0 are background

% Read the predicted matrix where each predicted nuclei is indexed by a
% unique integer
load('TCGA-18-5592-01Z-00-DX1_predicted_map.mat');
% maximum number of predicted nuclei
npredicted = nnz(unique(predicted_map)); % pixels labeled 0 are background

% intialize intersection and union pixel counts to zero
overall_correct_count = 0;
union_pixel_count = 0;

% select ground truth nuclei one at a time for computing AJI
for i=1:ngt %for each ground truth nuclei 
    fprintf('Processing object # %d \n',i);
    % ground truth nuclei
    gt = 1*(gt_map == i);
    % Compute JI of each gt with every predicted nuclei
    JI = zeros(nnz(npredicted),1);
    for j = 1:npredicted
        % extract j-th predicted nuclei
        predicted_nuclei = 1*(predicted_map == j);
        % compute ratio of cardinalities of intersection and union pixels 
        JI(j) = nnz(and(gt,predicted_nuclei))/nnz(or(gt,predicted_nuclei));
    end
    
    % Find best matched nuclei
    [maxji, indmax] = max(JI);
    predicted_nuclei = 1*(predicted_map == indmax);
    % update intersection and union pixel counts
    overall_correct_count = overall_correct_count + nnz(and(gt,predicted_nuclei));
    union_pixel_count = union_pixel_count + nnz(or(gt,predicted_nuclei));
    
    % remove used predicted nuclei from predicted map
    predicted_map = (1 - predicted_nuclei).*predicted_map;  
end

% add all unmatched pixels left in the predicted map to union set
union_pixel_count = union_pixel_count + nnz(predicted_map);
aji = overall_correct_count/union_pixel_count;

