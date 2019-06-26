% Script to computer Aggregated Jaccard Index
% Created by Ruchika Verma, please cite the following paper if you use this code-
% N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, 
% "A Dataset and a Technique for Generalized Nuclear Segmentation for 
% Computational Pathology," in IEEE Transactions on Medical Imaging, 
% vol. 36, no. 7, pp. 1550-1560, July 2017

%%%%% Comment from reposit author
% final version, distributed on 2018-07-17

% Read the ground truth matrix where each nuclei is indexed by a
% unique integer
load('TCGA-18-5592-01Z-00-DX1_gt_map.mat');
% maximum number of predicted nuclei
gt_list = unique(gt_map); % set of unique gt nuclei
gt_list = gt_list(2:end); % exclude 0
ngt = numel(gt_list);

% Read the predicted matrix where each predicted nuclei is indexed by a
% unique integer
load('TCGA-18-5592-01Z-00-DX1_predicted_map.mat');
% maximum number of predicted nuclei
pr_list = unique(predicted_map); % ordered set of unique gt nuclei
pr_list = pr_list(2:end); % exclude 0
% mark used nuclei by the number of uses (you can use any other criteria) 
pr_list =  [pr_list, zeros(size(pr_list,1),1)];
npredicted = numel(pr_list(:,1));

% intialize intersection and union pixel counts to zero
overall_correct_count = 0;
union_pixel_count = 0;
%gt_list = gt_list(10:12);
i = length(gt_list); % nuclei may not be unique integers from 1 to n
while ~isempty(gt_list) % loops over each gt
    fprintf('Processing object # %d \n',i);

    % ground truth nuclei
    gt = 1*(gt_map == gt_list(i));
    %% Compute JI of each gt with matched predicted nuclei,
    % A predicted nuclei is allowed to match multiple gts 
    predicted_match = gt.*predicted_map;
    % If there is no predicted nuclei for a gt then it's a false negative
    if nnz(predicted_match) == 0
        union_pixel_count = union_pixel_count + nnz(gt);
        gt_list(i) = []; % remove it from the list
        i  = length(gt_list); % move to next nuclei
    else    
    % If more than one predicted nuclei matches a gt - choose max JI
        predicted_nuc_index = unique(predicted_match);
        predicted_nuc_index = predicted_nuc_index(2:end); % exclude 0 from unique set
        % you can make it even faster if only one match is identified
        JI = 0;
        for j = 1:length(predicted_nuc_index)
            matched = 1*(predicted_map == predicted_nuc_index(j));
            nJI =   nnz(and(gt,matched))/nnz(or(gt,matched));
            if nJI > JI
            best_match = predicted_nuc_index(j);
            JI = nJI;
            end
        end 
        
        predicted_nuclei = 1*(predicted_map == best_match);
    
    % update intersection and union pixel counts
        overall_correct_count = overall_correct_count + nnz(and(gt,predicted_nuclei));
        union_pixel_count = union_pixel_count + nnz(or(gt,predicted_nuclei));
    
    % remove used gt from the list
        gt_list(i) = [];
    % move to next nuclei
        i  = length(gt_list); 
    % update the count for used predicted nuclei
    index = find(pr_list(:,1) == best_match);
    pr_list(index,2) = pr_list(index,2) + 1;
    end
end

% Find all unused nuclei
unused_nuclei_list = find(pr_list(:,2) == 0);
% add all unmatched pixels left in the predicted map to union set
for k = 1:numel(unused_nuclei_list)
    unused_nuclei = 1*(predicted_map == pr_list(unused_nuclei_list(k),1));
    union_pixel_count = union_pixel_count + nnz(unused_nuclei);
end
aji = overall_correct_count/union_pixel_count;
   

    
    
    
    
    
    
    
    
    
    