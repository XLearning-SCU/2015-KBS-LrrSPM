% =========================================================================
% An example code for the algorithm proposed in

% Xi Peng, Rui Yan, Bo Zhao, Huajin Tang, and Zhang Yi,
% Fast Low-rank Representation based Spatial Pyramid Matching for Image Classification,
% Knowledge based Systems, Volume 90, December 2015, Pages 14-22.

%    
% If the codes or data sets are helpful to you, please appropriately cite our works. Thank you very much!
% Written by Xi Peng @ I2R A*STAR,
% more information can be found from my website: http://machineilab.org/users/pengxi/ or www.pengxi.me
% Augest, 2014.
% =========================================================================

clear all; close all; clc;


% -------------------------------------------------------------------------
% set path
% addpath('./large_scale_svm/');        % we use Liblinear package, you need 
                                    % download and compile the matlab codes
addpath('../usages/SVM/');        % we use Liblinear package and compile it on MAC OS 64 bit. If you use other OS, please recompile it accordingly. 
addpath ('../usages/sift/')
addpath ('../usages/fastkmeans/')
addpath('../usages/large_scale_svm/');    % nonlinear SVM


% -------------------------------------------------------------------------
% parameter setting

% feature pooling parameters

skip_coding = false; 
lambda = 0.7;                      % regularization parameter for fast LRR
fprintf('**********SPM using fLRR when lambda = %f **********\n', lambda);
pyramid = [1, 2, 4];                % spatial block structure for the SPM
EngRatio = 0.98;                    % EngRatio=0 non-trucated; 
                                    % EngRatio：trucated the codes whose contribution is less than EngRatio.
c = 10;                             % regularization parameter for linear SVM
                                    % in Liblinear package

% sift descriptor extraction
skip_cal_sift = false;              % if 'skip_cal_sift' is false, set the following parameter
gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;    

% dictionary training for fLRR coding
skip_dic_training = true; % a tranined dictionary has been provided under the file fold 'dictionary'
nBases = 256;
nsmp = 200000;

% classification parameters  
nRounds = 10;                        % number of random test on the dataset
tr_num  = 100;                       % training examples per category
mem_block = 3000;                   % maxmum number of testing features loaded each time  

% directory setup
img_dir = '../image/';                  % directory for dataset images
data_dir = '../data/';                  % directory to save the sift features of the chosen dataset
dataSet = '8Scenes';
fea_dir = fullfile('./features/', dataSet, '/');   % directory to save the sift features of the chosen dataset in a single .mat

rt_img_dir = fullfile(img_dir, dataSet,'/');
rt_data_dir = fullfile(data_dir, dataSet, '/');

% -------------------------------------------------------------------------
% extract SIFT descriptors, we use Prof. Lazebnik's matlab codes in this package
% change the parameters for SIFT extraction inside function 'extr_sift'
% extr_sift(img_dir, data_dir);

% -------------------------------------------------------------------------
% retrieve the directory of the database and load the codebook

if skip_cal_sift       % load (or calcualte) the sift feature from a single .mat, multiple .mat
    database = retr_database_dir(rt_data_dir);
    fprintf(['load the sift features from ' rt_data_dir '\n']);
else
    [database, lenStat] = CalculateSiftDescriptor(rt_img_dir, rt_data_dir, gridSpacing, patchSize, maxImSize, nrml_threshold);
    fprintf(['calculated the sift features and stored it at ' rt_data_dir '\n']);
end;

if isempty(database),
    error('Data directory error!');
end

%% load fast LRR coding dictionary (one dictionary trained on Caltech101 is provided)

% Bpath = ['./dictionary/' dataSet '_SIFT_' num2str(nBases) '.mat'];
Xpath = ['../dictionary/rand_patches_' dataSet '_' num2str(nsmp) '.mat'];
Bpath = ['../dictionary/' dataSet '_SIFT_Kmeans_' num2str(nBases) '.mat'];

if ~skip_dic_training,
    try 
        load(Xpath);
    catch
        X = rand_sampling(database, nsmp);
        save(Xpath, 'X');
    end
    fprintf('dictionary learning begin!\n');
    [IDX,B,sumd] = kmeans2(X',nBases);
%     [label, B] = litekmeans(X', nBases, 'MaxIter', 50, 'Replicates', 10);
    B = B';
    save(Bpath, 'B');
    clear IDX sumd;
else
    load(Bpath);
end

nBases = size(B, 2);                    % size of the dictionary


load(Bpath);
nCodebook = size(B, 2);              % size of the codebook

%% calculate the features using fast LRR

dFea = sum(nCodebook*pyramid.^2);
nFea = length(database.path);

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % path for each image feature
fdatabase.label = zeros(nFea, 1);       % class label for each image feature

% calculate the projection matrix using the dictionary according to fLRR.
tic;
ProjM = inv(B'*B+lambda.*eye(size(B,2)));
ProjM = ProjM*B';

if ~skip_coding
%     if isdir(fea_dir),
%         rmdir(fea_dir,'s');
%     end 
    disp('==================================================');
    fprintf('Calculating the Low Rank Representation feature...\n');
    disp('==================================================');
    for iter1 = 1:nFea,  
        if ~mod(iter1, 20),
           fprintf('.');
        end
        if ~mod(iter1, 500),
            fprintf(' %d/%d images processed\n', iter1,nFea);
        end
        fpath = database.path{iter1};
        flabel = database.label(iter1);

        load(fpath);
        [rtpath, fname, fpostfix] = fileparts(fpath);
        if ~strcmp(fpostfix, '.mat')
            continue;
        end
        feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);

        fea = LrrSPM_pooling(feaSet, ProjM, pyramid, EngRatio);
        label = database.label(iter1);

        if ~isdir(fullfile(fea_dir, num2str(flabel))),
            mkdir(fullfile(fea_dir, num2str(flabel)));           
        end      
        save(feaPath, 'fea', 'label');  
        fdatabase.label(iter1) = flabel;
        fdatabase.path{iter1} = feaPath;
    end; 
else
    disp('\n==================================================');
    fprintf('retrieve the locality-constraint linear coding feature...\n');
    disp('==================================================');
    for iter1 = 1:nFea,  
        if ~mod(iter1, 5),
           fprintf('.');
        end
        if ~mod(iter1, 100),
            fprintf(' %d/%d images retrieved \n', iter1,nFea);
        end
        fpath = database.path{iter1};
        flabel = database.label(iter1);
        load(fpath);
        [rtpath, fname, fpostfix] = fileparts(fpath);
        if ~strcmp(fpostfix, '.mat')
             continue;
         end
        feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);

        fdatabase.label(iter1) = flabel;
        fdatabase.path{iter1} = feaPath;
    end; 
end;
% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code
time_coding = toc;

fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);
linear_accuracy = zeros(nRounds, 1);
kernel_accuracy = linear_accuracy;

time_linearSVM = zeros(nRounds, 1);
time_kernelSVM = zeros(nRounds, 1);

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    tic;
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    % load the training features 
    tr_fea = zeros(length(tr_idx), dFea);
    tr_label = zeros(length(tr_idx), 1);
    
    for jj = 1:length(tr_idx),
        fpath = fdatabase.path{tr_idx(jj)};
        load(fpath, 'fea', 'label');
        tr_fea(jj, :) = fea';
        tr_label(jj) = label;
    end
    
    options = ['-c ' num2str(c)];
    %% ---------------- training with linear SVM
    time_1 = toc;
    
    tic;
    fprintf('\n Training using linear SVM\n');
    model = train(double(tr_label), sparse(tr_fea), options);  
    time_linearSVM(ii) = time_1 + toc;
    
    tic;
    fprintf('\n Training using nonlinear SVM\n');
    [w, b, class_name] = li2nsvm_multiclass_lbfgs(tr_fea, tr_label, 0.1);  
    time_kernelSVM(ii) = time_1 + toc;
    
    clear tr_fea;  
    
    % load the testing features
    ts_num = length(ts_idx);
    ts_label = [];
    
    if ts_num < mem_block,
        % load the testing features directly into memory for testing
        ts_fea = zeros(length(ts_idx), dFea);
        ts_label = zeros(length(ts_idx), 1);
        tic;
        for jj = 1:length(ts_idx),
            fpath = fdatabase.path{ts_idx(jj)};
            load(fpath, 'fea', 'label');
            ts_fea(jj, :) = fea';
            ts_label(jj) = label;
        end
        time_1 = toc;
        tic;
        %% -------------- classification using linear SVM
        [LC] = predict(ts_label, sparse(ts_fea), model);  
        time_linearSVM(ii) = time_1 + toc;
        tic;
        [KC, Y] = li2nsvm_multiclass_fwd(ts_fea, w, b, class_name);   
        time_kernelSVM(ii) = time_1 + toc;
        clear Y;
    else
        % load the testing features block by block
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        LC = [];
        KC = [];
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            tic;
            % load the current block of features
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                curr_ts_label(kk) = label;
            end    
            % test the current block features
            ts_label = [ts_label; curr_ts_label];
            time_1 = toc;            
            %% -------------- classification using linear SVM
            tic;
            [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
            LC = [LC; curr_C];
            time_linearSVM(ii) = time_1 + toc + time_linearSVM(ii);
            %% -------------- classification using linear SVM
            tic;
            [curr_C, Y] = li2nsvm_multiclass_fwd(curr_ts_fea, w, b, class_name);   
            KC = [KC; curr_C];
            time_kernelSVM(ii) = time_1 + toc + time_kernelSVM(ii);
        end
        
        curr_ts_fea = zeros(rem_fea, dFea);
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
        tic;
        for kk = 1:rem_fea,
           fpath = fdatabase.path{curr_idx(kk)};
           load(fpath, 'fea', 'label');
           curr_ts_fea(kk, :) = fea';
           curr_ts_label(kk) = label;
        end  
        ts_label = [ts_label; curr_ts_label];
        time_1 = toc;

        %% -------------- classification using linear SVM
        tic;
        [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
        LC = [LC; curr_C];
        time_linearSVM(ii) = time_1 + toc + time_linearSVM(ii);
        %% -------------- classification using linear SVM
        tic;
        [curr_C, Y] = li2nsvm_multiclass_fwd(curr_ts_fea, w, b, class_name);   
        KC = [KC; curr_C];
        time_kernelSVM(ii) = time_1 + toc + time_kernelSVM(ii);
    end
    
    % normalize the classification accuracy by averaging over different
    % classes
    linear_acc = zeros(nclass, 1);
    kernel_acc = linear_acc;
    
    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = LC(idx);
        curr_gnd_label = ts_label(idx);    
        linear_acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
        
        curr_pred_label = KC(idx);
        curr_gnd_label = ts_label(idx);    
        kernel_acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);      
    end

    linear_accuracy(ii) = mean(linear_acc); 
    kernel_accuracy(ii) = mean(kernel_acc); 
   
    fprintf('Classification accuracy using linear SVM for round %d: %f\n', ii, linear_accuracy(ii));
    fprintf('Classification accuracy using nonlinear SVM for round %d: %f\n', ii, kernel_accuracy(ii));

end

clc;
fprintf('\n=====================Linear SVM==========================\n');
% fprintf('Max classification accuracy: %f\n', max(linear_accuracy)*100);
fprintf('Average classification accuracy: %f\n', mean(linear_accuracy)*100);
fprintf('STD classification accuracy: %f\n', std(linear_accuracy)*100);
fprintf('Time for coding: %f\n', mean(time_coding));
fprintf('Time for classification: %f\n', mean(time_linearSVM));

fprintf('\n=====================Kernel SVM==========================\n');
fprintf('Max classification accuracy: %f\n', max(kernel_accuracy));
fprintf('Average classification accuracy: %f\n', mean(kernel_accuracy));
fprintf('STD classification accuracy: %f\n', std(kernel_accuracy));
fprintf('Time for coding: %f\n', mean(time_coding));
fprintf('Time for classification: %f\n', mean(time_kernelSVM));
fprintf('===============================================\n');

clear time_1 LC KC ts_fea ts_idx fdatabse B block_idx C clabel curr_C curr_gnd_label curr_idx curr_pred_label curr_ts_fea curr_ts_label;
clear ProjM database  fea feaSet fpostfix idx idx_rand idx_label ii iter1 jj kk;
clear b classname X ProjM label model nclass options tr_idx tr_label ts_label;
save (['LrrSPM_' dataSet '_nBases' num2str(nBases) '_lambda' num2str(lambda) '_EngRatio' num2str(EngRatio) '_tr' num2str(tr_num) '.mat']);
