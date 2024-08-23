% Main script section (calls the function)
% this scripts is implementation of journal 
% "Training Dataset Curation by L1-norm Principal-Component Analysis for Support Vector Machines"
% Finds ouliers and mislabled examples training data sets for SVMs to
% enhance the model's performance in presence of mislabeling.
% % % % % % % % Input Parameters are :
% % % % % % % % input_file  : provide file name
% % % % % % % % target_col_idx : index of the target column
% % % % % % % % header : number of header lines in input file
% % % % % % % % percent_mislabel : percentage of mislabeling you want to induce in your file for test this algorithm
% % % % % % % % numclass :  number of classes in the data
% Below are some of the example files provided and their parameters:
% % % % % % % % input_file='Iris.csv';target_col_idx=5;header=1;percent_mislabel=15;numclass=3
% % % % % % % % input_file='cell_samples.csv';target_col_idx=10;header=1;percent_mislabel=15;numclass=2;
% % % % % % % % input_file='wine.csv';target_col_idx=1;header=1;percent_mislabel=15;numclass=3
% % % % % % % % input_file='penguins.csv';target_col_idx=1;header=1;percent_mislabel=15;numclass=3
%%%%% Add adition files in Input_Files folder and 
%%%%% run the script from l1pca_data_curation_matlab folder




%%Define the input parameters
input_file = 'Iris.csv'; % Replace with your actual file path
target_col_idx = 5; %  index of the target column
header = 1; %number of header lines
percent_mislabel = 25; %  percentage of mislabeling you want to induce in your file for test this algorithm
numclass = 3; %  number of classes in the data


% Call the function
[testAccuracy]=data_curation_l1pca(input_file, target_col_idx, header, percent_mislabel, numclass);
disp(['Test Accuracy: ', num2str(testAccuracy * 100), '%']);

% End of main script 


% Function definition
function [testAccuracy]=data_curation_l1pca(input_file,target_col_idx,header,percent_mislabel,numclass)
    cd 'Input_Files'
    mfile = readmatrix(input_file,"NumHeaderLines",header);
    m = mfile(sum(isnan(mfile),2)==0,:); % Removing nan values
    % Place the target col as the last col of input file
    % Extract the target column
    target_col = m(:, target_col_idx);

    % Create a new matrix without the target column
    m(:, target_col_idx) = [];

    % Append the extracted column to the end of the matrix
    m = [m, target_col];

    % Split the data into training and test sets
    [train_set, test_set] = train_test_split(m);
    cd ..
    % Mislabel the training set
    [mislabel_train] = mislabel_mat(train_set, numclass, percent_mislabel);

    % Perform outlier excision on training data
    [clean_mat] = outlier_excision(mislabel_train, numclass);

    % Prepare features and labels for SVM
    X = clean_mat(:, 1:end-1);
    Y = clean_mat(:, end);

    % Train an SVM model with RBF kernel and perform cross-validation
    SVMModel = fitcecoc(X, Y, ...
        'Learners', templateSVM('KernelFunction', 'RBF', 'KernelScale', 'auto', 'Standardize', true),'KFold', 5);

    % Compute cross-validation loss
    crossValLoss = kfoldLoss(SVMModel);

    % Train the final model on the full dataset
    finalModel = fitcecoc(X, Y, ...
        'Learners', templateSVM('KernelFunction', 'RBF', 'KernelScale', 'auto', 'Standardize', true));

    % Predict labels for the test data
    XTest = test_set(:, 1:end-1);
    YTest = test_set(:, end);
    predictedLabels = predict(finalModel, XTest);

    % Evaluate the test accuracy
    testAccuracy = sum(predictedLabels == YTest) / length(YTest);
    %disp(['Test Accuracy: ', num2str(testAccuracy * 100), '%']);
end

