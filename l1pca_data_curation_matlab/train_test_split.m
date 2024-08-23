function [train_set, test_set] = train_test_split(input_mat)
    % This function splits the input matrix into training and test sets
    % according to the specified percentage of test data.

    m = input_mat;
    percent = 30; % Percentage of test data
    [len, ~] = size(m); % Total number of records
    nrow = round(percent / 100 * len); 
    idx = randsample(1:size(m, 1), nrow); % Randomly selecting rows for test data
    test_set = m(idx, :); % Test data
    m(idx, :) = []; % Remove selected rows from the original matrix
    train_set = m; % Remaining data is the train data
end
