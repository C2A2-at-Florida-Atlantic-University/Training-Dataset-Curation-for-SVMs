function [mislabel_mat]    = mislabel_mat(train_set,numclass,percent)   
    final_mat=[];
    mat_new=[];
    % size(train_set)
    % distinct_values = unique(train_set(:,end))
    for j=1:numclass
        
        mat=train_set(train_set(:,end)==j,:);
        [n,~] =size(mat);
        nrow = round(percent/100*n);
        mat_cor_rec = randsample(1:size(mat,1),nrow) ;
        mat_cor=mat(mat_cor_rec,:) ;
        mat(mat_cor_rec,:)=[];
        for i=1: nrow
            class=setdiff(1:numclass, j);
            mat_cor(i,end)=class(randi(length(class),1));
        end
        mat_new=[mat; mat_cor];
        final_mat=[final_mat; mat_new];
        
    end
    mislabel_mat=final_mat;
end