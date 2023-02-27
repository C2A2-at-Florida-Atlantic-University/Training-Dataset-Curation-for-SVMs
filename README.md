# Training-Dataset-Curation-for-SVMs
Support Vector Machines, popularly known as SVMs, have been one of the best machine learning models to
solve classification problems. Although widely used and applied in real-world applications, SVMs are susceptible
to outliers or mislabeled examples in training datasets. The presence of a few anomalies in the training datasets
can affect the decision boundaries created by the SVM model, thereby affecting the model’s performance on the
unseen data. <br />
A new algorithm is proposed in journal "Training Dataset Curation for Support Vector 
Machines by L1-norm Joint Threshold-Rank Selection using Two-Line Fitting Method" to filter such atypical data instances before the training phase
of SVM and thus retain the good training examples to provide the classifier with better support vector candidates
for making classification boundaries. This data repository contains the datasets used to evaluate the efficiency of the proposed algorithm by implementing it in real-world classification problems. Following four publicly available datasets are used to compare the performance of the SVM classifier on the curated dataset versus the raw dataset. <br />
- MNIST Database of Handwritten Digits, <br />
* Iris Data Set <br />
+ Breast Cancer Wisconsin Data Set, and <br />
- San Diego Daily Weather Data Set <br /> <br />
For each of the above, ten different sets of the train and test files are created randomly by splitting
the source data into 70 - 30% respectively. For MNIST dataset a training file have been added with 20% of random label noise and stored in separate folder. Similarly for IRIS dataset 10% of random label noise to training dataset and stored in separate folder.
