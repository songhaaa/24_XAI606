# 242R 신경망응용및실습(Applications and Practice in Neural Networks)

Applications and Practice in Neural Networks, Department of Artificial Intelligence, Korea University </br>
(고려대학교 인공지능학과 신경망응용및실습 프로젝트)

### Google Drive Link

<a href="https://drive.google.com/drive/folders/1S1xckDX1waQaXRlF7Ka20ZTUCO5TWlRT?usp=sharing">
  <img src="https://img.shields.io/badge/Google Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white"/>
</a>

### I. Project title
- Categorical Feature Encoding Challenge: https://www.kaggle.com/c/cat-in-the-dat/overview
### II. Project introduction
Is there a cat in your dat?

A common task in machine learning pipelines is encoding categorical variables for a given algorithm in a format that allows as much useful signal as possible to be captured.

Because this is such a common task and important skill to master, we've put together a dataset that contains only categorical features, and includes:

- binary features
- low- and high-cardinality nominal features
- low- and high-cardinality ordinal features
- (potentially) cyclical features

This contest is characterized by the fact that the data is artificially generated and the meaning of each feature and target value is unknown. Also, all the data is categorical, with binary features starting with bin_, nominal features starting with nom_, and ordered features starting with ord_. The target value is also categorical data and consists of two values, 0 and 1, so it can be considered a binary classification problem.

### III. Dataset description (need details)
In this competition, you will be predicting the probability [0, 1] of a binary target column.

The data contains binary features (bin_*), nominal features (nom_*), ordinal features (ord_*) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features ord_{3-5} are lexically ordered according to string.ascii_letters.

Since the purpose of this competition is to explore various encoding strategies, the data has been simplified in that (1) there are no missing values, and (2) the test set does not contain any unseen feature values. (Of course, in real-world settings both of these factors are often important to consider!)

#### Files
- train.csv - the training set
- test.csv - the test set; you must make predictions against this data

