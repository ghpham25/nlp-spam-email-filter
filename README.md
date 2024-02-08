# *Spam Email Filter*

## Features
- [x] **Implemented from scratch KNN Model**
- [x] **Implemented from scratch Naive Bayes Model**
- [x] **Used each model to classify spam emails from regular emails using Enron email dataset, compare performance**
- [x] **Implemented data preprocessing process from scrach**

## Dataset 
Enron email dataset 

## Results
### 1. KNN performance 
#### a. On spiral Dataset
Spiral Dataset Visualization: 

<img width="430" alt="Screenshot 2024-02-08 at 15 03 02" src="https://github.com/ghpham25/nlp-spam-email-filter/assets/99609320/74841eb7-b773-4636-8f70-9c08cd45f9e8">

Results of KNN on differnt k's: 

<img width="562" alt="Screenshot 2024-02-08 at 15 04 56" src="https://github.com/ghpham25/nlp-spam-email-filter/assets/99609320/0c038e8f-c30d-4bbc-a670-10791a3f546a">


#### b. On Email Dataset 
Accuracy: 0.498 
Confusion Matrix: 

<img width="177" alt="Screenshot 2024-02-08 at 15 06 31" src="https://github.com/ghpham25/nlp-spam-email-filter/assets/99609320/f0865dc9-dc45-45ad-8ea7-9c9deeefe64e">


### 2. Naive Bayes Peformace on email dataset
Accuracy: 0.90
Confusion Matrix: 

<img width="142" alt="Screenshot 2024-02-08 at 15 08 02" src="https://github.com/ghpham25/nlp-spam-email-filter/assets/99609320/b0a98d49-582a-4500-a207-6e1d787788f9">

## Analysis
KNN: 
False positive portion: 178/500 = 0.356

False negative portion: 85/500 = 0.17

-> Total false portion: 0.526

Naive Bayes: 

False positive portion: 552/6525 = 0.084

False negative portion: 150/6525 = 0.023

-> Total false portion: 0.107

The first thing we can see from the confusion matrices of 2 algorithms is that KNN has more false positive than false negatives, which is the same as Naive Bayes. From the calculations above, we also see that both the false positive and false negative portion of Naive Bayes is less than that of KNN, so NB performs a better job.



