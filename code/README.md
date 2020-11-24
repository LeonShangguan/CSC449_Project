# Actor-Action Classification 

## Task description
Given an image, the goal of the task is to predict its actor-action classes. Since an image may have different actors and actions, the task is a multi-label and multi-class problem.

## Dataset
A2D dataset: it has 43 valid actor-action labels such as 'cat-climbing', 'dog-running', 'baby-crawling'. It consists of 4750 training images, 1209 validation images, and 1044 testing images.  

## Data Processing
We provided a dataloader for processing training or validation sets of A2D dataset for the actor-action classification task. It will read images and annotations in training or validation sets; do processing on images and original annotaions; then return processed images (224x224x3) and its class labels (43-D encoding). For the returned labels, it has 44 dimensions corresponding to 44 different classes. If an encoded label is [1, 1, 0, ..., 0] (the first two elements are 1 and the others are 0), it means the image has the first and second classes ("adult-climbing" and "adult-crawling"). 
We provide another dataloader for processing testing set of A2D dataset, which will only return images without annotation labels. 

## Evaluation Metrics
We use precision, recall, and F1-score to measure performance of trained models. The descriptions about the three metrics can be found in course slides or https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9.

## Submission 
We will evaluate your model on the testing set and the results should be a (NXnum_cls) array containing predictions saved as "results.pkl", where N (1044) refers to testing set size and num_cls (43) is the number of classes, and the elements are 0 or 1. You may follow the test.py to do testing.
