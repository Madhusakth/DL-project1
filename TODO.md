# DL-project1
1. Data Augmentation
Flipping and noise (randomly on 15% each) DONE - [data download link](https://drive.google.com/open?id=1BpSy8XRDbskvrBK9dFHw87ZoK-DRTt_a)

Also, mean image calculation done. Provided in data folder.

Just replace train_aug and yearbook_train_aug.txt in the starter code uploaded in github.
Also replace the valid dataset from the actual dataset. The starter code ahs just 1 example.

Later 
(* Maybe we can later do this on the fly using tensorflow for better performance once we get our best model *)
(* This is easier and more prevalent than storing all the data. We can directly implement them as a layer in tf [link](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced) *)

2. Preprocessing
Subtract mean of image
PCA (*later*)

3. Input Data (Can use the starter code as guidance)
Input the images both training and validation, labels, mean image, subtract mean image, shuffle, create batch size
http://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html

4. Network Model
VGG16
Change input layer size and channels (or duplicate the greyscale image accross all three channels)
Change final layer (number of classes)

5. Network Optimizer
SGD/Adam/Anything

6. Network Accuracy/Evaluation


### Extension

1. Benchmarking: Alexnet, Dropout, Batch Normalization, etc.
2. Losses: Classification (Cross Entropy) & Regression (L1, L2 etc.) anything else will work too.



