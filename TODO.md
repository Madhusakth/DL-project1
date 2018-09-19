# DL-project1
1. Data Augmentation
Flipping and noise (randomly on 15% each)
(* Maybe we can later do this on the fly for better performance once we get our best model *)

2. Preprocessing
Subtract mean of image
PCA (*later*)

3. Input Data (Can use the starter code as guidance)
Input the images both training and validation, labels, mean image, subtract mean image, shuffle, create batch size

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



