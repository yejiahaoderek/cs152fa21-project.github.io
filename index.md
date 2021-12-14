# Plant Recognition & Diagnosis App
A project for CS152

## Our Team
Magali Ngouabou, Eric Zhu, Camilla, Jiahao Ye

## Abstract
Plant species identification has been a growing field for the past decade and has been a subject of interest of various groups. While traditional methods relying on classification algorithms have different limitations, the introduction of convolutional neural networks has proven to be more effective. This paper explores the efficacy of plant species identification using five different neural network models (resnet18, resnet34, resnet50, resnet101, and AlexNet) with PlantNet dataset. The accuracy of each model is compared with benchmark from peer study and discussion on factors that might influenced the outcome is provided.

## Introduction
People without expertise in plant care always find it difficult to identify the plants they encounter. The need for plant recognition and plant pathology software arises from situations where people buy or receive new plants but do not know how to take care of them. 

We want to develop an app that tackles this problem. Using the extensive PlantNet database which includes 306,293 images and 1081 species, we will focus on training convolutional neural networks made using fastai to recognize plants by species. Inputs will be pre-processed to be 224x224 pixel three-channel images.

This paper is a study of various neural network efficacy in relation to the problem of plant species identification. We hope to find out which convolutional neural networks are best at Iefficiently and accurately classifying plants. Our benchmark accuracy is set in comparison to benchmark accuracies from a paper running PlantNet on various neural networks. In that paper, no neural network exceeded an accuracy of 80%, and thus we set that our benchmark accuracy. 

## Related Works

## 2.1 History of Plant Identification
Plant species identification has been a growing field for the past decade and has been a subject of interest for groups ranging from ecologists to hikers to architects. With the emergence of deep learning and image identification technology, the plant identification problem has seen significant exploration. As outlined in Waldchen et al., the plant identification problem is a supervised classification problem with a training phase of classified images and a testing phase and an application phase in which the classifier classifies new images and outputs the species. Photos of plants are generally so complex that previous implementations of plant recognition algorithms used feature vectors with models extracting features to create models that identified plants based on defining characteristics such as leaf shape<sup>7</sup>. The limitation to these classifiers is that they can only identify species that differ greatly in the feature the model is trained to differentiate e.g. leaves. Other approaches using "characteristic interest points" are able to be more general. According to Waldchen et al., the next step for plant identification is Convolutional neural networks which are optimal due to their increased processing power, and the ability to perform feature detection and extraction actions in a single step<sup>7</sup>.

The excitement over convolutional neural networks is also reflected in Mata-Montero and Carranza-Rojas (2016), who explored the difficulties of identifying the over 10 million plant species using computer vision and machine learning. They focus on methods that extract the following features: curvature, texture, venation, leaf morphometrics, or combinations of them. They conclude that convolutional neural networks would be most beneficial to tackling the complex issue of plant identification as it uses the whole or parts of the image directly in classifying rather than methods that more break the plant down into parts<sup>5</sup>.

## 2.2 Overview of Convolutional Neural Network (CNN)
Inspired by biological discovery (Matusugu 2003), the convolutional neural network (CNN) was invented by scientists as a supervised machine learning method. CNNs usually have many layers and are good at recognizing high resolution images<sup>6</sup>. The model first learns small abstract features on the image, lines or curves for example, and as the network gets to deeper layers, the features are added together, becoming recognizable<sup>3</sup>. In brief, CNNs learn and recognize different features through iteration and dimension reduction.

## 2.3 Existing Neural Nets Applications of the Plant Identification Problem
In the past couple of years, there has been significant headway in the plant identification problem with the use of neural networks, but the implementation has been met with mixed results primarily due to the difficulty of the problem, with data collection being a major roadblock. In Garcin et al., using various neural network models on Pl@ntNet resulted in no neural network achieving an accuracy of over 80%<sup>2</sup> while in Wulandhari et al., accuracies ranged wildly between 59.5% and 96%<sup>8</sup>. 

## Project Goals
- Find a dataset for training a NN for plant recognition
- Train the NN with high prediction accuracy
- Evaluate the performance of the NN
- Compare/evaluate the performance of different NN architecture



## Methods
To complete this project of creating a neural network that performs a plant species identification task, we used a convolutional neural network. The convolutional neural network was created using fastai and PyTorch based on existing convolutional neural network architectures such as ResNet and AlexNet. The data set we'll be using is the PlantNet data set which contains over 300,000 images of 1081 species. The dataset includes a csv file that pairs plant ID numbers with scientific names. The datasets are already split into training and validation sets with subfolders for photos corresponding to each species. To create the convolutional neural network with fastai, we first used the [ImageDataLoaders](https://docs.fast.ai/vision.data.html#ImageDataLoaders) class to create data loaders based on our training and validation data sets. In this step, we transformed the images to uniform 224x224 pixel three channel images. Then, we created a cnn_learner using our dataloaders and fastai's pre-set convolutional neural net models. We tested the PlantNet dataset on the models resnet18, resnet34, resnet50, resnet101, and alexnet. We used the Adam optimization function for our testing as well as a learning rate of 0.001. We ran each neural net through three epochs and recorded the accuracies and error rates. 

## Discussion 
For our results, we found varying accuracies based on each of the neural networks. For resnet18, we found that after 3 epochs, we had an accuracy of 76.1%, a training set loss of 0.85, and a validation set loss of 1.05. For resnet34, we saw an accuracy of 78.2%, a training set loss of 0.73, and a validation set loss of 0.96. For resnet50, we found an accuracy of 63.5%, a training set loss of 1.70, 1.67. For resnet101, we found an accuracy of 58.4% after three epochs, a training set loss of 2.04, and a validation set loss of 1.94. For AlexNet, we found an accuracy of 78.7%, a training set loss of 0.70, and a validation set loss of 0.94. 

| Model      | Accuracy         | Training Set Loss  | Validation Set Loss |
| ------------- |:-------------:| -----:|---:
| resnet18      | 76.1% | 0.85 | 1.05
| resnet34      | 78.2% | 0.73 | 0.96
| resnet50      | 63.5% | 1.70 | 1.67
| resnet101     | 58.4% | 2.04 | 1.94
| AlexNet       | 78.7% | 0.70 | 0.94

Images from the dataset:
![alt text](https://github.com/yejiahaoderek/cs152fa21-project.github.io/blob/gh-pages/IMG_3656.jpeg "A Batch of Plantnet Images")

Example of our results after running ResNet18:
![alt text](https://github.com/yejiahaoderek/cs152fa21-project.github.io/blob/gh-pages/Screen%20Shot%202021-12-13%20at%204.32.57%20PM.png "ResNet18 Sample Results")

It's interesting that similar to Garcin et al., none of our convolutional neural nets achieved an accuracy of over 80%. This is probably due to many of the factors discussed in Garcin et al. such as the uneven distribution of images in the dataset such that some plant species have many more than others. Interestingly enough, the most accurate neural networks were resnet18, resnet34, and AlexNet. The main difference between the resnets is the depth of the networks as indicated in their names. We see the best results with lower resnets, specifically resnet34 and resnet18, which could indicate a few things. It could be an issue of compatability with our specific dataset and the way that deeper resnet CNNs handle image convolution. We were also limited in how many times we could run the networks, so it's also possible that these specific resnet figures were a result of the specific times and conditions under which they were run rather than representative of their compatability with our dataset. Similarly with AlexNet, on one run it yielded an accuracy of 78.7% but on another it only got as high as 61.8%. This variation could be due to AlexNet's dropout layer and the batch of input units that were set to zero. This could be an issue especially with images that are less represented in the data set. Some plant species had as few as 2 images in the training set while others had as many as 7000. 

In trying to implement a streamlit version of our neural network, we ran into compatibility issues with conda so it wouldn't run the program. The structure was as follows: allow user to select upload method and choose between two of the neural networks that yielded the best results of the ones we tested out, cache the model after the first run and make inferences about the plant species based on the user's uploaded image. With more time, we could have tracked down the bug but as of now it's unclear how to solve the issue.

## Ethics
One of the more glaringly concerning applications of neural networks is in predicting criminal recidivism. The most prominent and most commonly used technology comes Northpointe, which several papers have demonstrated that the algorithm compounds the biases that exist in the criminal justice system. Northpointe uses a self-reporting test consisting 137 questions that inquire about everything from location to friendships in order to determine the likelihood of a someone committing a crime again. Dressel & Farid (2018) demonstrated that Northpointe's algorithm performed no better than a linear regression using only two of the 137 data points boasted by Northpointe's neural network. 

The criminal justice system is problematic enough without the added element of algorithmic bias. A false positive comprises not only that one person's future and wellbeing, but their family and everyone else connected to them. Some suggestions to using this technology have been limiting its scope to evaluating bail and bond amounts or in deciding whether or not to grant someone parole but even that scope is fraught. In Broward County, Florida, a county that spends $22,000 a year on Northpointe's neural network, there was a correlation between longer pretrial incarceration and higher Northpointe scores. Sade Jones, a young Black woman with no prior criminal record stole a bike and was rated medium risk by Northpointe's system. As a result of judges being swayed by the system, the young woman from a low income background was charged a $1000 bond, spend two nights in jail and would later struggle to find work.

There are implementations of neural networks that can scarcely be considered ethical to pursue, considering the consequences for people's lives and the fact that we have not developed robust systems to address our own bias. Technology like Northpointe are enabling broken systems like the criminal justice system to run more efficiently in their harmful operations.

## Reflection

If we were to do this again, we would look more into ways to build out a neural network beyond fastai in order to evaluate a wider range of convolutional neural networks ie use pytorch in order to have access to a wider range of CNNs like SqueezeNet, EffecientNet, etc. Also we had to reduce our dataset by about a third because some images didn't download properly, so potentially have safeguards for corrupted data next time. As we have learned, the dataset is a lot of the work in developing a good neural network. We would also explore the implementation of a streamlit application more in depth to create a user-friendly version of our neural network. In addition, adding to the usability of the neural network could come with other features such as planting advice for different plants and functionality to detet plant ailments.

To continue this work, we would try to have more control over the network itself, manipulate more of the hyperparameters under the fastai abstraction layer. Maybe also see what other plant datasets exist out there that are not just PlantNet. There are probably smaller ones that would work with smaller sets of plants, so it would be interesting to see how those results would compare to PlantNet. In addition, we would continue to work on creating a streamlit application to better present our results

## Presentation Video
https://youtu.be/nKOCknxnQYc

## References
> Barré, Pierre, et al. "LeafNet: A computer vision system for automatic plant species identification." *Ecological Informatics* 40 (2017): 50-56.

>Camille Garcin, Alexis Joly, Pierre Bonnet, Antoine Affouard, Jean-Christophe Lombardo, Mathias Chouet, Maximilien Servajean, Titouan Lorieul and Joseph Salmon, "Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution", in Proc. of Thirty-fifth Conference on Neural Information Processing Systems, Datasets and Benchmarks Track, 2021. 

> Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” Adv. Neural Inf. Process. Syst., pp. 1–9, 2012.

> Kumar, Neeraj, et al. "Leafsnap: A computer vision system for automatic plant species identification." *European conference on computer vision.* Springer, Berlin, Heidelberg, 2012.

> Mata-Montero, Erick, and Jose Carranza-Rojas. "Automated plant species identification: challenges and opportunities." *IFIP World Information Technology Forum*. Springer, Cham, 2016.

> Matusugu, Masakazu; Katsuhiko Mori; Yusuke Mitari; Yuji Kaneda (2003). "Subject independent facial expression recognition with robust face detection using a convolutional neural network". Neural Networks. 16 (5): 555–559. doi:10.1016/S0893-6080(03)00115-1. Retrieved 17 November 2013.

> Wäldchen, Jana et al. “Automated plant species identification-Trends and future directions.” *PLoS computational biology* vol. 14,4 e1005993. 5 Apr. 2018, doi:10.1371/journal.pcbi.1005993
 
> Wulandhari, Lili Ayu, et al. "Plant nutrient deficiency detection using deep convolutional neural network." 
*ICIC Express Letters* 13.10 (2019): 971-977.

