# Plant Recognition & Diagnosis App
A project for CS152

## Our Team
Magali Ngouabou, Eric Zhu, Camilla, Jiahao Ye

## Introduction
People without expertise in plant care always find it difficult to identify the plants they encounter. The need for plant recognition and plant pathology software arises from situations where people buy or receive new plants but do not know how to take care of them. 

We want to develop an app that tackles this problem. Using the extensive PlantNet database which includes 306,293 images and 1081 species, we will focus on training convolutional neural networks made using fastai to recognize plants by species. The project will be implemented to train the NN to identify the species of the plant while also offering some planting advice for the plant in the input picture. Inputs will be pre-processed to be 224x224 pixel three-channel images.

This paper is a study of various neural network efficacy in relation to the problem of plant species identification. We hope to find out which convolutional neural networks are best at Iefficiently and accurately classifying plants. Our benchmark accuracy is set in comparison to benchmark accuracies from a paper running PlantNet on various neural networks. In that paper, no neural network exceeded an accuracy of 80%, and thus we set that our benchmark accuracy. 

## Related Works

2.1 History of Plant Identification
Plant species identification has been a growing field for the past decade and has been a subject of interest for groups ranging from ecologists to hikers to architects. With the emergence of deep learning and image identification technology, the plant identification problem has seen significant exploration. As outlined in Waldchen et al., the plant identification problem is a supervised classification problem with a training phase of classified images and a testing phase and an application phase in which the classifier classifies new images and outputs the species. Photos of plants are generally so complex that previous implementations of plant recognition algorithms used feature vectors with models extracting features to create models that identified plants based on defining characteristics such as leaf shape (Waldchen et al.). The limitation to these classifiers is that they can only identify species that differ greatly in the feature the model is trained to differentiate e.g. leaves. Other approaches using "characteristic interest points" are able to be more general. According to Waldchen et al., the next step for plant identification is Convolutional neural networks which are optimal due to their increased processing power, and the ability to perform feature detection and extraction actions in a single step.

Mata-Montero and Carranza-Rojas (2016) explored the difficulties of identifying the over 10 million plant species using computer vision and machine learning. They focus on methods that extract the following features: curvature, texture, venation, leaf morphometrics, or combinations of them. They conclude that convolutional neural networks would be most beneficial to tackling the complex issue of plant identification as it uses the whole or parts of the image directly in classifying rather than methods that more break the plant down into parts.

2.2 Overview of Convolutional Neural Network (CNN)
Inspired by biological discovery (Matusugu 2003), the convolutional neural network (CNN) was invented by scientists as a supervised machine learning method. CNNs usually have many layers and are good at recognizing high resolution images. The model first learns small abstract features on the image, lines or curves for example, and as the network gets to deeper layers, the features are added together, becoming recognizable (Krizhevsky 2012). In brief, CNNs learn and recognize different features through iteration and dimension reduction.

## Project Goals
1. Create/find a dataset for training a NN for plant recognition
2. Train the NN with high prediction accuracy
3. Evaluate the performance of the NN

Given that there might be similar works conducted for this topic, further steps could be
- Optimize the NN to produce higher accuracy
- Compare/evaluate the performance of differnent NN architecture
- Compress the NN so that it could run on cellphones


## Literature Review
- **Automated plant species identification—Trends and future directions (Waldchen et al.)** [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5886388/)
  - The authors Waldchen, Rzanny, Seeland, and Mader write a survey paper on the current implementation of plant species identification and the way advances in deep learning are changing the way we identify plants. Waldchen et al. describe plant identification as a supervised classification problem. Many current implementations use feature identification by extracting features from plant images, creating a feature vector. Within these were model-specific implementations and model-free implementations. Of these, model-specific ended up being feature specific (e.g. a classifier that only worked when identifying very leafy plants) whereas model-free implementations were able to be more generalizable. When this paper was written in 2018, Waldchen et al. looked forward to using Convolutional Neural Networks because CNNs do not require custom, hand-crafted feature detection and extraction steps with both being part of the training process.


- **A computer vision system for automatic plant species identification** [Link](https://link.springer.com/chapter/10.1007/978-3-642-33709-3_36)
  - The paper presents an approach that does not use neural networks to identify plant species. This paper introduces a sequence of algorithmic procedures to identify tree species by leaf image extraction, stem removal, and nearest neighbors clustering.


- **Automated Plant Species Identification: Challenges and Opportunities** [Link](https://link.springer.com/chapter/10.1007/978-3-319-44447-5_3)
  - This paper discusses different ways of identifying plant species so far using computer vision and machine learning techniques, by stages. Common workflow is presented including data acquisition and sources, leaf segmentation, and feature extraction and identification. Deep learning is mentioned as approaches with huge success and an example about Convolutional Neural Network (CNN) is given.


- **PLANT NUTRIENT DEFICIENCY DETECTION USING DEEP CONVOLUTIONAL NEURAL NETWORK** [Link](http://www.icicel.org/ell/contents/2019/10/el-13-10-13.pdf)
  - The paper implements convolutional neural network to detect the health condition of a controlled species of plant. Particularly, the input pictures are kora and the health conditions are labelled manually in binary: either healthy or difficiency. The accuracy varies from 59.5% to 96% and analysis is given.


- **LeafNet: A computer vision system for automatic plant species identification** [Link](https://www.sciencedirect.com/science/article/abs/pii/S1574954116302515)
  - **No Access**


- **TA-CNN: Two-way attention models in deep convolutional neural network for plant recognition** [Link](https://www.sciencedirect.com/science/article/abs/pii/S0925231219309440)
  - **No Access**


## Data and Method Overview
- Data
  - Data set: https://gitlab.inria.fr/cgarcin/plantnet_dataset
- Neural Network we will use: a CNN built using Tensorflow (link: https://www.tensorflow.org/tutorials/images/cnn)
- Input: We will pre-process all the images to form uniform input files (64x64, three-channel images)
- Output: One neural network will output a binary classification of whether the plant is healthy or unhealthy. The other network will be more complicated, outputting values for a set of plants, inferring which plant the image most matches. 

## Project Milestone 2
- What have you completed or tried to complete?
  - Through this portion of the milestone, we have started to look into how we would create the convolutional neural network and looking into the dataset. The dataset has been downloaded from the gitlab link above and we have started research on how to use pytorch for a convolutional data set. In response to some of the Hypothesis comments, we have decided to switch from Tensorflow to Pytorch because it is the software we've been using in class and we're all more familiar with it.
- What issues have you encountered?
  - The main issue of the project has been fairly expected. As mentioned in the Hypothesis comments, we haven't really been able to find a data set for plant related ailments and that information doesn't seem to be included in the PlantNet data set we set above. We'll need to make a decision on how we want to proceed with this portion of the project soon but for the mean time are focusing on species identification. 

## Methods
To complete this project of creating a neural network that performs a plant species identification task, we used a convolutional neural network. The convolutional neural network was created using fastai and PyTorch based on existing convolutional neural network architectures such as ResNet and AlexNet. The data set we'll be using is the PlantNet data set which contains over 300,000 images of 1081 species. The dataset includes a csv file that pairs plant ID numbers with scientific names. The datasets are already split into training and validation sets with subfolders for photos corresponding to each species. To create the convolutional neural network with fastai, we first used the ImageDataLoaders class to create data loaders based on our training and validation data sets. In this step, we transformed the images to uniform 224x224 pixel three channel images. Then, we created a cnn_learner using our dataloaders and fastai's pre-set convolutional neural net models. We tested the PlantNet dataset on the models resnet18, resnet34, resnet50, resnet101, and alexnet. We used the Adam optimization function for our testing as well as a learning rate of 0.001. We ran each neural net through three epochs and recorded the accuracies and error rates. 

## Discussion 
For our results, we found varying accuracies based on each of the neural networks. For resnet18, we found that after 3 epochs, we had an accuracy of 76.1%, a training set loss of 0.85, and a validation set loss of 1.05. For resnet34, we saw an accuracy of 78.2%, a training set loss of 0.73, and a validation set loss of 0.96. For resnet50, we found an accuracy of 63.5%, a training set loss of 1.70, 1.67. For resnet101, we found an accuracy of 58.4% after three epochs, a training set loss of 2.04, and a validation set loss of 1.94. For AlexNet, we found an accuracy of 78.7%, a training set loss of 0.70, and a validation set loss of 0.94. 

It's interesting that similar to Garcin et al., none of our convolutional neural nets achieved an accuracy of over 80%. This is probably due to many of the factors discussed in Garcin et al. such as the uneven distribution of images in the dataset such that some plant species have many more than others. Interestingly enough, the most accurate neural networks were resnet18, resnet34, and AlexNet. Perhaps this is because the neural networks with more layers (resnet50 and resnet101), are overfitting the data, leading to worse accuracy on the validation set. This overfitting of those neural nets can also be seen in the losses. For the deeper neural networks, we find a validation set loss that's less than the training set loss, indicative of overfitting. Perhaps this tendency towards overfitting can be attributed to the uneven distribution of images where the neural net is over-inclined to categorize images as one of the species that's overwhelmingly represented.

## Ethics
One of the more glaringly concerning applications of neural networks is in predicting criminal recidivism. The most prominent and most commonly used technology comes Northpointe, which several papers have demonstrated that the algorithm compounds the biases that exist in the criminal justice system. Northpointe uses a self-reporting test consisting 137 questions that inquire about everything from location to friendships in order to determine the likelihood of a someone committing a crime again. Dressel & Farid (2018) demonstrated that Northpointe's algorithm performed no better than a linear regression using only two of the 137 data points boasted by Northpointe's neural network. 

The criminal justice system is problematic enough without the added element of algorithmic bias. A false positive comprises not only that one person's future and wellbeing, but their family and everyone else connected to them. Some suggestions to using this technology have been limiting its scope to evaluating bail and bond amounts or in deciding whether or not to grant someone parole but even that scope is fraught. In Broward County, Florida, a county that spends $22,000 a year on Northpointe's neural network, there was a correlation between longer pretrial incarceration and higher Northpointe scores. Sade Jones, a young Black woman with no prior criminal record stole a bike and was rated medium risk by Northpointe's system. As a result of judges being swayed by the system, the young woman from a low income background was charged a $1000 bond, spend two nights in jail and would later struggle to find work.

There are implementations of neural networks that can scarcely be considered ethical to pursue, considering the consequences for people's lives and the fact that we have not developed robust systems to address our own bias. Technology like Northpointe are enabling broken systems like the criminal justice system to run more efficiently in their harmful operations.

## Reflection

- What would you do differently next time?
- How would you continue this work (e.g., what extensions would you pursue)?
- 
If we were to do this again, we would look more into ways to build out a neural network beyond fastai in order to evaluate a wider range of convolutional neural networks. Also we had to reduce our dataset by about a third because some images didn't download properly, so potentially have safeguards for corrupted data next time. As we have learned, the dataset is a lot of the work in developing a good neural network. 

To continue this work, we would try to have more control over the network itself, manipulate more of the hyperparameters under the fastai abstraction layer. Maybe also see what other plant datasets exist out there that are not just PlantNet. There are probably smaller ones that would work with smaller sets of plants, so it would be interesting to see how those results would compare to PlantNet. 

## References

> Wäldchen, Jana et al. “Automated plant species identification-Trends and future directions.” *PLoS computational biology* vol. 14,4 e1005993. 5 Apr. 2018, doi:10.1371/journal.pcbi.1005993

>"Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution", Camille Garcin, Alexis Joly, Pierre Bonnet, Antoine Affouard, Jean-Christophe Lombardo, Mathias Chouet, Maximilien Servajean, Titouan Lorieul and Joseph Salmon, in Proc. of Thirty-fifth Conference on Neural Information Processing Systems, Datasets and Benchmarks Track, 2021. 

> Kumar, Neeraj, et al. "Leafsnap: A computer vision system for automatic plant species identification." *European conference on computer vision.* Springer, Berlin, Heidelberg, 2012.


> Mata-Montero, Erick, and Jose Carranza-Rojas. "Automated plant species identification: challenges and opportunities." *IFIP World Information Technology Forum*. Springer, Cham, 2016.


> Wulandhari, Lili Ayu, et al. "Plant nutrient deficiency detection using deep convolutional neural network." 
*ICIC Express Letters* 13.10 (2019): 971-977.


> Barré, Pierre, et al. "LeafNet: A computer vision system for automatic plant species identification." *Ecological Informatics* 40 (2017): 50-56.

> Matusugu, Masakazu; Katsuhiko Mori; Yusuke Mitari; Yuji Kaneda (2003). "Subject independent facial expression recognition with robust face detection using a convolutional neural network". Neural Networks. 16 (5): 555–559. doi:10.1016/S0893-6080(03)00115-1. Retrieved 17 November 2013.

> Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” Adv. Neural Inf. Process. Syst., pp. 1–9, 2012.



<!-- - Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src) -->

<!-- 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/yejiahaoderek/cs152sp21-project.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out. -->
