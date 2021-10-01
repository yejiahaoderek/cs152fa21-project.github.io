# Plant Recognition & Diagnosis App
A project for CS152

## Our Team
Magali Ngouabou, Eric Zhu, Camilla, Jiahao Ye

## Introduction
People without expertise in plant care always find it difficult to identify the plants they encounter. The need for plant recognition and plant pathology software arise from situations where people buy or receive new plants but do not know how to take care of them.

We want to develop an app that tackles this problem. We will focus on training neural networks to recognize plants as well as diagnose certain plant-related ailments. The project will be implemented to train the NN to identify the species, health conditions, and the app should offer some planting advice for the plant in the input picture.
The primary limitation of the project involves the two pronged nature of the app. Classifying plants by species and identifying plant ailments will require separate datasets and perhaps disease classification which discolors or damages distinctive plant features will interfere with species classification. 

Ideally, we would want to efficiently and accurately be able to classify plants and any diseases they may be suffering from. The benchmark accuracy isn’t something we know right now and hopefully as we progress with the project, we’ll get more clarity about our goals. 
We do not see any glaring ethical problems with this project but we will do some ethical exploration of other topics in AI, specifically, targeted advertisement.


## Project Goals
1. Create/find a dataset for training a NN for plant recognition
2. Train the NN with high prediction accuracy
3. Evaluate the performance of the NN
4. Deploy the NN on cellphones and create a simple app

Given that there might be similar works conducted for this topic, further steps could be
- Optimize the NN to produce higher accuracy
- Compare/evaluate the performance of differnent NN architecture
- Compress the NN so that it could run on cellphones
- Create a more user-friendly app that deploys the NN


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


> Wäldchen, Jana et al. “Automated plant species identification-Trends and future directions.” *PLoS computational biology* vol. 14,4 e1005993. 5 Apr. 2018, doi:10.1371/journal.pcbi.1005993


> Kumar, Neeraj, et al. "Leafsnap: A computer vision system for automatic plant species identification." *European conference on computer vision.* Springer, Berlin, Heidelberg, 2012.


> Mata-Montero, Erick, and Jose Carranza-Rojas. "Automated plant species identification: challenges and opportunities." *IFIP World Information Technology Forum*. Springer, Cham, 2016.


> Wulandhari, Lili Ayu, et al. "Plant nutrient deficiency detection using deep convolutional neural network." 
*ICIC Express Letters* 13.10 (2019): 971-977.


> Barré, Pierre, et al. "LeafNet: A computer vision system for automatic plant species identification." *Ecological Informatics* 40 (2017): 50-56.

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
