# Adversarial Neural Network Examples

This work is one of a series of three distinct repositories that collectively constitute the coursework focused on Neural Networks for the Data Science & Information Technologies masters course: Μ124 - Machine Learning at the National and Kapodistrian University of Athens (NKUA) during the Fall 2022 semester. The two other repositories deal with:
- [Activation functions & gradient analysis](https://github.com/KyriakosPsa/ActFunc-GradientAnalysis)
- [Facial Expression Recognition (FER)](https://github.com/mdarm/machine-learning-coursework)

This repository contains code to solve tasks related to building, training, and creating adversarial examples for classification models on the MNIST and CIFAR10 datasets.

# Overview

The following were done

- Two Convolutional Neural Network (CNN) models to classify iamges in the MNIST and CIFAR10 datasets with test accuracy of $0.99\%$ and $0.80\%$ respectively.

- A generative network model able to create fake images from noise for both datasets
![image](https://user-images.githubusercontent.com/68243875/228608984-52a08aa4-154b-488d-9fb3-cfce79cd2eb8.png)

- A composite Generative Adversarial Network (GAN) that combines the generator and the dataset-specific frozen CNN classifier model. This composite model generates adversarial examples in the form of fake images that resemble the real training images but with added noise to mislead the model into misclassifying them as a different target class.
![image](https://user-images.githubusercontent.com/68243875/228609066-7230064b-8ffb-47b2-bbb4-21bba814e82b.png)

--- 
# Results 
The CNN classifiers performed adequately on both datasets: MNIST (top), CIFAR (bottom):

![image](https://github.com/user-attachments/assets/5571f19a-6cab-44c0-8e3a-534ed5c89862)![image](https://github.com/user-attachments/assets/c499c99c-f1de-472c-8d32-3b9d6257a5be)

When the GAN was used to "attack" the CNN classifier by generating images that a human would classify as the correct class but the model would misclassify, it proved to be highly successful. 

MNIST dataset GAN example:

![image](https://github.com/user-attachments/assets/9205cbbb-d703-403d-af53-c676f17034b9)


CIFAR10 dataset GAN example:

![image](https://github.com/user-attachments/assets/4524fae3-f9cd-4f94-8933-04495d7af212)


