# Neural_Networks_Adversarial_examples
This Jupyter notebook contains code to solve  different tasks related to building, training and creating adversarial examples for classification models on the MNIST and CIFAR10 datasets.

Specifically:

- Two Convolutional Neural Network (CNN) models to classify iamges in the MNIST and CIFAR10 datasets with test accuracy of $0.99\%$ and $0.80\%$ respectively.

- A generator class able to create fake images from noise for both datasets
![image](https://user-images.githubusercontent.com/68243875/228608984-52a08aa4-154b-488d-9fb3-cfce79cd2eb8.png)

- A composite model class that combines the generator and the dataset-specific frozen CNN classifier model. This composite model generates adversarial examples in the form of fake images that resemble the real training images but with added noise to mislead the model into misclassifying them as a different target class.
![image](https://user-images.githubusercontent.com/68243875/228609066-7230064b-8ffb-47b2-bbb4-21bba814e82b.png)
