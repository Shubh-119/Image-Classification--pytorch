# Image-Classification--pytorch
This is an AI ML model which can predict animals based on input images
VLG Recruitment Challenge ‘25 Report:-
Shubh Khandelwal-24117133


INTRODUCTION
The objective of this VLG recruitment challenge was to classify animal species into 40 distinct categories. The dataset consisted of images from various classes, with significant diversity in quality, lighting, orientation, and labelling consistency.
For this task, I used ResNet50, a widely recognized architecture known for its residual learning framework that facilitates the training of deep networks. This model originally designed for ImageNet database proves to be really efficient in the task of Multi Label Image Classification.

Alongside ResNet50, I tried various different CNN architectures such as ResNet-101, inception-V4, VGG-16, ZFNet.. but ResNet50 proved to provide the best results.
Key Features of ResNet50:
Concept:

ResNet introduces the concept of residual blocks, where the output of a layer is added to the input of the same layer. This "shortcut connection" addresses the vanishing/exploding gradient problems, making it easier to train deeper networks.

Architecture:

ResNet50 has 50 layers consisting of convolutional layers, batch normalization layers, ReLU activations, and fully connected layers.
It includes 16 residual blocks, each with skip connections.
The architecture typically follows this pattern:
Conv1: 7x7 convolution, stride 2, followed by max pooling.
Conv2_x to Conv5_x: Multiple residual blocks with bottleneck architecture.
Bottleneck Design:
Each residual block consists of three convolutional layers:
A 1x1 convolution reduces the number of channels.
A 3x3 convolution performs the main feature extraction.
Another 1x1 convolution restores the original channel count.
Depth and Parameters:
ResNet50 has approximately 25.6 million parameters.
It is deeper than simpler architectures like AlexNet or VGG but remains efficient due to the residual connections.

Activation Function:

Uses ReLU (Rectified Linear Unit) for non-linearity after each convolutional layer.
Pooling:
Includes global average pooling before the final fully connected layer.
Output:
The final fully connected layer outputs predictions, typically with a softmax function for classification tasks.


Advantages:

Handles Vanishing Gradient Problem:
Residual connections allow gradients to flow through deeper layers, improving convergence.
Efficient Training:
The skip connections make it easier to train deeper networks without overfitting.
High Accuracy:
Achieves state-of-the-art performance on various benchmark datasets, such as ImageNet.
Versatility:
ResNet50 can be fine-tuned for a variety of tasks like object detection, segmentation, and feature extraction.
Use Cases:
Image classification (e.g., ImageNet).
Feature extraction in transfer learning.
Object detection and semantic segmentation (used in frameworks like Mask R-CNN).
Medical imaging, video analysis, and other specialized applications.


Dataset Analysis
1. Dataset Composition
Training Set: Contained images divided into 40 subfolders (classes), each representing an animal species. Each class had varied numbers of images, ranging from as low as 150 to as high as 250.
This was further split in a 15:85 ratio towards training and validation class. 
This made it easier to understand test results before submitting the finalised model.
Test Set: Consisted of 3,000 unlabeled images to evaluate the model's generalization.
2. Dataset Challenges
.Existence of 10 extra groups of animals which were not labelled in the dataset.
Class imbalance: classes containing difference in type and number of images can create a bias towards certain species.. Which could cause various redundancies.


Model Development:
PREPROCESSING:
To ensure uniformity and better analysis:
Images were resized to 224 x 224
Converted to tensors
Normalised to: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
Each training image was labelled with one of 40 provided labels:
Additionally 10 unlabelled animals were also present.. Which I neglected, due to technical complexity.




Training:
Loss Function: CrossEntropyLoss() function was used as our go to function due to its good relevance with classification tasks.




Optimiser:AdamW optimiser proved to provide best results in this project.. Which is an improved version of adam optimiser.




Epoch & Batch Size: After various different testings batch size:32 and epochs:10
                                   Were the best setting for better accuracy and efficiency.
Hardware: Kaggle website was used as a platform for creating a notebook of the above   model.. Which consisted of various fast options for GPU’s suitable for computation.



 How is it recognized?
Feature Hierarchy
Low-Level: Edges, textures.
Mid-Level: Body parts like wings and tails.
High-Level: Species-specific features.
Grad-CAM Analysis
Model focused on features like fur texture and facial patterns.
Misclassifications occurred in visually similar species.


Results and Observations:
After analysis of test set, our model performed the image classification of 40 species of animals with net accuracy of approximately 56%.
This metric could be improved by merging it with transformers or expanding the database.

This model, while generalising few similar appearing species as the same.. In general can provide accurate results towards two visually distinguishable species.






