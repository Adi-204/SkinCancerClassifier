# Skin-Cancer-Classification-Using-CNN-Deep-Learning-Algorithm

# Theme
As skin cancer is one of the most frequent cancers globally, accurate, non-invasive dermoscopy-based diagnosis becomes essential and promising. A task of Easy Company’s Deep Learning CNN model is to predict seven disease classes with skin lesion images, including melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis / Bowens disease (intraepithelial carcinoma) (AKIEC), benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis) (BKL), dermatoﬁbroma (DF) and vascular lesion (VASC) as deﬁned by the International Dermatology Society.

# Dataset
The model is trained on the HAM10000 dataset, which contains dermoscopic lesion images. Each image is labeled based on histopathological or diagnostic criteria. The dataset includes 10,015 images across the seven classes, with distributions as follows:
AKIEC: 327 samples
BCC: 514 samples
BKL: 1,099 samples
DF: 115 samples
MEL: 1,113 samples
NV: 6,705 samples
VASC: 142 samples
The Dataset is too large for the Github Free LFS servers, hence, a link to the original KAGGLE dataset has been provided:
https://kaggle.com/kmader/skin-cancer-mnist-ham10000

# Types of Skin Cancer
Melanocytic Nevi (NV)
Melanoma (MEL)
Benign Keratosis-like Lesion (BKL)
Dermatofibroma (DF)
Basal Cell Carcinoma (BCC)
Actinic Keratoses (Akiec)
Vascular Lesions (Vasc)

# Architecture
The architecture consists of a CNN model built with the following components:

1 ResNet50 Model: Used as the base model with pre-trained weights.
2 Classifier Block 1:
    Convolutional layers (32 filters, 3x3, ReLU)
    MaxPooling layer (2x2)
    Dropout (0.15)
3 Classifier Block 2:
    Convolutional layers (64 filters, 3x3, ReLU)
    MaxPooling layer (2x2)
    Dropout (0.20)
4 Fully Connected Layers:
    Flatten layer
    Dense layer (128 neurons, ReLU)
    Dropout (0.5)
5 Output Dense layer (Softmax for classification across the seven disease classes)

This multi-stage approach allows the model to learn complex features effectively while minimizing overfitting through dropout regularization.

# Tech Stack and Dependencies
Python: Programming language for the entire pipeline.
TensorFlow/Keras: Used to build and train the CNN model.
OpenCV: For image preprocessing.
Pandas and NumPy: For data manipulation and handling.
Matplotlib: To visualize training results.
Scikit-learn: For metrics and additional machine learning utilities

# Conclusion
Skin cancer diagnosis is a time-sensitive and crucial task. Our deep learning model aims to assist in early diagnosis by accurately classifying skin lesion types, potentially expediting the diagnostic process for healthcare providers. Early detection and classification can save valuable time in treatment planning, providing patients with a better chance for recovery and improved quality of life. This tool can be instrumental for medical institutions aiming to enhance diagnostic efficiency, especially in areas with limited dermatology expertise.

# Contributing
We welcome contributions from the community to enhance SkinCancerClassifier and make it even more valuable for users. If you'd like to contribute, please follow these steps:

1 Fork the repository.
2 Create a new branch for your feature or bug fix.
3 Commit your changes and push the branch to your fork.
4 Submit a pull request with a detailed description of your changes.


# Contact
For questions, feedback, or support, please contact us at adiboghawala@gmail.com.

