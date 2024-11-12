# Skin-Cancer-Classification-Using-CNN-Deep-Learning-Algorithm

# Theme
As skin cancer is one of the most frequent cancers globally, accurate, non-invasive dermoscopy-based diagnosis becomes essential and promising. A task of Easy Company’s Deep Learning CNN model is to predict seven disease classes with skin lesion images, including melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC), actinic keratosis / Bowens disease (intraepithelial carcinoma) (AKIEC), benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis) (BKL), dermatoﬁbroma (DF) and vascular lesion (VASC) as deﬁned by the International Dermatology Society.

# Dataset
The model is trained on the HAM10000 dataset, which contains dermoscopic lesion images. Each image is labeled based on histopathological or diagnostic criteria. The dataset includes 10,015 images across the seven classes, with distributions as follows:
1. AKIEC: 327 samples
2. BCC: 514 samples
3. BKL: 1,099 samples
4. DF: 115 samples
5. MEL: 1,113 samples
6. NV: 6,705 samples
7. VASC: 142 samples
The Dataset is too large for the Github Free LFS servers, hence, a link to the original KAGGLE dataset has been provided:
https://kaggle.com/kmader/skin-cancer-mnist-ham10000

# Types of Skin Cancer
1. Melanocytic Nevi (NV)
2. Melanoma (MEL)
3. Benign Keratosis-like Lesion (BKL)
4. Dermatofibroma (DF)
5. Basal Cell Carcinoma (BCC)
6. Actinic Keratoses (Akiec)
7. Vascular Lesions (Vasc)

# Architecture
The architecture consists of a CNN model built with the following components:

1. ResNet50 Model: Used as the base model with pre-trained weights.
2. Classifier Block 1:
    Convolutional layers (32 filters, 3x3, ReLU)
    MaxPooling layer (2x2)
    Dropout (0.15)
3. Classifier Block 2:
    Convolutional layers (64 filters, 3x3, ReLU)
    MaxPooling layer (2x2)
    Dropout (0.20)
4. Fully Connected Layers:
    Flatten layer
    Dense layer (128 neurons, ReLU)
    Dropout (0.5)
5. Output Dense layer (Softmax for classification across the seven disease classes)

This multi-stage approach allows the model to learn complex features effectively while minimizing overfitting through dropout regularization.

# Tech Stack and Dependencies
1. Python: Programming language for the entire pipeline.
2. TensorFlow/Keras: Used to build and train the CNN model.
3. OpenCV: For image preprocessing.
4. Pandas and NumPy: For data manipulation and handling.
5. Matplotlib: To visualize training results.
6. Scikit-learn: For metrics and additional machine learning utilities

# Results
After training, the model achieved the following performance on the test set:

1. Accuracy: 78.98%
2. Precision: 77.82%
3. Recall: 78.98%
4. F1 Score: 77.47%

# Conclusion
Skin cancer diagnosis is a time-sensitive and crucial task. Our deep learning model aims to assist in early diagnosis by accurately classifying skin lesion types, potentially expediting the diagnostic process for healthcare providers. Early detection and classification can save valuable time in treatment planning, providing patients with a better chance for recovery and improved quality of life. This tool can be instrumental for medical institutions aiming to enhance diagnostic efficiency, especially in areas with limited dermatology expertise.

# Contributing
We welcome contributions from the community to enhance SkinCancerClassifier and make it even more valuable for users. If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch to your fork.
4. Submit a pull request with a detailed description of your changes.


# Contact
For questions, feedback, or support, please contact us at adiboghawala@gmail.com. 
