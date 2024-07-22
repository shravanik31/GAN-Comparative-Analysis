# Comparative Analysis of GAN Models: DCGAN, WGAN, and ACGAN

## Project Description
This project examines three significant Generative Adversarial Network (GAN) models: Deep Convolutional GAN (DCGAN), Wasserstein GAN (WGAN), and Auxiliary Classifier GAN (ACGAN). The analysis focuses on understanding their architectural nuances, performance metrics, and the quality of generated images using the CIFAR-10 dataset.

## Workflow Overview

### Data Collection and Preparation
- **Dataset**: CIFAR-10 dataset containing 60,000 32x32 colored images across 10 categories.
- **Preprocessing Steps**: Data is split into 50,000 training and 10,000 testing images.

### Model Architectures

#### DCGAN (Deep Convolutional GAN)
- **Generator**: Embeds textual input into a dense vector combined with a noise vector, projected and reshaped into a high-dimensional space. This feature map is progressively upsampled using transposed convolutions with batch normalization and ReLU activations.
- **Discriminator**: Processes the image and embedded textual information through a series of convolutional layers with LeakyReLU and batch normalization. Features are combined and undergo a final convolution to assess authenticity.
- **Hyperparameters**:
  - Epochs: 50
  - Learning Rate: 0.0002
  - Optimizer: Adam
  - Batch Size: 128
  - Noise Vector Dimension: 100
  - Text Embedding Dimension: 256
  - Loss Function: Binary Cross-Entropy

#### WGAN (Wasserstein GAN)
- **Generator**: Starts with text feature embedding into a vector, expanded and concatenated with a noise vector. It maps the noise-text vector to a higher dimension and reshapes it, increasing spatial size via transposed convolutions with batch normalization and ReLU activations.
- **Discriminator (Critic)**: Processes text input into a vector combined with flattened image features. The features are combined and reduced by a linear layer to output a realness score.
- **Hyperparameters**:
  - Learning Rate: 0.0002
  - Critic Iterations: 5 per generator iteration
  - Gradient Penalty Weight: 10
  - Noise Dimension: 100
  - Epochs: 50
  - Optimizer: Adam
  - Batch Size: 128
  - Loss Function: BCELoss (Binary Cross-Entropy Loss)

#### ACGAN (Auxiliary Classifier GAN)
- **Generator**: Embeds class labels into dense vectors combined with noise to guide image generation, using transposed convolutions for increasing resolution.
- **Discriminator**: Evaluates both the authenticity (real or fake) and class prediction of images through convolutional processing, with outputs via sigmoid and log-softmax functions.
- **Hyperparameters**:
  - Epochs: 50
  - Learning Rate: 0.0002
  - Optimizer: Adam
  - Batch Size: 128
  - Noise Dimension: 100
  - Number of Classes: 10
  - Feature Maps for Generator: 64
  - Feature Maps for Discriminator: 64
  - Loss Function: BCELoss and NLLLoss

### Training and Evaluation
- **Training**: Models trained on CIFAR-10 dataset with visualizations of loss and FID (Fréchet Inception Distance) scores over epochs.
- **Evaluation**: FID scores calculated using InceptionV3 model to compare statistical similarity between real and generated images.

## Results
- **DCGAN**: Exhibited the lowest and most stable FID scores across epochs, indicating high-quality image generation.
- **WGAN**: Showed fluctuating FID scores, suggesting less stable image quality.
- **ACGAN**: Initially outperformed WGAN but demonstrated increased variability in later epochs, possibly indicating overfitting.

## File Descriptions
- `DCGAN.py`: Implements DCGAN model with training and evaluation functions.
- `WGAN.py`: Implements WGAN model with training and evaluation functions.
- `ACGAN.py`: Implements ACGAN model with training and evaluation functions.
- `FIDComparisonAnalysis.py`: Compares FID scores of the three models using InceptionV3 model.

## Instructions to Execute Code

### 1) Training the Models
To train the models, run the respective scripts with appropriate parameters.

### 2) Comparative Analysis of the Models
To get the comparative analysis, run the FIDComparisonAnalysis script.

### Conclusion
This comparative analysis provides valuable insights into the effectiveness of different GAN architectures in generating high-quality images. It helps in understanding how different configurations and learning strategies can impact the performance and stability of GANs in real-world applications.

