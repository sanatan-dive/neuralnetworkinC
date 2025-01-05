# Neural Network for Cat vs Dog Classification

A simple feedforward neural network implemented in C for binary image classification between cats and dogs. This project uses the [stb_image](https://github.com/nothings/stb) library for image processing and includes custom implementations of neural network operations.

## Features

- Image preprocessing and resizing
- Single hidden-layer neural network architecture
- Sigmoid activation function
- Gradient-based learning with backpropagation
- Configurable hyperparameters (learning rate, hidden layer size, epochs)

## Prerequisites

- GCC or any compatible C compiler
- Make utility
- `stb_image` headers (included in repository):
  - `stb_image.h`
  - `stb_image_write.h`
  - `stb_image_resize.h`

## Getting Started

### Installation

```bash
git clone https://github.com/sanatan-dive/neuralnetworkinC
cd neuralnetworkinC
```

### Building and Running

Compile the project:
```bash
make
```

Run the program:
```bash
./main
```

## Dataset Structure

Place your dataset in the following directory structure:
```
data/
└── train/
    ├── cat.0.jpg
    ├── cat.1.jpg
    ├── ...
    ├── dog.0.jpg
    ├── dog.1.jpg
    └── ...
```

- Images should follow the naming convention: `label.index.jpg` (e.g., `cat.0.jpg`)
- All images will be automatically resized to 128x128 pixels and normalized during preprocessing

## How It Works

1. **Initialization**: Creates a neural network with randomized weights and biases
2. **Data Processing**:
   - Loads images from `data/train`
   - Performs preprocessing (resizing and normalization)
   - Splits data into training and test sets
3. **Training**:
   - Runs for the specified number of epochs
   - Uses backpropagation to update weights
   - Displays loss after each epoch
4. **Evaluation**: Tests model accuracy on the held-out test set

## Configuration

Modify these parameters in `main.c` to customize the network:

```c
#define IMAGE_WIDTH     128    // Input image width
#define IMAGE_HEIGHT    128    // Input image height
#define HIDDEN_SIZE     64     // Hidden layer neurons
#define TRAIN_COUNT     1000   // Training examples per class
#define EPOCHS          50     // Training epochs
#define LEARNING_RATE   0.01   // Gradient descent step size
```

## Output

The program provides:
- Training loss after each epoch
- Final test set accuracy