# Neural Network for Cat vs Dog Classification  

This project implements a simple feedforward neural network for binary image classification (cats vs dogs) using C. It uses the [stb_image](https://github.com/nothings/stb) library for image processing and includes custom implementations for neural network forward propagation, backpropagation, and training.

---

## Features  

- Resizes and preprocesses images for neural network input.  
- Implements a single hidden-layer neural network.  
- Uses the sigmoid activation function for non-linearity.  
- Supports gradient-based learning with backpropagation.  
- Customizable hyperparameters: learning rate, hidden layer size, epochs, etc.  

---

## Prerequisites  

Before running the program, ensure the following tools are installed:  

- GCC or any compatible C compiler.  
- Make utility.  
- `stb_image.h`, `stb_image_write.h`, and `stb_image_resize.h` (already included in the repository).  

---

## Getting Started  

### Clone the Repository  

```bash  
git clone <repository-url>  
cd <repository-folder>  
Here’s a README.md file for your project:

# Neural Network for Cat vs Dog Classification  

This project implements a simple feedforward neural network for binary image classification (cats vs dogs) using C. It uses the [stb_image](https://github.com/nothings/stb) library for image processing and includes custom implementations for neural network forward propagation, backpropagation, and training.

---

## Features  

- Resizes and preprocesses images for neural network input.  
- Implements a single hidden-layer neural network.  
- Uses the sigmoid activation function for non-linearity.  
- Supports gradient-based learning with backpropagation.  
- Customizable hyperparameters: learning rate, hidden layer size, epochs, etc.  

---

## Prerequisites  

Before running the program, ensure the following tools are installed:  

- GCC or any compatible C compiler.  
- Make utility.  
- `stb_image.h`, `stb_image_write.h`, and `stb_image_resize.h` (already included in the repository).  

---

## Getting Started  

### Clone the Repository  

```bash  
git clone https://github.com/sanatan-dive/neuralnetworkinC 
cd deepLearningInC
```
Build and Run

Compile the Program:
Use make to compile the code.
```
make
```
  

Run the Program:
After compiling, execute the program.
```
./main  
```

Dataset

The program expects a dataset of cat and dog images in the following structure:

data/  
└── train/  
    ├── cat.0.jpg  
    ├── cat.1.jpg  
    ├── ...  
    ├── dog.0.jpg  
    ├── dog.1.jpg  
    └── ...  

    Images should be named in the format label.index.jpg (e.g., cat.0.jpg).
    The images will be resized to 128x128 and normalized during preprocessing.

How It Works

    Initialization:
        The program initializes a neural network with random weights and biases.

    Dataset Loading:
        Loads and preprocesses images from the data/train directory.
        Divides the dataset into training and testing sets.

    Training:
        Trains the neural network for a specified number of epochs using backpropagation.
        Displays the loss after each epoch.

    Testing:
        Evaluates the model's accuracy on the test dataset.

Output

    Loss values after each epoch.
    Final accuracy on the test dataset.

Customization

You can adjust the following hyperparameters by modifying the #define macros at the top of main.c:

    IMAGE_WIDTH and IMAGE_HEIGHT: Input image size.
    HIDDEN_SIZE: Number of neurons in the hidden layer.
    TRAIN_COUNT: Number of training examples per class.
    EPOCHS: Number of training epochs.
    LEARNING_RATE: Step size for gradient descent.

License

This project is licensed under the MIT License. See the LICENSE file for details.
