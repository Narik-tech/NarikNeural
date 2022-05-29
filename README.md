# NarikNeural

This is a framework for creating Neural Networks in C#.

It is intended to be compatible with any provided activation functions, training processes or evaluation metrics.

A neural network can be created with the NeuralNetwork class constructor.
A list of ints designating the number of nodes to be created in each layer must be provided as a parameter. An activation function must be included as a parameter as well, the ReLU function is included within the project.

The train method updates the network with a provided trainer and evaluator. The evaluator defines what is accurate for the network and is the base for how the neural network learns. The trainer defines the process for how the neural network learns.
The BruteTrainer trainer is provided, which is effective, but prohibitively slow for large neural networks.

The Predict method can be called directly to produce a list of doubles prediction with a list of doubles as input. This is also utilized by trainers.
