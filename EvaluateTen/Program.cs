
﻿using NarikNeural;
using NarikNeural.Trainers;
using NarikNeural.Activators;
using EvaluateTen;

//creates 5 layer neural network with 3 nodes in each layer, with ReLU activator
var network = new NeuralNetwork(new List<int>()
{
    3,
    3,
    3,
    3,
    3,
},
new ReLU());

//example of how to import network XML
//var network = new NeuralNetwork(@"C:\Users\PC\Documents\TestFiles\testFile", new ReLU());

int roundsOfTraining = 200;
//trains neural network and writes accuracy for each round to console
//network is being trained to output 10
for (int i = 0; i < roundsOfTraining; i++)
{
    Console.WriteLine($"round {i} accuracy:" + network.Train(
        new BruteTrainer(.5),
        new TrainToTen(network.inputLayerSize)));
}

//example of how to write network XML
//network.WriteNetworkXML(@"C:\Users\PC\Documents\TestFiles", new TrainToTen(network.inputLayerSize));

//displays predicted outputs after some provided inputs
for (int i = -30; i < 30; i += 10)
{
    var list = new List<double>();
    for (int j = 0; j < network.inputLayerSize; j++)
        list.Add(i);

    var prediction = network.Predict(list);
    var result = $"Outputs for inputs of {i}:";
    for (int j = 0; j < prediction.Count; j++)
    {
        result += "  " + prediction[j];
    }
    Console.WriteLine(result);
}
