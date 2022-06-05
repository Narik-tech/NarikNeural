//#define IMPORT

using System.Diagnostics;
﻿using NarikNeural;
using NarikNeural.Trainers;
using NarikNeural.Activators;
using EvaluateTen;

#if !IMPORT
//creates 5 layer neural network with 3 nodes in each layer, with ReLU activator
var network = new NeuralNetwork(new List<int>()
{
    3,
    3,
    3,
},
new ReLU());
#endif



#if IMPORT
//example of how to import network XML
var network = new NeuralNetwork(@"C:\Users\PC\Documents\TestFiles\NeuralNetwork2022_06_04_37_a99.798.xml", new ReLU());
#endif

var evaluator = new TrainToTen(network.inputLayerSize);
Console.WriteLine($"accuracy: {evaluator.Eval(network.Predict)}");

var watch = Stopwatch.StartNew();

#if !IMPORT
int roundsOfTraining = 200;
//trains neural network and writes accuracy for each round to console
//network is being trained to output 10
double accuracy = -1;
double variance = 5;
for (int i = 0; i < roundsOfTraining; i++)
{
    var newAccuracy = network.Train(
       new BruteTrainer(variance),
       evaluator);

    if (newAccuracy == accuracy)
        variance *= .1;
    accuracy = newAccuracy;
    Console.WriteLine($"round {i} accuracy: {accuracy}");
}

//watch.Stop();

//Writes training time to console
TimeSpan ts = watch.Elapsed;
string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
    ts.Hours, ts.Minutes, ts.Seconds,
    ts.Milliseconds / 10);
Console.WriteLine($"Time for {roundsOfTraining} rounds of training: {elapsedTime}");
#endif

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
