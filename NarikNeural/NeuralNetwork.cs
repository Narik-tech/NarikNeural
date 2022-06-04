using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using System.Xml.Serialization;
using NarikNeural.Evaluators;
using NarikNeural.Activators;
using NarikNeural.Trainers;

namespace NarikNeural
{
    public class NeuralNetwork
    {
        public List<List<NeuralNode>> neuralLayers = new List<List<NeuralNode>>();
        public int inputLayerSize { get { return neuralLayers.Count > 0 ? neuralLayers.First().Count : 0; } }
        public int outputLayerSize{ get { return neuralLayers.Count > 0 ? neuralLayers.Last().Count : 0; } }
        public IActivator activator;
        private XmlSerializer nodeSerializer = new XmlSerializer(typeof(List<List<NeuralNode>>));

        /// <summary>
        /// Constructor for the Neural Network. Each layer is created corresponding to provided number of nodes.  
        /// All nodes between layers are connected.
        /// </summary>
        /// <param name="layerLengths">List of ints representing number of nodes in each layer.</param>
        /// <param name="activator">Activation function for non-linear transformations between layers.</param>
        public NeuralNetwork(List<int> layerLengths, IActivator activator)
        {
            this.activator = activator;
            //runs for all but last layer length
            for(int i = 0; i + 1 < layerLengths.Count; i++)
            {
                List<NeuralNode> layer = new List<NeuralNode>(layerLengths[i]);

                for (int j = 0; j < layerLengths[i]; j++)
                    layer.Add(new NeuralNode(layerLengths[i + 1]));

                neuralLayers.Add(layer);
            }
            //add last layer
            var tempLayer = new List<NeuralNode>();

            for (int i = 0; i < outputLayerSize; i++)
                tempLayer.Add(new NeuralNode(0));

            neuralLayers.Add(tempLayer);
        }

        public NeuralNetwork(string inboundNetworkPath, IActivator activator)
        {
            CreateNetworkByPath(inboundNetworkPath);
            this.activator = activator;
        }

        /// <summary>
        /// Accepts a list of inputs, and produces a list of outputs as a prediction.  Utilizes all nodes, weights and biases.
        /// </summary>
        /// <returns>Predicted values.</returns>
        public List<double> Predict(List<double> input)
        {
            if (input.Count != inputLayerSize)
                throw new Exception($"Invalid input list count.  Expected: {inputLayerSize}, actual: {input.Count}");

            var previousLayer = new List<double>();
            var currentLayer = new List<double>();
            
            //add layer one biases to input
            for(int i = 0; i < inputLayerSize; i++)
            {
                previousLayer.Add(input[i] + neuralLayers[0][i].bias);
            }  
        
            //calculate all layers
            for(int layer = 1; layer < neuralLayers.Count; layer++)
            {
                //calculate weighted inputs from previous layer
                for (int currentNode = 0; currentNode < neuralLayers[layer].Count; currentNode++)
                {
                    double weightedValue = 0;
                    //adds each weighted value from previous layer
                    for (int previousNode = 0; previousNode < previousLayer.Count; previousNode++)
                    {
                        weightedValue += previousLayer[previousNode] * neuralLayers[layer-1][previousNode].weights[currentNode];
                    }
                    weightedValue += neuralLayers[layer][currentNode].bias;
                    currentLayer.Add(activator.Activate(weightedValue));
                }
                previousLayer = currentLayer.ToList();
                currentLayer = new List<double>();
            }
            return previousLayer;
        }

        /// <summary>
        /// Trains the network using the provided trainer and evaluator.  Returns evaluated accuracy after training.
        /// </summary>
        public double Train(BaseTrainer trainer, IEvaluator evaluator)
        {
            trainer.SetVals(neuralLayers, Predict);
            trainer.Train(evaluator);
            return evaluator.Eval(Predict);
        }

        public void WriteNetworkXML(string outputPath, IEvaluator? evaluator = null)
        {
            //generate timestamped filename
            var outputFileName = $"NeuralNetwork{DateTime.Now.ToString("yyyy_MM_dd_ss")}";
            //append accuracy if evaluator is provided
            if (evaluator != null)
                outputFileName += "_a" + evaluator.Eval(Predict).ToString("F3");

            var fullPath = Path.Join(outputPath, outputFileName);

            XmlSerializer serializer = new XmlSerializer(typeof(List<List<NeuralNode>>));

            using (XmlWriter writer = XmlWriter.Create(fullPath))
            {
                serializer.Serialize(writer, neuralLayers); 
            }
        }

        public void CreateNetworkByPath(string inputPath)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(List<List<NeuralNode>>));

            using (XmlReader reader = XmlReader.Create(inputPath))
            {
                var parsedLayers = serializer.Deserialize(reader) as List<List<NeuralNode>>;
                if (parsedLayers is null)
                    throw new Exception($"Neural layers at path {inputPath} could not be parsed.");
                neuralLayers = parsedLayers;
            }
        }
    }
}
