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
        private XmlSerializer nodeSerializer = new XmlSerializer(typeof(NeuralXML));

        /// <summary>
        /// Constructor for the Neural Network. Each layer is created corresponding to provided number of nodes.  
        /// If weightLimit is provided, the number of weights for a node will not exceed this, creating partially connected layers.
        /// </summary>
        /// <param name="layerLengths">List of ints representing number of nodes in each layer.</param>/param>
        /// <param name="activator">Activation function for non-linear transformations between layers.<
        public NeuralNetwork(List<int> layerLengths, IActivator activator, int? weightLimit = null)
        {
            this.activator = activator;
            //runs for all but last layer length
            for(int layerNum = 0; layerNum + 1 < layerLengths.Count; layerNum++)
            {
                int nextLayer = layerLengths[layerNum + 1];
                List<NeuralNode> layer = new List<NeuralNode>(layerLengths[layerNum]);
                int numOfWeights = weightLimit == null ? nextLayer : new int[] { (int)weightLimit, nextLayer }.Min();
                for (int j = 0; j < layerLengths[layerNum]; j++)
                {
                    var node = new NeuralNode(j);
                    node.CreateWeightsBias(numOfWeights, nextLayer);
                    layer.Add(node);
                }
                neuralLayers.Add(layer);
            }
            //add last layer
            var tempLayer = new List<NeuralNode>();

            for (int i = 0; i < outputLayerSize; i++)
            {
                var node = new NeuralNode(i);
                node.CreateWeightsBias(0, 0);
                tempLayer.Add(node);
            }

            neuralLayers.Add(tempLayer);
            SetNodeDictionaries();
        }

        public NeuralNetwork(string inboundNetworkPath, IActivator activator)
        {
            CreateNetworkByPath(inboundNetworkPath);
            this.activator = activator;
        }

        //sets values in dictionaries referencing connected weighted nodes from previous layer
        public void SetNodeDictionaries()
        {
            if (neuralLayers.Count < 2)
                throw new Exception("Not enough layers to for connections.");
            for (int layer = 1; layer < neuralLayers.Count; layer++)
            {
                foreach(var node in neuralLayers[layer])
                {
                    node.previousConnectedNodes.Clear();
                }

                foreach(var node in neuralLayers[layer - 1])
                {
                    foreach(int weightKey in node.weights.Keys)
                    {
                        neuralLayers[layer][weightKey].previousConnectedNodes.Add(node.positionInLayer, node);
                    }
                }
            }
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

            ////add layer one biases to input
            for (int i = 0; i < inputLayerSize; i++)
            {
                previousLayer.Add(input[i] + neuralLayers[0][i].bias);
            }

            for (int layer = 1; layer < neuralLayers.Count; layer++)
            {
                for (int currentNode = 0; currentNode < neuralLayers[layer].Count; currentNode++)
                {
                    double weightedValue = 0;
                    //adds each weighted value from previous layer
                    foreach (KeyValuePair<int, NeuralNode> node in neuralLayers[layer][currentNode].previousConnectedNodes)
                    {
                        weightedValue += previousLayer[node.Key] * node.Value.weights[currentNode];
                    }
                    weightedValue += neuralLayers[layer][currentNode].bias;
                    currentLayer.Add(activator.Activate(weightedValue));
                }
                previousLayer = currentLayer.ToList();
                currentLayer = new List<double>();
            }
            return previousLayer;

            ////calculate all layers
            //for(int layer = 1; layer < neuralLayers.Count; layer++)
            //{
            //    //calculate weighted inputs from previous layer
            //    for (int currentNode = 0; currentNode < neuralLayers[layer].Count; currentNode++)
            //    {
            //        double weightedValue = 0;
            //        //adds each weighted value from previous layer
            //        for (int previousNode = 0; previousNode < previousLayer.Count; previousNode++)
            //        {
            //            weightedValue += previousLayer[previousNode] * neuralLayers[layer-1][previousNode].weights[currentNode];
            //        }
            //        weightedValue += neuralLayers[layer][currentNode].bias;
            //        currentLayer.Add(activator.Activate(weightedValue));
            //    }
            //    previousLayer = currentLayer.ToList();
            //    currentLayer = new List<double>();
            //}
            //return previousLayer;
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

            outputFileName += ".xml";

            var fullPath = Path.Join(outputPath, outputFileName);

            using (XmlWriter writer = XmlWriter.Create(fullPath))
            {
                nodeSerializer.Serialize(writer, (NeuralXML)neuralLayers); 
            }
        }

        public void CreateNetworkByPath(string inputPath)
        {
            NeuralXML? xmlLayers;
            using (XmlReader reader = XmlReader.Create(inputPath))
            {
                xmlLayers = nodeSerializer.Deserialize(reader) as NeuralXML;
                if (xmlLayers is null)
                    throw new Exception($"Neural layers at path {inputPath} could not be parsed.");
            }
            var parsedLayers = (List<List<NeuralNode>>)xmlLayers;
            neuralLayers = parsedLayers;
            SetNodeDictionaries();
        }
    }
}
