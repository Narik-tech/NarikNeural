using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NarikNeural
{
    /// <summary>
    /// Network format that is directly serializable.  Only includes crucial information.
    /// </summary>
    public class NeuralXML
    {
        public List<List<NodeXML>> neuralLayers = new List<List<NodeXML>>();
        // conversion operator NeuralNetwork.Layers => NeuralXML
        public static explicit operator NeuralXML(List<List<NeuralNode>> layers)
        {
            var layersXML = new List<List<NodeXML>>();
            foreach (var layer in layers)
            {
                var layerXML = new List<NodeXML>();
                foreach (var node in layer)
                {
                    var nodeXML = new NodeXML() { bias = node.bias , positionInLayer = node.positionInLayer};

                    foreach (var weight in node.weights)
                    {
                        nodeXML.weights.Add(new WeightXML() { targetNode = weight.Key, weight = weight.Value });
                    }
                    layerXML.Add(nodeXML);
                }
                layersXML.Add(layerXML);
            }
            return new NeuralXML() { neuralLayers = layersXML };
        }
        // conversion operator NeuralXML => NeuralNetwork.Layers
        public static explicit operator List<List<NeuralNode>>(NeuralXML network)
        {
            var layers = new List<List<NeuralNode>>();
            foreach(var layer in network.neuralLayers)
            {
                var newLayer = new List<NeuralNode>();

                foreach(var node in layer)
                {
                    var newNode = new NeuralNode(node.positionInLayer) { bias = node.bias };
                    foreach(var weight in node.weights)
                    {
                        newNode.weights.Add(weight.targetNode, weight.weight);
                    }
                    newLayer.Add(newNode);
                }
                layers.Add(newLayer);
            }
            return layers;
        }
    }

    public class NodeXML
    {
        public int positionInLayer;
        public double bias;
        public List<WeightXML> weights = new List<WeightXML>();
    }

    public class WeightXML
    {
        public double weight;
        public int targetNode;
    }
}
