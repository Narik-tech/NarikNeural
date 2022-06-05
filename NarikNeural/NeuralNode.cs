using System.Collections.Generic;

namespace NarikNeural
{
    public class NeuralNode
    {
        public int positionInLayer;
        //nodes from previous layer that have a weighted connection to this node
        public Dictionary<int, NeuralNode> previousConnectedNodes = new Dictionary<int, NeuralNode>();

        //weights for connection to upcoming layer
        public Dictionary<int, double> weights = new Dictionary<int, double>();

        //bias to be applied before reaching next layer
        public double bias;

        public NeuralNode (int position)
        {
            positionInLayer = position;
        }

        /// <summary>
        /// Sets weights and biases for this node.
        /// </summary>
        /// <param name="numOfWeights">Number of weights to be created.</param>
        /// <param name="center">If provided, will create number of weights centralized at this index.  Weights loop if exceeding bounds for nodes in next layer.</param>
        /// <param name="forceBias">Sets bias to this value.</param>
        /// <param name="forceWeights">Sets all weights to this value.</param>
        public void CreateWeightsBias(int numOfWeights, int nodesInNextLayer, int? center = null, double? forceWeights = null, double? forceBias = null)
        {
            if (numOfWeights > nodesInNextLayer)
                throw new Exception($"Number of weights to be created: {nodesInNextLayer} is greater than nodes in next layer: {nodesInNextLayer}");

            weights.Clear();

            //lower bound for weight creation
            int start = center == null ? 0 : (int)center - numOfWeights/2;

            //creates weights
            for (int i = start; i < numOfWeights; i++)
            {
                weights.Add(i % nodesInNextLayer, forceWeights != null ? (double)forceWeights : StaticMethods.GetRandomNumber());
            }
            //create bias
            bias = forceBias != null ? (double)forceBias : StaticMethods.GetRandomNumber();
        }

    }

    ///// <summary>
    ///// A node in the neural network.  Biases, then weights, are applied before reaching the next layer.
    ///// </summary>
    //public class NeuralNode
    //{
    //    public List<double> weights;
    //    public double bias;
        
    //    /// <summary>
    //    /// Constructor for nodes in the neural network.  Initial Weights and Biases are randomized to a value between -0.5 and 0.5.
    //    /// </summary>
    //    /// <param name="numOfWeights"></param>
    //    public NeuralNode(int numOfWeights)
    //    {
    //        weights = new List<double>(numOfWeights);
    //        for (int i = 0; i < numOfWeights; i++)
    //        {
    //            weights.Add(StaticMethods.GetRandomNumber());
    //        }
    //        bias = StaticMethods.GetRandomNumber();
    //    }
    //    public NeuralNode()
    //    {

    //    }
    //}
}
