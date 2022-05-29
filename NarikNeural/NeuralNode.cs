using System.Collections.Generic;

namespace NarikNeural
{
    /// <summary>
    /// A node in the neural network.  Biases, then weights, are applied before reaching the next layer.
    /// </summary>
    public class NeuralNode
    {
        public List<double> weights;
        public double bias;
        
        /// <summary>
        /// Constructor for nodes in the neural network.  Initial Weights and Biases are randomized to a value between -0.5 and 0.5.
        /// </summary>
        /// <param name="numOfWeights"></param>
        public NeuralNode(int numOfWeights)
        {
            weights = new List<double>(numOfWeights);
            for (int i = 0; i < numOfWeights; i++)
            {
                weights.Add(StaticMethods.GetRandomNumber());
            }
            bias = StaticMethods.GetRandomNumber();
        }
    }
}
