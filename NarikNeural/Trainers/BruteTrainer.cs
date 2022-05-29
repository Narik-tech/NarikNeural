using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NarikNeural.Evaluators;

namespace NarikNeural.Trainers
{
    /// <summary>
    /// Compares three values for each weight and bias.  
    /// The current value, and the current value +- a random value in the provided variance range.  
    /// Updates the network to the most accurate value by the evaluator.  Computationally expensive.
    /// </summary>
    public class BruteTrainer : BaseTrainer
    {
        
        private double variance;


        public BruteTrainer(double variance)
        {
            this.variance = variance;
        }
        public override void Train(IEvaluator evaluator)
        {
            for (int j = 0; j < neuralLayers.Count; j++)
            {
                for (int i = 0; i < neuralLayers[j].Count; i++)
                {
                    //bias training
                    var originalVal = neuralLayers[j][i].bias;
                    var randomNum = StaticMethods.GetRandomNumber() * variance * 2;
                    var newAcc = alterNodeBias(neuralLayers[j][i], evaluator, evaluator.Eval(predictor), originalVal + randomNum);
                    alterNodeBias(neuralLayers[j][i], evaluator, newAcc, originalVal - randomNum);

                    //weight training
                    for (int k = 0; k < neuralLayers[j][i].weights.Count; k++)
                    {
                        originalVal = neuralLayers[j][i].weights[k];
                        randomNum = StaticMethods.GetRandomNumber() * variance * 2;
                        newAcc = alterNodeWeight(neuralLayers[j][i], k, evaluator, evaluator.Eval(predictor), originalVal + randomNum);
                        alterNodeWeight(neuralLayers[j][i], k, evaluator, newAcc, originalVal - randomNum);
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private double alterNodeBias(NeuralNode node, IEvaluator evaluator, double previousAcc, double newVal)
        {
            var original = node.bias;
            node.bias = newVal;
            var newAcc = evaluator.Eval(predictor);

            if (newAcc > previousAcc)
            {
                return newAcc;
            }
            else
            {
                node.bias = original;
                return previousAcc;
            }
        }

        private double alterNodeWeight(NeuralNode node, int k, IEvaluator evaluator, double previousAcc, double newVal)
        {
            var original = node.bias;
            node.weights[k] = newVal;
            var newAcc = evaluator.Eval(predictor);

            if (newAcc > previousAcc)
            {
                return newAcc;
            }
            else
            {
                node.weights[k] = original;
                return previousAcc;
            }
        }
    }
}
