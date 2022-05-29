using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NarikNeural.Evaluators
{
    /// <summary>
    /// Evaluates the accuracy of a Neural Network predictor.  Used for training.
    /// </summary>
    public interface IEvaluator
    {
        /// <summary>
        /// This method passes in a list of inputs to the predictor, and evaluates the accuracy of the output.  Predictor may be called multiple times.
        /// </summary>
        /// <param name="predictor">Provided by the neural network, generates a prediction as a list of floats based on provided inputs.</param>
        /// <returns>float to represent the accuracy of the predictor</returns>
        public double Eval(Func<List<double>, List<double>> predictor);

    }
}
