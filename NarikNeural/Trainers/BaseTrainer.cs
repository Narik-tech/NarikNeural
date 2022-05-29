using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NarikNeural.Evaluators;

namespace NarikNeural.Trainers
{
    /// <summary>
    /// Abstract class all trainers should inherit from.
    /// </summary>
    public abstract class BaseTrainer
    {
        protected List<List<NeuralNode>> neuralLayers;
        protected Func<List<double>, List<double>> predictor;

        /// <summary>
        /// Makes layers and predictor availiable to the trainer.
        /// </summary>
        internal void SetVals(List<List<NeuralNode>> neuralLayers, Func<List<double>, List<double>> predictor)
        {
            this.predictor = predictor;
            this.neuralLayers = neuralLayers;
        }

        public abstract void Train(IEvaluator evaluator);
    }
}
