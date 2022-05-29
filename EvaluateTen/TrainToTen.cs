
﻿using NarikNeural.Evaluators;
using NarikNeural;

namespace EvaluateTen
{
    /// <summary>
    /// Eval passes 500 sets of random numbers into neural network predictor.  
    /// Returns an accuracy based on how close the outputs are to 10.
    /// </summary>
    public class TrainToTen : IEvaluator
    {
        int numOfInputs;
        public TrainToTen(int numOfInputs)
        {
            this.numOfInputs = numOfInputs;
        }

        public double Eval(Func<List<double>, List<double>> predictor)
        {
            var accuracyList = new List<double>();
            for (int j = 0; j < 500; j++)
            {
                var listOfInputs = new List<double>();

                for (int i = 0; i < numOfInputs; i++)
                {
                    listOfInputs.Add(StaticMethods.GetRandomNumber() * 100);
                }

                var predictedVals = predictor.Invoke(listOfInputs);


                foreach (var val in predictedVals)
                {
                    Math.Abs(10 - val);

                    accuracyList.Add((10 - Math.Abs(10 - val)) * 10);
                }
            }

            return accuracyList.Sum() / accuracyList.Count;
        }
    }
}
