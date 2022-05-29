using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NarikNeural
{
    public static class StaticMethods
    {
        static Random random = new Random();
        /// <summary>
        /// Generates a random number between -0.5 and 0.5
        /// </summary>
        /// <returns></returns>
        public static double GetRandomNumber()
        {
            return random.NextDouble() - .5;
        }
    }
}
