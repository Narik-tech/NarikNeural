using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NarikNeural.Activators
{
    public class ReLU : IActivator
    {
        /// <summary>
        /// The ReLU activator converts all negative values to 0, and leaves positive values unchanged.
        /// </summary>
        public double Activate(double input)
        {
            return input < 0 ? 0 : input;
        }
    }
}
