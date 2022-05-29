using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NarikNeural.Activators
{
    public interface IActivator
    { 
        /// <summary>
        /// Performs non linear conversions between layers of the network, makes deep learning possible.
        /// </summary>
        public double Activate(double input);
    }
}
