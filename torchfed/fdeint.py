



import os
import torch
import numpy as np
import random

from .utils_fde import _check_inputs, _check_inputs1, _check_inputs2
from .explicit_solver import Predictor, Predictor1, Predictor2, Predictor3, Predictor4, Predictor5, Predictor5_Modify, Predictor5_1, Predictor5_2, Predictor6,Predictor7, Predictor_Corrector, Predictor_Corrector1
from .implicit_solver import Implicit_l1, Implicit_l11
from .riemann_liouville_solver import GLmethod,GLmethod1,Product_Trap
SOLVERS = {"predictor":Predictor5,
          "corrector":Predictor_Corrector1,
           "implicitl1":Implicit_l11,
           "gl":GLmethod1,
           "trap":Product_Trap

}


def fdeint1(a1,b1,c1, d1, func, y0, learnable_t, t, fc_layer0, fc_layer1,step_size,method, options=None):


    a1, b1, c1, d1, func, y0, tspan, fc_layer0, fc_layer1, method, learnable_t = _check_inputs1(a1, b1, c1, d1, func, y0, t, fc_layer0, fc_layer1, step_size, method, learnable_t, SOLVERS)
    if options is None:
        options = {}
    solution = SOLVERS[method](a1=a1, b1=b1, c1=c1, d1=d1, func=func, y0=y0, learnable_t=learnable_t, tspan=tspan, fc_layer0=fc_layer0, fc_layer1=fc_layer1,  **options)

    return solution



