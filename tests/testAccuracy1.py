import unittest
import patsy as pt
from pygam import LinearGAM, f, s, l
import numpy as np
import pandas as pd
import statsmodels as statsmodels
import pygam
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
import prophet
import statsmodels.tsa.holtwinters as hw

# Import your code from parent directory
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from assignment1 import model, modelFit, pred

# Run the checks

def checkNumbers(series):
    for i in series:
        if not isinstance(i, (float, int)):
            return False
    return True


class testCases(unittest.TestCase):
    def testValidPred(self):
        dataTest = pd.read_csv(currentdir + "/testData.csv")['trips']
        rmse = sum([(np.squeeze(pred)[i]-dataTest[i])**2 for i in range(len(np.squeeze(pred)))])
        rmse = np.sqrt(rmse)*1/744

        self.assertTrue(rmse<220, "Your forecasts have an RMSE above 220")
        