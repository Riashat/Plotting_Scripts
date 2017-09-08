import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from itertools import cycle

from numpy import genfromtxt
from scipy import stats

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ave_out", help="the output file")
parser.add_argument("sem_out", help="the output file")

args = parser.parse_args()

data = pd.read_csv(args.ave_out)
data_sem = pd.read_csv(args.sem_out)

data["TrueStdReturn"] = data_sem["TrueAverageReturn"]

data.to_csv(args.ave_out)
