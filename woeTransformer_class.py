"""
WOE-трансформер в виде класса с интерфейсом как в sklearn - с методами fit 
и predict
"""
import math
import time

import numpy as np
import pandas as pd
import seaborn as sns

class WoeTransformer:
    def __init__(self, min_sample_rate=0.05, min_count=3):
        self.min_sample_rate = min_sample_rate
        self.min_count = min_count
    

