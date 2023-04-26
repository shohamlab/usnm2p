# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-25 09:12:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-04 15:54:11

import numpy as np
import pandas as pd
from utils import pressure_to_intensity

MPa = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])  # MPa
I = pressure_to_intensity(MPa / 2 * 1e6) / 1e4  # W/cm2
df = pd.DataFrame({'P (MPa)': MPa, 'Isppa (W/cm2)': I})
print(df)