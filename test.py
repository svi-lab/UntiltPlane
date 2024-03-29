#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:14:51 2023

@author: dejan
"""
import os
import numpy as np
from UntiltPlane import UntiltPlane


folder = os.path.realpath("../../Data_other/Apoline_Miguel/")
filename = os.path.join(folder, 'export_data_texte1.txt')
# Load the data:
raw = np.loadtxt(filename, delimiter=',')

# %%
my_plane = UntiltPlane(raw)

# %%
######## Run the line below only afer you've finished the above ########
########          (shift + enter to run cell by cell)           ########
my_corrected_data = my_plane.corrected_data
