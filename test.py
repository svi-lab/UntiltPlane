#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:14:51 2023

@author: dejan
"""
import os
import numpy as np
from UntiltPlane import PlaneUntilt


folder = os.path.realpath("../../Data_other/Apoline_Miguel/")
filename = os.path.join(folder, 'export_data_texte1.txt')
# Load the data:
raw = np.flipud(np.loadtxt(filename, delimiter=','))

my_plane = PlaneUntilt(raw)
