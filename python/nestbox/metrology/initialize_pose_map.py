"""
This script initializes a pose map for nestbox.

There are three main steps:

1. Resample the measured points to smooth out noise
2. Calculate the pose for each set of 20 points
3. Create and return a map object that maps commanded_position -> estimated_pose
"""

import os
import sys

