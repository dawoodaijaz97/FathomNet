#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug entry point for FathomNet EDA
Used for easily launching the debugger on the EDA script
"""

import sys
import os

# Add the src directory to the path if not already there
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the main EDA function
from EDA.EDA import run_eda

if __name__ == "__main__":
    # Set a breakpoint here to start debugging
    # You can also set breakpoints in the imported modules
    print("Starting EDA debugging session...")
    
    # Run the EDA process
    results = run_eda()
    
    # You can inspect the results here when debugging
    print("EDA completed.") 