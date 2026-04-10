"""
MooveFree Production Test Suite
================================
Tests every feature of the MooveFree system without requiring real hardware,
cameras, Firebase credentials, or GPIO pins.

Run:
    cd "C:\\BTech\\IPD\\Final code"
    python -m pytest tests/ -v --tb=short

Requirements:
    pip install pytest pytest-mock requests-mock numpy opencv-python flask
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import math
import time
import queue
import threading
import types
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import numpy as np
