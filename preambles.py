from __future__ import annotations
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import time
from datetime import datetime
import platform
import json
import torch
from torch import Tensor
import ast
from string import ascii_uppercase
from typing import List, Sequence, Tuple, Literal, Optional, Dict
from pathlib import Path
import argparse
from datasets import load_dataset, get_dataset_config_names, DatasetDict, concatenate_datasets
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from contextlib import contextmanager
from huggingface_hub import login, logout
import sys
import os
import yaml
import jsonlines


