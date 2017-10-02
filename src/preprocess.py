import pandas as pd
import numpy as np

train_text = pd.read_csv('./data/training_text', sep='\|\|', engine='python', names=['ID','Text'], skiprows=1)
test_text = pd.read_csv('./data/test_text', sep='\|\|', engine='python', names=['ID','Text'], skiprows=1)
train_var = pd.read_csv('./data/training_variants')
test_var = pd.read_csv('./data/test_variants')

# Data Check
train_var.info()
train_var.describe()
train_var.head()

test_var.info()
test_var.describe()

train_text.info()
train_text.describe()

test_text.info()
test_text.describe()

