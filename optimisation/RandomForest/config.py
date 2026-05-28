# Load Common Helpers
import sys
sys.path.append('../../helper_functions')
from amex_metric import amex_metric_numpy
from importer import import_dataset

# Load Configuration file
from configparser import ConfigParser
config = ConfigParser()
config.read("config.ini")