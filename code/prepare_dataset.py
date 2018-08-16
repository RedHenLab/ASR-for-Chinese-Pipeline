from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import soundfile
import json
import argparse


DATA_HOME = os.path.expanduser('/mnt/rds/redhen/gallina/Singularity')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
	"--target_dir",
	default=DATA_HOME + '/Chinese_Pipeline/code',
	type=str,
	help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
	"--manifest_prefix",
	default="manifest",
	type=str,
	help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()

