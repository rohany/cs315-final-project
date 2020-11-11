from graph_utils import get_filename_with_prefix, parse_times

import os
import sys

# TODO (rohany): Turn this into using argparse.

mpidir = sys.argv[1]
regentdir = sys.argv[2]

def read_times(d):
  print(d, parse_times(get_filename_with_prefix(d, "output"))[0])

# read_times(os.path.join(regentdir, "1-core", "strong", "1-node"))
# read_times(os.path.join(regentdir, "1-core", "strong", "2-node"))
# read_times(os.path.join(regentdir, "1-core", "strong", "4-node"))
# read_times(os.path.join(regentdir, "8-cores", "strong", "1-node"))
# read_times(os.path.join(regentdir, "8-cores", "strong", "2-node"))
# read_times(os.path.join(regentdir, "8-cores", "strong", "4-node"))
# 
# read_times(os.path.join(mpidir, "strong", "serial"))
# read_times(os.path.join(mpidir, "strong", "FINE_GRAIN", "1-node"))
# read_times(os.path.join(mpidir, "strong", "FINE_GRAIN", "2-node"))
# read_times(os.path.join(mpidir, "strong", "FINE_GRAIN", "4-node"))
# read_times(os.path.join(mpidir, "strong", "NO_TALK", "1-node"))
# read_times(os.path.join(mpidir, "strong", "NO_TALK", "2-node"))
# read_times(os.path.join(mpidir, "strong", "NO_TALK", "4-node"))
# read_times(os.path.join(mpidir, "strong", "HIGH_WATER", "2-node"))
# read_times(os.path.join(mpidir, "strong", "HIGH_WATER", "4-node"))

read_times(os.path.join(mpidir, "serial"))
read_times(os.path.join(mpidir, "regent", "1-core"))
read_times(os.path.join(mpidir, "regent", "2-core"))
read_times(os.path.join(mpidir, "regent", "4-core"))
read_times(os.path.join(mpidir, "regent", "8-core"))
# read_times(os.path.join(mpidir, "regent", "1-gpu"))
read_times(os.path.join(mpidir, "regent", "2-gpu"))
read_times(os.path.join(mpidir, "regent", "4-gpu"))
