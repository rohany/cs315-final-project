import csv
import os

def get_filename_with_prefix(folder, prefix):
    try:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        candidates = [f for f in files if f.startswith(prefix)]
        if len(candidates) == 0 or len(candidates) > 1:
          raise Exception("could not find file with prefix: " + prefix)
        return os.path.join(folder, candidates[0])
    except:
        return None

def parse_times(fname):
    f = open(fname, 'r')
    times = []
    for l in f:
      if "Elapsed time" in l or "ELAPSED TIME" in l:
        split = l.split(" ")
        if split[3] == "":
          times.append(float(split[4]))
        else:
          times.append(float(split[3]))
    # times.sort()
    # times = times[2:]
    # times = times[:-2]
    return times 
