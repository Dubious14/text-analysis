import numpy as np
import sys

# More tolerant file opening
fh = open(sys.argv[1], 'r', encoding='utf-8', errors='replace')
foutname = sys.argv[2]

# Read header line (num_words, vector_dim)
first = next(fh)
size = list(map(int, first.strip().split()))
wvecs = np.zeros((size[0], size[1]), float)

vocab = []
for i, line in enumerate(fh):
    line = line.strip().split()
    if len(line) != size[1] + 1:
        continue  # skip malformed lines
    vocab.append(line[0])
    wvecs[i, :] = np.array(list(map(float, line[1:])))

np.save(foutname + ".npy", wvecs)

with open(foutname + ".vocab", "w", encoding='utf-8') as outf:
    outf.write(" ".join(vocab) + "\n")

