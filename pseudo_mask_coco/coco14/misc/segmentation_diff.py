import glob
import os
with open('../train2014.txt', 'r') as f:
    longlist = set([line.strip() for line in f])

# with open('../train2014_semseg.txt', 'r') as f:
#     shortlist = set([line.strip() for line in f])
shortlist = [os.path.basename(x).split(".")[0] for x in glob.glob("../../result/sem_seg/*.png")]

notdone = list(set(longlist - set(shortlist)))

with open('../train2014_notdone.txt', 'w') as fp:
    for item in notdone:
        # write each item on a new line
        fp.write("%s\n" % item)