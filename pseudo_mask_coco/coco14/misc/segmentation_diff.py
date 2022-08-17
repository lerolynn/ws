import glob
import os
with open('../train2014.txt', 'r') as f:
    longlist = [line.strip() for line in f]
    print(len(longlist))

# with open('../train2014_semseg.txt', 'r') as f:
#     shortlist = set([line.strip() for line in f])
shortlist = [os.path.basename(x).split(".")[0] for x in glob.glob("../../result/sem_seg/*.png")]

with open('../train2014_semseg.txt', 'w') as f:
    print(len(shortlist))
    for item in shortlist:
        f.write("%s\n" % item)

notdone = list(set(longlist) - set(shortlist))

with open('../train2014_notdone.txt', 'w') as fp:
    print(len(notdone))
    for item in notdone:
        # write each item on a new line
        fp.write("%s\n" % item)