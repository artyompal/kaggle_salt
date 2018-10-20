import glob, os, sys
for f in glob.glob(os.path.join(sys.argv[1], "*.png")):
    print(f)
