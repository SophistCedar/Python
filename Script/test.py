import sys
for i in sys.argv:
    print(i)
print(type(sys.argv))# sys.argv is a command line arguments passed to a python script
print("len(sys.argv)",len(sys.argv))# sys.argv[0] contains the script file name