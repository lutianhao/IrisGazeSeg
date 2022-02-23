from lib.config import cfg, args
import numpy as np
import os


def run_test():
    from tools import test
    test.test()

if __name__ == '__main__':
    print(args.type)
    globals()['run_'+args.type]()
