import json
import requests
import time
from game import *

def main():
    env = DraftEnv()
    env.reset()
    for i in range(45):
        mask = env.get_mask()
        for b in range(len(mask)):
            if mask[b] == True:
                action = b
        env.step(action)

if __name__ == "__main__":
    main()
