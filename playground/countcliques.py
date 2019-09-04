# countcliques.py

from deletion_utils.clique import *

if __name__ == "__main__":
    for n in range(4, 8):
        cliques = findclique(n)
        print(f'n = {n}')
        print(cliques)
