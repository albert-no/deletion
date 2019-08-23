import sys

from subprocess import call


def main(argv):
    edge_num = argv[0]
    print(edge_num)

    commands = ['./bin/kernel-mis',
            f'--input-file=/home/noa/mycode/07_strash_mis/graphs/edge_n{edge_num}_metis.graph',
            '--experiment=simple-mcs',
            '--table',
            '>',
            f'edge_n{edge_num}_result.txt']
    print(commands)
    call(commands)


if __name__ == "__main__":
    main(sys.argv[1:])
