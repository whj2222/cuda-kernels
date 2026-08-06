import cProfile
from workload import slow_sum, call_heavy_sum
def main():
    cProfile.run("slow_sum(2_000_000)")

if __name__ == '__main__':
    main()