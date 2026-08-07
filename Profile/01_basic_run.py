import cProfile
from workload import slow_sum, call_heavy_sum
def main():
    print("slow_sum : ")
    cProfile.run("slow_sum(2_000_000)")

    print("call_heavy_sum : ")
    cProfile.run("call_heavy_sum(2_000_000)")

    n = 1000000
    cProfile.runctx("slow_sum(n)", globals(), locals(), sort="cumulative")

    cProfile.run("slow_sum(1000000)", filename="01_basic.prof")
if __name__ == '__main__':
    main()