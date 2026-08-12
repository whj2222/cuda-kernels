import cProfile
import pstats
import functools

from workload import slow_sum, call_heavy_sum, make_strings

_active = False


def profileit(sort='tottime', limit=8, dump=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _active
            if _active:
                return func(*args, **kwargs)

            pr = cProfile.Profile()
            _active = True

            try:
                pr.enable()
                return func(*args, **kwargs)
            finally:
                pr.disable()
                _active = False
                print(f"\n--- profile of {func.__qualname__} ---")
                pstats.Stats(pr).sort_stats(sort).print_stats(limit)
                if dump:
                    pr.dump_stats(dump)
                    print(f"（已落盘 {dump}）")

        return wrapper

    return decorator

@profileit(sort='tottime', limit=8, dump="03_decorator.prof")
def pipeline(n):
    a = slow_sum(n)
    b = call_heavy_sum(n)
    s = make_strings(n)
    return a + b + len(s)

@profileit()
def will_raise():
    """验证finally里边disable执行"""
    slow_sum(200000)
    raise RuntimeError("boom")

@profileit()
def outer():
    """验证_active保护生效"""
    return pipeline(300000)

def main():
    print("=" * 70)
    print("[1] 正常调用：返回值不受影响，profile 自动打印")
    print("=" * 70)
    print("pipeline 返回:", pipeline(500_000))

    print("=" * 70)
    print("[2] 装饰后 __name__ 仍然正确（functools.wraps 的功劳）")
    print("=" * 70)
    print("pipeline.__name__ :", pipeline.__name__)
    print("pipeline.__doc__  :", pipeline.__doc__)

    print("=" * 70)
    print("[3] 抛异常时：profile 照样打印，profiler 不会泄漏")
    print("=" * 70)
    try:
        will_raise()
    except RuntimeError as e:
        print("捕获异常：", e)

    print("=" * 70)
    print("[4] 嵌套 profile：内层自动跳过，不会 ValueError")
    print("=" * 70)
    print("outer 返回:", outer())


if __name__ == '__main__':
    main()

