import cProfile

from workload import slow_sum, sleepy, make_strings

def main():
    print("=" * 70)
    print("[1] enable/disable：只测中间那段，前后的代码不计入")
    print("=" * 70)

          # 开关外面，不会出现在结果里

      # 只有这一句被统计

          # 开关外面，不会出现在结果里

    pr.print_stats(sort="cumulative")

    print("=" * 70)
    print("[2] 多段累加：第二次 enable 的数据会叠加到同一个 Profile 上")
    print("=" * 70)

      # 再测一遍同样的负载

    # make_strings 的 ncalls 从 1 变成了 2，tottime 也翻倍


    print("=" * 70)
    print("[3] with 语句（3.8+）：不会忘记 disable，异常时也能正确关闭")
    print("=" * 70)


    print("=" * 70)
    print("[4] runcall：传函数对象而不是字符串，而且能拿到返回值")
    print("=" * 70)
   # <- 返回值原样返回
    print(f"slow_sum 的返回值 = {result}")

    print("已写入 02_object.prof")

if __name__ == '__main__':
    main()