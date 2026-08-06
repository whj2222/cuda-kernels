# 各个 demo 共用的被测代码
# 故意写成有快有慢、有嵌套调用，这样 profile 结果才有东西可看

import time


def slow_sum(n):
    """纯 Python 循环求和：函数调用少，但自身耗时长 -> tottime 大"""
    total = 0
    for i in range(n):
        total += i
    return total


def add(a, b):
    """极小的函数：单次几乎不耗时，但被调用几百万次 -> ncalls 大"""
    return a + b


def call_heavy_sum(n):
    """把加法拆成函数调用：暴露 Python 函数调用开销"""
    total = 0
    for i in range(n):
        total = add(total, i)
    return total


def sleepy(seconds=0.05):
    """睡眠：墙钟时间长，但 CPU 不干活。用来说明 cProfile 计的是墙钟时间"""
    time.sleep(seconds)
    return seconds


def make_strings(n):
    """字符串拼接：内部大量调用 C 函数（str.join / list.append）"""
    parts = []
    for i in range(n):
        parts.append(str(i))
    return "".join(parts)


def top_level(n=200000):
    """一个有层次的入口函数，方便观察 cumtime 是怎么向上累加的"""
    a = slow_sum(n)
    b = call_heavy_sum(n)
    s = make_strings(n // 10)
    return a + b + len(s)
