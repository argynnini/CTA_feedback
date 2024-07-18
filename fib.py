def fibonacci(n):
    fib_seq = [0, 1]
    for i in range(2, n):
        fib_seq.append(fib_seq[i-1] + fib_seq[i-2])
    return fib_seq

fib_numbers = fibonacci(100)
for num in fib_numbers:
    if num > 100:
        break
    print(num)