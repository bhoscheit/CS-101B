n = 100
name = str(n) + ".txt"

with open("simple.txt", 'r') as f:
    with open(name, 'w') as f2:
        for i in range(n):
            f2.write(f.readline())

