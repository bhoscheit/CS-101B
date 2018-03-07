def prep(string):
    for i in range(len(string)):
        if string[i] == ":":
            return string[i+2:]

n = 1000
name = str(n) + ".txt"

with open("simple.txt", 'r') as f:
    with open(name, 'w') as f2:
        for i in range(n):
            f2.write(prep(f.readline()))
    with open("test_"+name, 'w') as f3:
        for i in range(n):
            f3.write(prep(f.readline()))

