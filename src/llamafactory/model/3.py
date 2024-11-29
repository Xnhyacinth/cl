

x = 625
y = 0.33
z = int(x * y)
for i in range(x):
    if (i + 1) % z == 0 and x - i >= z:
        print(i)
        