aa = 10
k=0
for i in range(aa):
    for j in range(aa):
        if  i>0 and j>0 and i<(aa-1) and j<(aa-1):
            k += 1
print(k)
