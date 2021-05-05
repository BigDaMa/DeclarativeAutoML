
zeros = 212
ones = 357

n = zeros + ones

percentage = 0.6
zero_coefficient = (n / (2 * zeros)) *2 / n
one_coefficient = (n/ (2*ones)) *2 / n


print(zero_coefficient)
print(zero_coefficient * zeros)
print(one_coefficient * ones)
print(n)

def get_x(k, p):
    return p/k

print(get_x(zeros, 0.5))

