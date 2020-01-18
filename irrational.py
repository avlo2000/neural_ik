def _calc(ind, base, prec):
    ind1 = ind % len(base)
    ind2 = (ind + 1) % len(base)
    if (prec == 0):
        return base[ind1] / base[ind2]
    return (base[ind1] + base[ind2] /  _calc(ind + 1, base, prec - 1))

# class Irrational:
#     def __init__(self, base, prec):
#         self.digits = []
#         self.base = base


irr = _calc(0, [10], 300)
print(irr)



