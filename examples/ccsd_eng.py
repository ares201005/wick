
from sys import path as syspath
syspath.append('../')

from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E0, E1, E2, braE1, commute

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("v", ["occ", "vir"], norder=True)
H = H1 + H2

bra = E0('')
ket = E0('')
T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
