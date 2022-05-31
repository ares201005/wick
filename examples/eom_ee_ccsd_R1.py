
import sys
from sys import path as syspath
syspath.append('../')

from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, braE1, braEip1, Eip1, Eip2, commute

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

#bra = braEip1("occ")
bra = braE1("occ", "vir")
T1 = E1("T1", ["occ"], ["vir"])
T2 = E2("T2", ["occ"], ["vir"])

R1 = E1("RS", ["occ"], ["vir"])
R2 = E2("RD", ["occ"], ["vir"])

T = T1 + T2
R = R1 + R2
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT

S0 = Hbar
E0 = apply_wick(S0)
E0.resolve()

Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT
S = bra*(Hbar - E0)*R

out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print(final._print_einsum('SigmaS'))
