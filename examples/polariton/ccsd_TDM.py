import sys

from sys import path as syspath
syspath.append('../../')

from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E0, E1, E2, braE1, commute
from wick.convenience import deP1, deE1, deE2, deEPS1, ketE1, ketE2, ketP1, ketP1E1

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

'''
bra = braE1("occ", "vir")
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final._print_einsum('T1'))

print('T1=',T1)
print('H1=',H1)

dip = one_e("d", ["occ", "vir"], norder=True)


RS = E1("RS", ["occ"], ["vir"])
RD = E2("RD", ["occ"], ["vir"])
R = RS + RD

# <HF| e^{-T} D e^{T} |HF>
HT = commute(dip, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
dbar = dip + HT + Fraction('1/2')*HTT

dbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT


S0 = dbar
out = apply_wick(S0)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print(final)
print(final._print_einsum())

'''

i = Idx(0, "occ")
a = Idx(0, "vir")
j = Idx(1, "occ")
b = Idx(1, "vir")

T1 = E1("T1", ["occ"], ["vir"])
T2 = E2("T2", ["occ"], ["vir"])
T = T1 + T2

L1 = E1("LS", ["vir"], ["occ"])
L2 = E2("LD", ["vir"], ["occ"])
L = L1 + L2


RS = E1("RS", ["occ"], ["vir"])
RD = E2("RD", ["occ"], ["vir"])
R = RS + RD


HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT

S0 = Hbar
out = apply_wick(S0)
out.resolve()
final = AExpression(Ex=out)
print("\n E = ")
print(final._print_einsum())


bra = E0('')
ket = E0('')
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("\n E = ")
print(final._print_einsum())

print('\n L1=', L1)

# vo block
operators = [FOperator(i, True), FOperator(a, False)]
pvo = Expression([Term(1, [], [Tensor([a, i], "")], operators, [])])

print('\npvo operator=', pvo)

PT = commute(pvo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvo + PT + Fraction('1/2')*PTT
mid += Fraction('1/6')*PTTT 
#mid += Fraction('1/24')*PTTTT

LS = deE1("LS", ["occ"], ["vir"])
LD = deE2("LD", ["occ"], ["vir"])

bra = E0('')
ket = E0('')

L = LS + LD

##full = (bra+L) * mid * R
#full = bra * L * mid * R * ket
full = (bra + L) * mid * (R + ket)

out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
print("\nP_{vo} = ")
print(final)
print("\nP_{vo} = ")
print(final._print_einsum())


# ov block
operators = [FOperator(a, True), FOperator(i, False)]
pvo = Expression([Term(1, [], [Tensor([i, a], "")], operators, [])])

print('\npvo operator=', pvo)

PT = commute(pvo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvo + PT + Fraction('1/2')*PTT
#mid += Fraction('1/6')*PTTT 
#mid += Fraction('1/24')*PTTTT

LS = deE1("LS", ["occ"], ["vir"])
LD = deE2("LD", ["occ"], ["vir"])
L = LS + LD

bra = E0('')
ket = E0('')

full = (bra+L) * mid #* R
#full = bra * L * mid * R * ket

out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
print("\nP_{ov} = ")
print(final)
print("\nP_{ov} = ")
print(final._print_einsum())


'''
# vv block
operators = [FOperator(a, True), FOperator(b, False)]
pvv = Expression([Term(1, [], [Tensor([b, a], "")], operators, [])])

PT = commute(pvv, T)
PTT = commute(PT, T)
mid = pvv + PT + Fraction('1/2')*PTT
full = L*mid*R

out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("\nP_{vv} = ")
#print(final)
print(final._print_einsum())


# oo block
operators = [FOperator(i, True), FOperator(j, False)]
poo = Expression([Term(1, [], [Tensor([j, i], "")], operators, [])])

PT = commute(poo, T)
PTT = commute(PT, T)
mid = poo + PT + Fraction('1/2')*PTT
full = L*mid*R

out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("\nP_{oo} = ")
#print(final)
print(final._print_einsum())
'''
