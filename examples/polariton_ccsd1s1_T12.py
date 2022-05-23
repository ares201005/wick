
import sys
from sys import path as syspath
syspath.append('../')

from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, one_p, two_p, ep11
from wick.convenience import P3, P2, P1, E1, E2, EPS1, braE1, braE2, braP1, braP1E1, commute

# f: fork matrix
# I: two body integral (ERIs)
# w: vibrational modes
# g: e-ph coupling matrix
# h: 
# G: single model peice of Hamiltonian
# 
H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("V", ["occ", "vir"], norder=True, compress=True)

# 1-boson operator
# two_p is: \sum_{01}w_{01}[b^{\dagger}_0(nm)b_1(nm)]
# one_p is: \sum_{0} G_{0} [b^{\dagger}_0(nm) + b_0(nm)]
#  Fock matrix and matrix of normal modes are not assumed to be diagonal, even though they will usually be diagonal
# i./e, w_{01} = 0, w_{00} is the vibrational mode (freq)

Hp = two_p("w") + one_p("G")
print('\ntwo_p(w)=', two_p("w"))
print('\none_p(G)=', one_p("G"))
print('\nHp=', Hp, '\n')

# e-p coupling, g^x_{qp}
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
H = H1 + H2 + Hp + Hep

print('H1', H1)
print('H2', H2)
print('Hep', Hep)
print('Hp', Hp)

#print('\nHamiltonain is\n', H)

# Fermionic excitations (single and double)
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])

# Bosonic excitation
#S1 = P1("s", ["nm"])
S1 = P1("s1", ["nm"])
S2 = P2("s2", ["nm"])
S3 = P3("s3", ["nm"])
print('S1=', S1,'\n')
print('S2=', S2,'\n')
print('S3=', S3,'\n')

# coupled fermion-boson excitation
U11 = EPS1("u", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + S2 + U11

print('\n ----------- T-1 term -----------\n')
bra = braE1("occ", "vir")
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print(final._print_einsum('T1'))

print('\n ----------- T-2 term -----------\n')
bra = braE2("occ", "vir", "occ", "vir")
#HTTT = commute(HTT, T)
#HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT #+ Fraction('1/24')*HTTTT

#S = bra*Hbar
#out = apply_wick(S)
#out.resolve()
#final = AExpression(Ex=out)
#print(final)

print('\n ----------- p1 term -----------\n')
bra = braP1('nm') 
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT

S = bra*Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print(final._print_einsum())

print('\n ----------- p1E1 term -----------\n')
bra = braP1E1('nm','occ', 'vir') 
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT

S = bra*Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print(final._print_einsum())

#======================================================================
#   EOM_CCSD for polariton
#======================================================================
from wick.convenience import one_e, two_e, E1, E2, braE1, E1, E2, commute
print(' \n EOM -CCSD equations\n')

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

bra = braE1("occ", "vir")
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])

R1 = E1("r", ["occ"], ["vir"])
R2 = E2("r", ["occ"], ["vir"])

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

#   EOM - sigS part
print('\nEOM-CCSD sigmaS part\n')

S = bra*(Hbar - E0)*R

out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print(final._print_einsum())

# =================  EOM - sigD part =================
print('\nEOM-CCSD sigmaD part\n')
bra = braE2("occ", "vir", "occ", "vir")

S = bra*(Hbar - E0)*R

out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print(final._print_einsum())



