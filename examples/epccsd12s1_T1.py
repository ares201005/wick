
from sys import path as syspath
syspath.append('../')

from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, one_p, two_p, ep11
from wick.convenience import P3, P2, P1, E1, E2, EPS1, braE1, commute

# f: fork matrix
# I: two body integral (ERIs)
# w: vibrational modes
# g: e-ph coupling matrix
# h: 
# G: single model peice of Hamiltonian
# 
H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)

# 1-boson operator
# two_p is: \sum_{01}w_{01}[b^{\dagger}_0(nm)b_1(nm)]
# one_p is: \sum_{0} G_{0} [b^{\dagger}_0(nm) + b_0(nm)]
#  Fock matrix and matrix of normal modes are not assumed to be diagonal, even though they will usually be diagonal
# i./e, w_{01} = 0, w_{00} is the vibrational mode (freq)

Hp = two_p("w") #+ one_p("G")
print('\ntwo_p(w)=', two_p("w"))
print('\none_p(G)=', one_p("G"))
print('\nHp=', Hp, '\n')

# e-p coupling, g^x_{qp}
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
H = H1 + H2 + Hp + Hep

print('\nH1 is\n', H1)
print('\nH2 is\n', H2)
print('\nHep is\n', Hep)
print('Hp', Hp)

#print('\nHamiltonain is\n', H)

# Fermionic excitations (single and double)
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])

print('\n ----------- T-1 term -----------\n')
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
bra = braE1("occ", "vir")
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
print('T1 term=\n',final._print_einsum('T1'))
#print(final._print_einsum('T1'))
