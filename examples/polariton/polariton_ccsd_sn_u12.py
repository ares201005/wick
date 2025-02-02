
import sys
from sys import path as syspath
syspath.append('../')
syspath.append('../../')

from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, one_p, two_p, ep11
from wick.convenience import P3, P2, P1, E1, E2, EPS1, braE1, braE2, braPn, braP1, braP1E1, commute
from wick.convenience import Pn, braP2E1

# f: fork matrix
# I: two body integral (ERIs)
# w: vibrational modes
# g: e-ph coupling matrix
# h: 
# G: single model peice of Hamiltonian
# H:  
#

H1 = one_e("F", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
#H2 = two_e("I", ["occ", "vir"], norder=True)

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

print('\n H1  =\n', H1)
print('\n H2  =\n', H2)
print('\n Hep =\n', Hep)
print('\n Hp  =\n', Hp)

#print('\nHamiltonain is\n', H)

get_CCSD = False
get_CCSD = True
get_EOMCCSD = True

# test braPn
bra = braP1('nm') 
bra1 = braPn('nm',n=4) 
print(bra)
print(bra1)

nfock = 3
if get_CCSD:
    # Fermionic excitations (single and double)
    T1 = E1("T1old", ["occ"], ["vir"])
    T2 = E2("T2old", ["occ"], ["vir"])

    # Bosonic excitation
    #S1 = P1("S1old", ["nm"])
    #S2 = P2("s2", ["nm"])
    #S3 = P3("s3", ["nm"])
    #print('S1=', S1,'\n')
    #print('S2=', S2,'\n')
    #print('S3=', S3,'\n')

    T = T1 + T2
    # Bosonic excitation up to nfock order
    for i in range(1,nfock+1):
        name = 'S%s'%i + 'old'
        Sn = Pn(name, ["nm"], n=i)
        T += Sn
        print('S%s='%i, Sn)
    # coupled fermion-boson excitation
    U11 = EPS1("U11old", ["nm"], ["occ"], ["vir"])
    #T = T1 + T2 + S1 + S2 + U11
    T += U11

    print('\n ----------- CCSD energy -----------\n')
    HT = commute(H, T)
    HTT = commute(HT, T)
    HTTT = commute(HTT, T)
    HTTTT = commute(HTTT, T)

    Hbar = H + HT + Fraction('1/2')*HTT
    Hbar += Fraction('1/6')*HTTT
    Hbar += Fraction('1/24')*HTTTT

    S0 = Hbar
    E0 = apply_wick(S0)
    E0.resolve()
    final = AExpression(Ex=E0)
    print('\n',final._print_einsum('Eccsd'))

    print('\n ----------- T-1 term -----------\n')
    bra = braE1("occ", "vir")
    HT = commute(H, T)
    HTT = commute(HT, T)
    HTTT = commute(commute(commute(H2+Hep, T1), T1), T1)

    # e^{-T} H e^T 
    # <HF | e^{-T} H e^{T} | HF> 
    S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(final)
    print(final._print_einsum('T1'))

    print('\n ----------- T-2 term -----------\n')
    bra = braE2("occ", "vir", "occ", "vir")
    HTTT = commute(HTT, T)
    HTTTT = commute(HTTT, T)
    Hbar = H + HT + Fraction('1/2')*HTT
    Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

    S = bra*Hbar
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(final)
    print(final._print_einsum('T2'))

    Hbar = H + HT + Fraction('1/2')*HTT
    Hbar += Fraction('1/6')*HTTT
    #bra = braP1('nm') 

    for i in range(1,nfock+1):
       print('\n ----------- p%s term -----------\n' % i)
       sys.stdout.flush()
       bra = braPn('nm',n=i) 
       S = bra*Hbar
       out = apply_wick(S)
       out.resolve()
       final = AExpression(Ex=out)
       print(final)
       print(final._print_einsum('S%s'%i))

    print('\n ----------- p1E1 term -----------\n')
    bra = braP1E1('nm','occ', 'vir') 
    Hbar = H + HT + Fraction('1/2')*HTT
    Hbar += Fraction('1/6')*HTTT

    S = bra*Hbar
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(final)
    print(final._print_einsum('U11'))

    print('\n ----------- U12 term -----------\n')
    bra = braP2E1('nm','occ', 'vir') 
    Hbar = H + HT + Fraction('1/2')*HTT
    Hbar += Fraction('1/6')*HTTT

    S = bra*Hbar
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(final)
    print(final._print_einsum('U12'))



from wick.convenience import one_e, two_e, E1, E2, braE1, E1, E2, commute
if get_EOMCCSD:
    #======================================================================
    #   EOM_CCSD for polariton
    #======================================================================
    print(' \n EOM-CCSD equations\n')
    
    print('\n CCSD Hamiltonian=\n')
    print(H)

    T1 = E1("T1", ["occ"], ["vir"])
    T2 = E2("T2", ["occ"], ["vir"])

    #S1 = P1("S1", ["nm"])
    #S2 = P2("s2", ["nm"])
    #S3 = P3("s3", ["nm"])
    #print('S1=', S1,'\n')
    #print('S2=', S2,'\n')
    #print('S3=', S3,'\n')

    # coupled fermion-boson excitation
    U11 = EPS1("U11", ["nm"], ["occ"], ["vir"])

    RS = E1("RS", ["occ"], ["vir"])
    RD = E2("RD", ["occ"], ["vir"])
    R1 = P1("R1", ["nm"])
    R11 = EPS1("R11", ["nm"], ["occ"], ["vir"])

    T = T1 + T2 + U11
    R = RS + RD + R11

    # Bosonic excitation up to nfock order
    for i in range(1,nfock+1):
        name = 'S%s'%i
        Sn = Pn(name, ["nm"], n=i)
        T += Sn
        print('\nS%s='%i, Sn)

        name = 'R%s'%i
        Rn = Pn(name, ["nm"], n=i)
        R += Rn

    #print('\nRS=', RS)
    #print('\nRD=', RD)

    HT = commute(H, T)
    HTT = commute(HT, T)
    HTTT = commute(HTT, T)
    HTTTT = commute(HTTT, T)
    Hbar = H + HT + Fraction('1/2')*HTT

    S0 = Hbar
    E0 = apply_wick(S0)
    E0.resolve()

    final = AExpression(Ex=E0)

    Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

    # =================  EOM - sigS part =================
    print('\nEOM-CCSD sigS part\n')
    bra = braE1("occ", "vir")
    sys.stdout.flush()

    S = bra*(Hbar - E0)*R

    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    final.sort_tensors()
    print(final)
    print(final._print_einsum('SigS'))

    sys.stdout.flush()

    # =================  EOM - sigD part =================
    print('\nEOM-CCSD sigD part\n')
    bra = braE2("occ", "vir", "occ", "vir")

    S = bra*(Hbar - E0)*R

    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(final)
    print('\n einsum format=\n')
    print(final._print_einsum('SigD'))
    sys.stdout.flush()

    # =================  EOM - sign part =================
    for i in range(1, nfock+1):
        bra = braPn('nm', n=i)
        print('\nEOM-CCSD Sig%s part\n'%i)

        bra = braP1('nm') 
        S = bra*(Hbar - E0)*R
        out = apply_wick(S)
        out.resolve()
        final = AExpression(Ex=out)

        print(final)
        print('\n einsum format=\n')
        print(final._print_einsum('Sig%'%i))
        sys.stdout.flush()

    # =================  EOM - sigU1 part =================
    print('\nEOM-CCSD sigU1 part\n')
    bra = braP1E1('nm','occ', 'vir') 

    S = bra*(Hbar - E0)*R

    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(final)
    print('\n einsum format=\n')
    print(final._print_einsum('SigU1'))

