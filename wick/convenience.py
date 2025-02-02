# Copyright (c) 2020-2021 Alec White
# Licensed under the MIT License (see LICENSE for details)
from fractions import Fraction
from .index import Idx
from .operator import Projector, BOperator, FOperator
from .operator import TensorSym, Tensor, Sigma, normal_ordered
from .expression import Term, Expression


def one_e(name, spaces, norder=False, index_key=None):
    """
    Return an expression representing all pieces of a one-electron operator

    name (str): Name of the operator.
    spaces (list): List orbital subspaces
    norder (bool): Return only normal-ordered part?
    """
    terms = []
    for s1 in spaces:
        p = Idx(0, s1)
        for s2 in spaces:
            i = 0 if s2 != s1 else 1
            q = Idx(i, s2)
            operators = [FOperator(p, True), FOperator(q, False)]
            sigmas = [Sigma(p), Sigma(q)]
            tensors = [Tensor([p, q], name)]
            nsign = 1
            if norder:
                operators, nsign = normal_ordered(
                    [FOperator(p, True), FOperator(q, False)])
            t = Term(
                nsign, sigmas, tensors, operators, [], index_key=index_key)
            terms.append(t)
    return Expression(terms)


def get_sym(anti):
    if anti:
        return TensorSym(
            [(0, 1, 2, 3), (1, 0, 2, 3),
             (0, 1, 3, 2), (1, 0, 3, 2)],
            [1, -1, -1, 1])
    else:
        return TensorSym([(0, 1, 2, 3), (1, 0, 3, 2)], [1, 1])


def get_sym_ip2():
    return TensorSym([(0, 1, 2), (0, 2, 1)], [1, -1])


def get_sym_ea2():
    return TensorSym([(0, 1, 2), (1, 0, 2)], [1, -1])


def two_e_compressed(name, spaces, anti=True, norder=False, index_key=None):
    if not anti:
        raise Exception(
            "Minimal representation of symmetric integrals is not implemented")
    terms = []
    sym = get_sym(anti)
    one_half = Fraction(1, 2)
    for i1, s1 in enumerate(spaces):
        p = Idx(0, s1)
        for s2 in spaces[i1:]:
            i = 0 if s2 != s1 else 1
            q = Idx(i, s2)
            facbra = one_half if s1 == s2 else 1
            for i3, s3 in enumerate(spaces):
                xx = list(filter(lambda x: x, [s3 == s for s in [s1, s2]]))
                r = Idx(len(xx), s3)
                for s4 in spaces[i3:]:
                    xx = list(filter(
                        lambda x: x, [s4 == s for s in [s1, s2, s3]]))
                    s = Idx(len(xx), s4)
                    operators = [
                        FOperator(p, True), FOperator(q, True),
                        FOperator(s, False), FOperator(r, False)]
                    sigmas = [Sigma(p), Sigma(q), Sigma(r), Sigma(s)]
                    tensors = [Tensor([p, q, r, s], name, sym=sym)]
                    nsign = 1
                    facket = one_half if s3 == s4 else 1
                    fac = facbra*facket
                    if norder:
                        operators, nsign = normal_ordered(operators)
                    t = Term(
                        nsign*fac, sigmas, tensors,
                        operators, [], index_key=index_key)
                    terms.append(t)
    return Expression(terms)


def two_e_full(name, spaces, anti=True, norder=False, index_key=None):
    terms = []
    sym = get_sym(anti)
    fac = Fraction(1, 4) if anti else Fraction(1, 2)
    for s1 in spaces:
        p = Idx(0, s1)
        for s2 in spaces:
            i = 0 if s2 != s1 else 1
            q = Idx(i, s2)
            for s3 in spaces:
                xx = list(filter(lambda x: x, [s3 == s for s in [s1, s2]]))
                r = Idx(len(xx), s3)
                for s4 in spaces:
                    xx = list(filter(
                        lambda x: x, [s4 == s for s in [s1, s2, s3]]))
                    s = Idx(len(xx), s4)
                    operators = [
                        FOperator(p, True), FOperator(q, True),
                        FOperator(s, False), FOperator(r, False)]
                    sigmas = [Sigma(p), Sigma(q), Sigma(r), Sigma(s)]
                    tensors = [Tensor([p, q, r, s], name, sym=sym)]
                    nsign = 1
                    if norder:
                        operators, nsign = normal_ordered(operators)
                    t = Term(
                        nsign*fac, sigmas, tensors,
                        operators, [], index_key=index_key)
                    terms.append(t)
    return Expression(terms)


def two_e(name, spaces, anti=True,
          norder=False, compress=False, index_key=None):
    """
    Return expression representing a two electron operator

    name (str): Name of the operator
    spaces (list): List of orbital spaces
    anti (bool): Return anti-symmetrized representation
    norder (bool): Return only normal-ordered part
    compress (bool): Return only unique parts
    """
    if compress:
        return two_e_compressed(
            name, spaces, anti=anti, norder=norder, index_key=index_key)
    else:
        return two_e_full(
            name, spaces, anti=anti, norder=norder, index_key=index_key)


def one_p(name, space="nm", name2=None, index_key=None):
    """
    Return expression representing a 1-boson operator

    name (str): Name of operator
    space (str): Name of boson space
    name2 (str): Name of creation piece if different from name
    """
    if name2 is None:
        name2 = name
    I1 = Idx(0, space, fermion=False)
    tc = Term(
        1, [Sigma(I1)], [Tensor([I1], name2)],
        [BOperator(I1, True)], [], index_key=index_key)
    ta = Term(
        1, [Sigma(I1)], [Tensor([I1], name)],
        [BOperator(I1, False)], [], index_key=index_key)
    terms = [tc, ta]
    return Expression(terms)


def two_p(name, space="nm", index_key=None):
    """
    Return expression representing a 1-boson operator

    name (str): Name of operator
    space (str): Name of boson space
    """
    # x, y, index to bosonic modes

    x = Idx(0, space, fermion=False)
    y = Idx(1, space, fermion=False)
    t1 = Term(
        1, [Sigma(x), Sigma(y)], [Tensor([x, y], name)],
        [BOperator(x, True), BOperator(y, False)], [], index_key=index_key)
    return Expression([t1])

def occ_p(name, space="nm", index_key=None):
    """
    Return expression representing a 1-boson operator

    name (str): Name of operator
    space (str): Name of boson space
    """
    # x, y, index to bosonic modes

    x = Idx(0, space, fermion=False)
    t1 = Term(
        1, [Sigma(x), Sigma(x)], [Tensor([x, x], name)],
        [BOperator(x, True), BOperator(x, False)], [], index_key=index_key)
    return Expression([t1])

def _get_tc(x, p1, p2, name2, norder, index_key):
    sigmas = [Sigma(x), Sigma(p1), Sigma(p2)]
    if norder:
        operators, nsign = normal_ordered(
            [FOperator(p1, True), FOperator(p2, False)])
    else:
        nsign = 1
        operators = [FOperator(p1, True), FOperator(p2, False)]
    operators = [BOperator(x, True)] + operators
    tensors = [Tensor([x, p1, p2], name2)]
    tc = Term(nsign, sigmas, tensors, operators, [], index_key=index_key)
    return tc


def _get_ta(x, p1, p2, name, norder, index_key):
    sigmas = [Sigma(x), Sigma(p1), Sigma(p2)]
    if norder:
        operators, nsign = normal_ordered(
            [FOperator(p1, True), FOperator(p2, False)])
    else:
        nsign = 1
        operators = [FOperator(p1, True), FOperator(p2, False)]
    operators = [BOperator(x, False)] + operators
    tensors = [Tensor([x, p1, p2], name)]
    ta = Term(nsign, sigmas, tensors, operators, [], index_key=index_key)
    return ta


def ep11(name, fspaces, bspaces, norder=False, name2=None, index_key=None):
    """
    Return an coupled boson fermion operator

    name (str): Name of operator
    fspaces (list): Femion orbital spaces
    bspaces (list): Boson orbital spaces
    norder (bool): Only normal ordered piece?
    name2 (str): Name of boson creation piece if different
    """
    terms = []
    if name2 is None:
        name2 = name
    for sb in bspaces:
        x = Idx(0, sb, fermion=False)
        for s1 in fspaces:
            p1 = Idx(0, s1)
            for s2 in fspaces:
                i = 0 if s2 != s1 else 1
                p2 = Idx(i, s2)
                tc = _get_tc(x, p1, p2, name2, norder, index_key)
                ta = _get_ta(x, p1, p2, name, norder, index_key)
                terms.append(ta)
                terms.append(tc)
    return Expression(terms)


def E0(name, index_key=None):
    """
    Return constant multiplying the vacuum.

    name (str): Name of operator
    """
    return Expression([
        Term(1, [], [Tensor([], name)], [], [], index_key=index_key)])


def deE1(name, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion excitation operator

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for os in ospaces:
        for vs in vspaces:
            i = Idx(0, os)
            a = Idx(0, vs)
            sigmas = [Sigma(i), Sigma(a)]
            tensors = [Tensor([a, i], name)]
            operators = [FOperator(i, True), FOperator(a, False)]
            e1 = Term(1, sigmas, tensors, operators, [], index_key=index_key)
            terms.append(e1)
    return Expression(terms)

def deE2(name, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion excitation operator

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = get_sym(True)
    for i1, o1 in enumerate(ospaces):
        for o2 in ospaces[i1:]:
            for j1, v1 in enumerate(vspaces):
                for v2 in vspaces[j1:]:
                    i = Idx(0, o1)
                    a = Idx(0, v1)
                    j = Idx(1, o2)
                    b = Idx(1, v2)
                    scalar = 1
                    if o1 == o2:
                        scalar *= Fraction(1, 2)
                    if v1 == v2:
                        scalar *= Fraction(1, 2)
                    sums = [Sigma(a), Sigma(i), Sigma(b), Sigma(j)]
                    tensors = [Tensor([a, b, i,j], name, sym=sym)]
                    operators = [
                        FOperator(i, True), FOperator(j, True),
                        FOperator(b, False), FOperator(a, False)]
                    e2 = Term(scalar, sums, tensors, operators,
                              [], index_key=index_key)
                    terms.append(e2)
    return Expression(terms)

def deP1(name, spaces, index_key=None):
    """
    Return the tensor representation of a Boson excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    for s in spaces:
        x = Idx(0, s, fermion=False)
        sums = [Sigma(x)]
        tensors = [Tensor([x], name)]
        operators = [BOperator(x, False)]
        e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
        terms.append(e1)
    return Expression(terms)

def deEPS1(name, bspaces, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion-Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for bs in bspaces:
        for os in ospaces:
            for vs in vspaces:
                x = Idx(0, bs, fermion=False)
                i = Idx(0, os)
                a = Idx(0, vs)
                sums = [Sigma(x), Sigma(i), Sigma(a)]
                tensors = [Tensor([x, a, i], name)]
                operators = [BOperator(x, False),
                             FOperator(i, True), FOperator(a, False)]
                e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
                terms.append(e1)
    return Expression(terms)


def E1(name, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion excitation operator

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for os in ospaces:
        for vs in vspaces:
            i = Idx(0, os)
            a = Idx(0, vs)
            sigmas = [Sigma(i), Sigma(a)]
            tensors = [Tensor([a, i], name)]
            operators = [FOperator(a, True), FOperator(i, False)]
            e1 = Term(1, sigmas, tensors, operators, [], index_key=index_key)
            terms.append(e1)
    return Expression(terms)


def E2(name, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion excitation operator

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = get_sym(True)
    for i1, o1 in enumerate(ospaces):
        for o2 in ospaces[i1:]:
            for j1, v1 in enumerate(vspaces):
                for v2 in vspaces[j1:]:
                    i = Idx(0, o1)
                    a = Idx(0, v1)
                    j = Idx(1, o2)
                    b = Idx(1, v2)
                    scalar = 1
                    if o1 == o2:
                        scalar *= Fraction(1, 2)
                    if v1 == v2:
                        scalar *= Fraction(1, 2)
                    sums = [Sigma(i), Sigma(a), Sigma(j), Sigma(b)]
                    tensors = [Tensor([a, b, i, j], name, sym=sym)]
                    operators = [
                        FOperator(a, True), FOperator(b, True),
                        FOperator(j, False), FOperator(i, False)]
                    e2 = Term(scalar, sums, tensors, operators,
                              [], index_key=index_key)
                    terms.append(e2)
    return Expression(terms)


def Eip1(name, ospaces, index_key=None):
    """
    Return the tensor representation of a Fermion ionization

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    """
    terms = []
    for os in ospaces:
        i = Idx(0, os)
        sums = [Sigma(i)]
        tensors = [Tensor([i], name)]
        operators = [FOperator(i, False)]
        e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
        terms.append(e1)
    return Expression(terms)


def Eip2(name, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion ip (trion)

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = get_sym_ip2()
    for i1, o1 in enumerate(ospaces):
        for o2 in ospaces[i1:]:
            for v1 in vspaces:
                i = Idx(0, o1)
                a = Idx(0, v1)
                j = Idx(1, o2)
                sums = [Sigma(i), Sigma(a), Sigma(j)]
                tensors = [Tensor([a, i, j], name, sym=sym)]
                operators = [FOperator(a, True),
                             FOperator(j, False), FOperator(i, False)]
                s = Fraction('1/2')
                e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
                terms.append(e2)
    return Expression(terms)


def Eea1(name, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion attachment operator

    name (string): name of the tensor
    ospaces (list): list of virtual spaces
    """
    terms = []
    for vs in vspaces:
        a = Idx(0, vs)
        sums = [Sigma(a)]
        tensors = [Tensor([a], name)]
        operators = [FOperator(a, True)]
        e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
        terms.append(e1)
    return Expression(terms)


def Eea2(name, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a Fermion ea (trion)

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = get_sym_ea2()
    for o1 in ospaces:
        for j1, v1 in enumerate(vspaces):
            for v2 in vspaces[j1:]:
                i = Idx(0, o1)
                a = Idx(0, v1)
                b = Idx(1, v2)
                sums = [Sigma(i), Sigma(a), Sigma(b)]
                tensors = [Tensor([b, a, i], name, sym=sym)]
                operators = [FOperator(b, True),
                             FOperator(a, True), FOperator(i, False)]
                s = Fraction('1/2')
                e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
                terms.append(e2)
    return Expression(terms)



def P1(name, spaces, index_key=None, creation=True):
    """
    Return the tensor representation of a Boson excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    for s in spaces:
        x = Idx(0, s, fermion=False)
        sums = [Sigma(x)]
        tensors = [Tensor([x], name)]
        operators = [BOperator(x, creation)]
        e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
        terms.append(e1)
    return Expression(terms)


def P2(name, spaces, index_key=None, creation=True):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    sym = TensorSym([(0, 1), (1, 0)], [1, 1])
    for s1 in spaces:
        for s2 in spaces:
            x = Idx(0, s1, fermion=False)
            i = 1 if s1 == s2 else 0
            y = Idx(i, s2, fermion=False)
            sums = [Sigma(x), Sigma(y)]
            tensors = [Tensor([x, y], name, sym=sym)]
            operators = [BOperator(x, creation), BOperator(y, creation)]
            s = Fraction('1/2')
            e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
            terms.append(e2)
    return Expression(terms)

def P3(name, spaces, index_key=None, creation=True):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    sym = TensorSym([(0, 1, 2), (1, 2, 0)], [1, 1])
    for s1 in spaces:
        for s2 in spaces:
            for s3 in spaces:
                x = Idx(0, s1, fermion=False)
                i = 1 if s1 == s2 else 0
                y = Idx(i, s2, fermion=False)
                i = 2 if s1 == s3 else 0
                z = Idx(i, s3, fermion=False)
                sums = [Sigma(x), Sigma(y), Sigma(z)]
                tensors = [Tensor([x, y, z], name, sym=sym)]
                operators = [BOperator(x, creation), BOperator(y, creation), BOperator(z, creation)]
                s = Fraction('1/6')
                e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
                terms.append(e2)
    return Expression(terms)

def P4(name, spaces, index_key=None, creation=True):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    sym = TensorSym([(0, 1, 2, 3), (1, 2, 3, 0)], [1, 1])
    for s1 in spaces:
        for s2 in spaces:
            for s3 in spaces:
               for s4 in spaces:
                 x = Idx(0, s1, fermion=False)
                 i = 1 if s1 == s2 else 0
                 y = Idx(i, s2, fermion=False)
                 i = 2 if s1 == s3 else 0
                 z = Idx(i, s3, fermion=False)
                 i = 3 if s1 == s4 else 0
                 w = Idx(i, s3, fermion=False)
                 sums = [Sigma(x), Sigma(y), Sigma(z), Sigma(w)]
                 tensors = [Tensor([x, y, z, w], name, sym=sym)]
                 operators = [BOperator(x, creation), BOperator(y, creation),
                         BOperator(z, creation), BOperator(w, creation)]
                 s = Fraction('1/24')
                 e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
                 terms.append(e2)
    return Expression(terms)


def P5(name, spaces, index_key=None, creation=True):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    sym = TensorSym([(0, 1, 2, 3, 4), (1, 2, 3, 4, 0)], [1, 1])
    for s1 in spaces:
        for s2 in spaces:
            for s3 in spaces:
               for s4 in spaces:
                 for s5 in spaces:
                     x = Idx(0, s1, fermion=False)
                     i = 1 if s1 == s2 else 0
                     y = Idx(i, s2, fermion=False)
                     i = 2 if s1 == s3 else 0
                     z = Idx(i, s3, fermion=False)
                     i = 3 if s1 == s4 else 0
                     w = Idx(i, s3, fermion=False)
                     i = 4 if s1 == s5 else 0
                     u = Idx(i, s3, fermion=False)

                     sums = [Sigma(x), Sigma(y), Sigma(z), Sigma(w), Sigma(u)]
                     tensors = [Tensor([x, y, z, w, u], name, sym=sym)]
                     operators = [BOperator(x, creation), BOperator(y, creation),
                             BOperator(z, creation), BOperator(w, creation), BOperator(u, creation)]
                     s = Fraction('1/120')
                     e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
                     terms.append(e2)
    return Expression(terms)

def gen_terms (terms, creation, name, sym, index_key, norder, number, spaces, slist):

    if (number > 0):
       for s in spaces:
           slist.append(s)
           gen_terms(terms, creation, name, sym, index_key, norder, number - 1, spaces, slist)
           slist.pop()
    else:
       #print('test_slit=', slist)
       xyz = []
       for k, s in enumerate(slist):
           i = 0
           if s == slist[0]:
               i = k
           #print('print test-i', k, i,s, slist[0], s==slist[0])
           x = Idx(i, s, fermion = False)
           xyz.append(x)

       sums = [Sigma(x) for x in xyz]
       tensors = [Tensor(xyz, name, sym=sym)]
       operators = [BOperator(x, creation) for x in xyz]
       s = Fraction('1/%d'%norder)
       e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
       terms.append(e2)

def Pn(name, spaces, index_key=None, n = 1, creation=True):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    list1 = tuple(range(n))
    list2 = tuple(list(list1[1:]) + list([list1[0]]))
    sym = TensorSym([list1, list2], [1, 1])
    norder = 1
    for i in range(1,n+1):
        norder *= i

    for s1 in spaces:
        slist = []
        slist.append(s1)
        gen_terms(terms, creation, name, sym,index_key, norder, n-1, spaces, slist)
    return Expression(terms)



def Pn_old(name, spaces, index_key=None, n = 1, creation=True):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """

    if n == 1:
        return P1(name, spaces, index_key, creation)
    elif n == 2:
        return P2(name, spaces, index_key, creation)
    elif n == 3:
        return P3(name, spaces, index_key, creation)
    elif n == 4:
        return P4(name, spaces, index_key, creation)
    elif n == 5:
        return P5(name, spaces, index_key, creation)
    else:
        print('warning: only fock space <=5 is supported')
        sys.exit()



def EPS1(name, bspaces, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion-Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for bs in bspaces:
        for os in ospaces:
            for vs in vspaces:
                x = Idx(0, bs, fermion=False)
                i = Idx(0, os)
                a = Idx(0, vs)
                sums = [Sigma(x), Sigma(i), Sigma(a)]
                tensors = [Tensor([x, a, i], name)]
                operators = [BOperator(x, True),
                             FOperator(a, True), FOperator(i, False)]
                e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
                terms.append(e1)
    return Expression(terms)


def EPS2(name, bspaces, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion-double Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = TensorSym([(0, 1, 2, 3), (1, 0, 2, 3)], [1, 1])
    for b1 in bspaces:
        for b2 in bspaces:
            for os in ospaces:
                for vs in vspaces:
                    x = Idx(0, b1, fermion=False)
                    i = 1 if b1 == b2 else 0
                    y = Idx(i, b2, fermion=False)
                    i = Idx(0, os)
                    a = Idx(0, vs)
                    sums = [Sigma(x), Sigma(y), Sigma(i), Sigma(a)]
                    tensors = [Tensor([x, y, a, i], name, sym=sym)]
                    operators = [BOperator(x, True), BOperator(y, True),
                                 FOperator(a, True), FOperator(i, False)]
                    s = Fraction('1/2')
                    e1 = Term(
                        s, sums, tensors, operators, [], index_key=index_key)
                    terms.append(e1)
    return Expression(terms)


def EPS3(name, bspaces, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion-double Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = TensorSym([(0, 1, 2, 3, 4), (1, 2, 0, 3, 4)], [1, 1])
    for b1 in bspaces:
       for b2 in bspaces:
          for b3 in bspaces:
            for os in ospaces:
                for vs in vspaces:
                    x = Idx(0, b1, fermion=False)
                    i = 1 if b1 == b2 else 0
                    j = 2 if b1 == b3 else 0
                    y = Idx(i, b2, fermion=False)
                    z = Idx(j, b3, fermion=False)
                    i = Idx(0, os)
                    a = Idx(0, vs)
                    sums = [Sigma(x), Sigma(y), Sigma(z), Sigma(i), Sigma(a)]

                    tensors = [Tensor([x, y, z, a, i], name, sym=sym)]
                    operators = [BOperator(x, True), BOperator(y, True), BOperator(z, True),
                                 FOperator(a, True), FOperator(i, False)]
                    s = Fraction('1/6')
                    e1 = Term(
                        s, sums, tensors, operators, [], index_key=index_key)
                    terms.append(e1)
    return Expression(terms)

def gen_u2terms (terms, creation, name, sym, index_key, norder, number, bspaces, ospaces, vspaces, slist):

    if (number > 0):
       for s in bspaces:
           slist.append(s)
           gen_u2terms(terms, creation, name, sym, index_key, norder, number - 1, bspaces, ospaces, vspaces, slist)
           slist.pop()
    else:
       #print('test_slit=', slist)
       for i1, o1 in enumerate(ospaces):
           for o2 in ospaces[i1:]:
               for j1, v1 in enumerate(vspaces):
                   for v2 in vspaces[j1:]:
                       xyz = []
                       for k, s in enumerate(slist):
                           i = 0
                           if s == slist[0]:
                               i = k
                           x = Idx(i, s, fermion = False)
                           xyz.append(x)

                       i = Idx(0, o1)
                       a = Idx(0, v1)
                       j = Idx(1, o2)
                       b = Idx(1, v2)
                       abij = [a, b, i, j]


                       scalar = 1
                       if o1 == o2:
                           scalar *= Fraction(1, 2)
                       if v1 == v2:
                           scalar *= Fraction(1, 2)

                       sums = [Sigma(x) for x in xyz] + [Sigma(i), Sigma(a), Sigma(j), Sigma(b)]
                       tensors = [Tensor(xyz+abij, name, sym=sym)]
                       E2 = [FOperator(a, True), FOperator(b, True),
                       FOperator(j, False), FOperator(i, False)]

                       operators = [BOperator(x, creation) for x in xyz] + E2
                       scalar *= Fraction('1/%d'%norder)
                       e2 = Term(scalar, sums, tensors, operators, [], index_key=index_key)
                       terms.append(e2)

def gen_uterms (terms, creation, name, sym, index_key, norder, number, bspaces, ospaces, vspaces, slist):

    if (number > 0):
       for s in bspaces:
           slist.append(s)
           gen_uterms(terms, creation, name, sym, index_key, norder, number - 1, bspaces, ospaces, vspaces, slist)
           slist.pop()
    else:
       #print('test_slit=', slist)
       for os in ospaces:
           for vs in vspaces:
               xyz = []
               for k, s in enumerate(slist):
                   i = 0
                   if s == slist[0]:
                       i = k
                   #print('print test-i', k, i,s, slist[0], s==slist[0])
                   x = Idx(i, s, fermion = False)
                   xyz.append(x)

               i = Idx(0, os)
               a = Idx(0, vs)
               ai = [a, i]

               sums = [Sigma(x) for x in xyz] + [Sigma(i), Sigma(a)]

               tensors = [Tensor(xyz+ai, name, sym=sym)]
               operators = [BOperator(x, creation) for x in xyz] + [FOperator(a, True), FOperator(i, False)]
               s = Fraction('1/%d'%norder)
               e2 = Term(s, sums, tensors, operators, [], index_key=index_key)
               terms.append(e2)

def EPSn(name, bspaces, ospaces, vspaces, n=1, index_key=None, creation=True):
    """
    Return the tensor representation of a coupled
    Fermion-double Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    list1 = tuple(range(n+2))
    list2 = tuple(list(list1[1:-2]) + list([list1[0]])+list(list1[-2:]))
    sym = TensorSym([list1, list2], [1, 1])
    norder = 1
    for i in range(1,n+1):
        norder *= i

    for s1 in bspaces:
        slist = []
        slist.append(s1)
        gen_uterms(terms, creation, name, sym,index_key, norder, n-1, bspaces, ospaces, vspaces, slist)
    return Expression(terms)

def E2PSn(name, bspaces, ospaces, vspaces, n=1, index_key=None, creation=True):
    """
    Return the tensor representation of a coupled
    Fermion-double Boson excitation operator

    Fermion: double excitation
    Boson: n excitation (>=1)

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """

    terms = []
    list1 = tuple(range(n+4))
    list2 = tuple(list(list1[1:-4]) + list([list1[0]])+list(list1[-4:]))
    # a <-> i
    list3 = tuple(list(list1[:-4]) + [list1[-3],list1[-4], list1[-2], list1[-1]] ) #-1
    # b <-> j
    list4 = tuple(list(list1[:-4]) + [list1[-4], list1[-3], list1[-1], list1[-2]] ) #-1
    # (a, b) <-> (i, j)
    list5 = tuple(list(list1[:-4]) + [list1[-3], list1[-4], list1[-1], list1[-2]] ) #1
    #

    print("List1=", list1)
    print("List2=", list2)
    print("List3=", list3)
    print("List4=", list4)
    print("List5=", list5)

    sym = TensorSym([list1, list2, list3, list4, list5], [1, 1, -1, -1, 1])
    norder = 1
    for i in range(1,n+1):
        norder *= i

    for s1 in bspaces:
        slist = []
        slist.append(s1)
        gen_u2terms(terms, creation, name, sym,index_key, norder, n-1, bspaces, ospaces, vspaces, slist)
    return Expression(terms)


def EP1ip1(name, bspaces, ospaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion ionization-Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    """
    terms = []
    for bs in bspaces:
        for os in ospaces:
            x = Idx(0, bs, fermion=False)
            i = Idx(0, os)
            sums = [Sigma(x), Sigma(i)]
            tensors = [Tensor([x, i], name)]
            operators = [BOperator(x, True), FOperator(i, False)]
            e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
            terms.append(e1)
    return Expression(terms)


def EP1ea1(name, bspaces, vspaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion attachment-Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of virtial spaces
    """
    terms = []
    for bs in bspaces:
        for vs in vspaces:
            x = Idx(0, bs, fermion=False)
            a = Idx(0, vs)
            sums = [Sigma(x), Sigma(a)]
            tensors = [Tensor([x, a], name)]
            operators = [BOperator(x, True), FOperator(a, True)]
            e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
            terms.append(e1)
    return Expression(terms)


def braE1(ospace, vspace, index_key=None):
    """
    Return left-projector onto a space of single excitations

    ospace (str): occupied space
    vspace (str): virtual space
    """
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [FOperator(i, True), FOperator(a, False)]
    return Expression([
        Term(1, [], [Tensor([a, i], "")], operators, [], index_key=index_key)])


def braE2(o1, v1, o2, v2, index_key=None):
    """
    Return left-projector onto a space of double excitations

    o1 (str): 1st occupied space
    v1 (str): 1st virtual space
    o2 (str): 2nd occupied space
    v2 (str): 2nd virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    jx = 1 if o2 == o1 else 0
    bx = 1 if v2 == v1 else 0
    j = Idx(jx, o1)
    b = Idx(bx, v1)
    operators = [
        FOperator(i, True), FOperator(j, True),
        FOperator(b, False), FOperator(a, False)]
    tensors = [Tensor([a, b, i, j], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def braEip1(ospace, index_key=None):
    """
    Return left-projector onto a space of ionized determinants

    ospace (str): occupied space
    """
    i = Idx(0, ospace)
    operators = [FOperator(i, True)]
    return Expression([
        Term(1, [], [Tensor([i], "")], operators, [], index_key=index_key)])


def braEip2(o1, o2, v1, index_key=None):
    """
    Return left-projector onto a space of (trion) N-1 particle determinants

    o1 (str): first occupied space
    o2 (str): second occupied space
    v1 (str): virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    jx = 1 if o2 == o1 else 0
    j = Idx(jx, o2)
    operators = [FOperator(i, True), FOperator(j, True), FOperator(a, False)]
    tensors = [Tensor([a, i, j], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def braEdip1(o1, o2, index_key=None):
    """
    Return left-projector onto a space of N-2 particle determinants

    o1 (str): first occupied space
    o2 (str): second occupied space
    """
    i = Idx(0, o1)
    jx = 1 if o2 == o1 else 0
    j = Idx(jx, o2)
    operators = [FOperator(i, True), FOperator(j, True)]
    return Expression([
        Term(1, [], [Tensor([i, j], "")], operators, [], index_key=index_key)])


def braEea1(space, index_key=None):
    """
    Return left-projector onto a space of N+1 electron states

    space (str): orbital space
    """
    a = Idx(0, space)
    operators = [FOperator(a, False)]
    return Expression([
        Term(1, [], [Tensor([a], "")], operators, [], index_key=index_key)])


def braEea2(o1, v1, v2, index_key=None):
    """
    Return left-projector onto a space of (trion) N+1 electron states

    o1 (str): occupied space
    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    bx = 1 if v2 == v1 else 0
    b = Idx(bx, v2)
    operators = [FOperator(i, True), FOperator(b, False), FOperator(a, False)]
    tensors = [Tensor([a, b, i], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def braEdea1(v1, v2, index_key=None):
    """
    Return left-projector onto a space of N+2 electron states

    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    a = Idx(0, v1)
    bx = 1 if v2 == v1 else 0
    b = Idx(bx, v2)
    operators = [FOperator(b, False), FOperator(a, False)]
    return Expression([
        Term(1, [], [Tensor([a, b], "")], operators, [], index_key=index_key)])


def braP1(space, index_key=None):
    """
    Return projection onto single Boson space

    space (str): Name of boson space
    """
    x = Idx(0, space, fermion=False)
    operators = [BOperator(x, False)]
    return Expression([
        Term(1, [], [Tensor([x], "")], operators, [], index_key=index_key)])


def braP2(space, index_key=None):
    """
    Return projection onto space of Boson pairs

    space (str): Name of boson space
    """
    x = Idx(0, space, fermion=False)
    y = Idx(1, space, fermion=False)
    operators = [BOperator(x, False), BOperator(y, False)]
    tensors = [Tensor([x, y], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def braP3(space, index_key=None):
    """
    Return projection onto space of Boson pairs

    space (str): Name of boson space
    """
    x = Idx(0, space, fermion=False)
    y = Idx(1, space, fermion=False)
    z = Idx(2, space, fermion=False)
    operators = [BOperator(x, False), BOperator(y, False),BOperator(z, False)]
    tensors = [Tensor([x, y, z], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def braPn(space, index_key=None,n=4):
    """
    Return projection onto space of Boson pairs

    space (str): Name of boson space
    """

    operators = []
    tensor_list = []

    for k in range(n):
        x = Idx(k, space, fermion=False)
        operators.append(BOperator(x, False))
        tensor_list.append(x)

    tensors = [Tensor(tensor_list, "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def braP1E1(bspace, ospace, vspace, index_key=None):
    """
    Return left-projector onto a space of single excitations coupled to
    boson excitations

    bspace (str): boson space
    ospace (str): occupied space
    vspace (str): virtual space
    """
    x = Idx(0, bspace, fermion=False)
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [BOperator(x, False), FOperator(i, True), FOperator(a, False)]
    tensors = [Tensor([x, a, i], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])

def braP2E1(b1space, b2space, ospace, vspace, index_key=None):
    """
    Return left-projector onto a space of single excitations coupled pairs
    of bosons

    b1space (str): first boson space
    b2space (str): second boson space
    ospace (str): occupied space
    vspace (str): virtual space
    """
    x = Idx(0, b1space, fermion=False)
    yx = 1 if b1space == b2space else 0
    y = Idx(yx, b2space, fermion=False)
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [
        BOperator(x, False), BOperator(y, False),
        FOperator(i, True), FOperator(a, False)]
    tensors = [Tensor([x, y, a, i], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])

def braPnE1(bspace_list, ospace, vspace, index_key=None):
    """
    Return left-projector onto a space of single excitations coupled pairs
    of bosons

    bspace [(str)]: boson space list
    ospace (str): occupied space
    vspace (str): virtual space
    """

    xyz = []
    for k, bspace in enumerate(bspace_list):
        i = 0
        if bspace == bspace_list[0]:
            i = k
        x = Idx(i, bspace, fermion=False)
        xyz.append(x)

    i = Idx(0, ospace)
    a = Idx(0, vspace)
    ai = [a,i]
    operators = [BOperator(x, False) for x in xyz] + [FOperator(i, True), FOperator(a, False)]
    tensors = [Tensor(xyz+ai, "")]
    return Expression([Term(1, [], tensors, operators, [], index_key=index_key)])

def braPnE2(bspace_list, o1, v1, o2, v2, index_key=None):
    """
    Return left-projector onto a space of single excitations coupled pairs
    of bosons

    bspace [(str)]: boson space list
    o1 (str): 1st occupied space
    v1 (str): 1st virtual space
    o2 (str): 2nd occupied space
    v2 (str): 2nd virtual space
    """

    xyz = []
    for k, bspace in enumerate(bspace_list):
        i = 0
        if bspace == bspace_list[0]:
            i = k
        x = Idx(i, bspace, fermion=False)
        xyz.append(x)

    i = Idx(0, o1)
    a = Idx(0, v1)
    jx = 1 if o2 == o1 else 0
    bx = 1 if v2 == v1 else 0
    j = Idx(jx, o1)
    b = Idx(bx, v1)
    ai = [a, b, i, j]

    E2 = [
        FOperator(i, True), FOperator(j, True),
        FOperator(b, False), FOperator(a, False)]
    operators = [BOperator(x, False) for x in xyz] + E2
    tensors = [Tensor(xyz+ai, "")]
    return Expression([Term(1, [], tensors, operators, [], index_key=index_key)])



def braP1Eea1(bspace, vspace, index_key=None):
    """
    Return left-projector onto a space of N+1 electron states coupled to
    boson excitations

    bspace (str): boson space
    vspace (str): orbital space
    """
    x = Idx(0, bspace, fermion=False)
    a = Idx(0, vspace)
    operators = [BOperator(x, False), FOperator(a, False)]
    return Expression([
        Term(1, [], [Tensor([x, a], "")], operators, [], index_key=index_key)])


def braP1Eip1(bspace, ospace, index_key=None):
    """
    Return left-projector onto a space of N-1 electron states coupled to
    boson excitations

    bspace (str): boson space
    ospace (str): orbital space
    """
    x = Idx(0, bspace, fermion=False)
    i = Idx(0, ospace)
    operators = [BOperator(x, False), FOperator(i, True)]
    return Expression([
        Term(1, [], [Tensor([x, i], "")], operators, [], index_key=index_key)])


def ketE1(ospace, vspace, index_key=None):
    """
    Return right-projector onto a space of single excitations

    ospace (str): occupied space
    vspace (str): virtual space
    """
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [FOperator(a, True), FOperator(i, False)]
    return Expression([
        Term(1, [], [Tensor([i, a], "")], operators, [], index_key=index_key)])


def ketE2(o1, v1, o2, v2, index_key=None):
    """
    Return right-projector onto a space of double excitations

    o1 (str): 1st occupied space
    v1 (str): 1st virtual space
    o2 (str): 2nd occupied space
    v2 (str): 2nd virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    jx = 1 if o2 == o1 else 0
    bx = 1 if v2 == v1 else 0
    j = Idx(jx, o1)
    b = Idx(bx, v1)
    operators = [
        FOperator(a, True), FOperator(b, True),
        FOperator(j, False), FOperator(i, False)]
    tensors = [Tensor([i, j, a, b], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def ketEea1(space, index_key=None):
    """
    Return right-projector onto a space of N+1 electron states

    space (str): orbital space
    """
    a = Idx(0, space)
    operators = [FOperator(a, True)]
    return Expression([
        Term(1, [], [Tensor([a], "")], operators, [], index_key=index_key)])


def ketEea2(o1, v1, v2, index_key=None):
    """
    Return right-projector onto a space of trion N+1 electron states

    o1 (str): occupied space
    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    bx = 1 if v2 == v1 else 0
    b = Idx(bx, v2)
    operators = [FOperator(a, True), FOperator(b, True), FOperator(i, False)]
    tensors = [Tensor([i, a, b], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def ketEip1(space, index_key=None):
    """
    Return right-projector onto a space of N-1 electron states

    space (str): orbital space
    """
    i = Idx(0, space)
    operators = [FOperator(i, False)]
    return Expression([
        Term(1, [], [Tensor([i], "")], operators, [], index_key=index_key)])


def ketEip2(o1, o2, v1, index_key=None):
    """
    Return right-projector onto a space of trion N-1 electron states

    o1 (str): occupied space
    o2 (str): second occupied space
    v1 (str): first virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    jx = 1 if o2 == o1 else 0
    j = Idx(jx, o2)
    operators = [FOperator(a, True), FOperator(j, False), FOperator(i, False)]
    tensors = [Tensor([i, j, a], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def ketEdea1(v1, v2, index_key=None):
    """
    Return right-projector onto a space of N+2 electron states

    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    a = Idx(0, v1)
    bx = 1 if v2 == v1 else 0
    b = Idx(bx, v2)
    operators = [FOperator(a, True), FOperator(b, True)]
    return Expression([
        Term(1, [], [Tensor([a, b], "")], operators, [], index_key=index_key)])


def ketEdip1(o1, o2, index_key=None):
    """
    Return right-projector onto a space of N+2 electron states

    o1 (str): first occupied space
    o2 (str): second occupied space
    """
    i = Idx(0, o1)
    jx = 1 if o2 == o1 else 0
    j = Idx(jx, o2)
    operators = [FOperator(j, False), FOperator(i, False)]
    return Expression([
        Term(1, [], [Tensor([i, j], "")], operators, [], index_key=index_key)])


def ketP1(space, index_key=None):
    """
    Return right-projection onto single Boson space
    """
    x = Idx(0, space, fermion=False)
    return Expression([
        Term(1, [], [Tensor([x], "")],
             [BOperator(x, True)], [], index_key=index_key)])


def ketP2(space, index_key=None):
    """
    Return right-projection onto space of Boson pairs
    """
    x = Idx(0, space, fermion=False)
    y = Idx(1, space, fermion=False)
    return Expression([
        Term(
            1, [], [Tensor([x, y], "")],
            [BOperator(x, True), BOperator(y, True)],
            [], index_key=index_key)])

def ketPn(space, index_key=None,n=4):
    """
    Return projection onto space of Boson pairs

    space (str): Name of boson space
    """

    operators = []
    tensor_list = []
    for k in range(n):
        x = Idx(k, space, fermion=False)
        operators.append(BOperator(x, True))
        tensor_list.append(x)

    tensors = [Tensor(tensor_list, "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def ketP1E1(bspace, ospace, vspace, index_key=None):
    """
    Return right-projector onto a space of single excitations coupled to
    boson excitations

    bspace (str): boson space
    ospace (str): occupied space
    vspace (str): virtual space
    """
    x = Idx(0, bspace, fermion=False)
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [BOperator(x, True), FOperator(a, True), FOperator(i, False)]
    tensors = [Tensor([x, i, a], "")]
    return Expression([
        Term(1, [], tensors, operators, [], index_key=index_key)])


def ketP1Eea1(bspace, vspace, index_key=None):
    """
    Return right-projector onto a space of electron attachment coupled to
    boson excitations

    bspace (str): boson space
    vspace (str): virtual space
    """
    x = Idx(0, bspace, fermion=False)
    a = Idx(0, vspace)
    operators = [BOperator(x, True), FOperator(a, True)]
    return Expression([
        Term(1, [], [Tensor([x, a], "")], operators, [], index_key=index_key)])


def ketP1Eip1(bspace, ospace, index_key=None):
    """
    Return right-projector onto a space of N-1 electron states coupled to
    boson excitations

    bspace (str): boson space
    ospace (str): orbital space
    """
    x = Idx(0, bspace, fermion=False)
    i = Idx(0, ospace)
    operators = [BOperator(x, True), FOperator(i, False)]
    return Expression([
        Term(1, [], [Tensor([x, i], "")], operators, [], index_key=index_key)])


def PE1(ospace, vspace, index_key=None):
    """
    Return the projector onto a space of single excitations

    ospace (str): occupied space
    vspace (str): virtual space
    """
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    P = Projector()
    operators = [
        FOperator(a, True), FOperator(i, False),
        P, FOperator(i, True), FOperator(a, False)]
    exp = Expression([
        Term(1, [Sigma(i), Sigma(a)], [], operators, [], index_key=index_key)])
    return exp


def commute(A, B):
    """ Return the commutator of two operators"""
    return A*B - B*A
