from itertools import product
from .index import Idx
from .operator import Operator, TensorSym, Tensor, Sigma
from .expression import Term, Expression

def one_e(name, spaces, indices=['ijklmno','abcdefg']):
    assert(len(spaces) == len(indices))
    terms = []
    for s1,i1 in zip(spaces,indices):
        I1 = Idx(i1[0], s1)
        for s2,i2 in zip(spaces,indices):
            i = i2[0] if s2 != s1 else i2[1]
            I2 = Idx(i,s2)
            t = Term(1.0, [Sigma(I1),Sigma(I2)],
                    [Tensor([I1,I2],name)],
                    [Operator(I1, True), Operator(I2, False)],
                    [])
            terms.append(t)
    return Expression(terms)

def get_sym(anti):
    if anti:
        return TensorSym([(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2)],
                [1.0, -1.0, -1.0, 1.0])
    else:
        return TensorSym([(0,1,2,3),(1,0,3,2)],[1.0,1.0])

def two_e(name, spaces, indices=['ijklmno','abcdefg'], anti=True):
    assert(len(spaces) == len(indices))
    terms = []
    sym = get_sym(anti)
    fac = 0.25 if anti else 0.5
    for s1,i1 in zip(spaces,indices):
        I1 = Idx(i1[0], s1)
        for s2,i2 in zip(spaces,indices):
            i = i2[0] if s2 != s1 else i2[1]
            I2 = Idx(i,s2)
            for s3,i3 in zip(spaces, indices):
                xx = filter(lambda x: x,[s3 == s for s in [s1,s2]])
                I3 = Idx(i3[len(xx)],s3)
                for s4,i4 in zip(spaces, indices):
                    xx = filter(lambda x: x,[s4 == s for s in [s1,s2,s3]])
                    I4 = Idx(i4[len(xx)],s4)
                    t = Term(fac, [Sigma(I1),Sigma(I2),Sigma(I3),Sigma(I4)],
                            [Tensor([I1,I2,I3,I4],name,sym=sym)],
                            [Operator(I1, True), Operator(I2, True), Operator(I4,False), Operator(I3,False)],
                            [])
                    terms.append(t)
    return Expression(terms)
