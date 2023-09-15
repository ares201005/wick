# todolist
# 1) return navie scaling and optimized scaling, Done!
# 2) recycle imds and check the sign or transpose
# 3) del intermediate if they are not used later

"""
This is the auto code optimizer used to optimize the scaling and prefactor
of the auto-generated QED-CCSD and QED-EOM-CCSD code
"""

import copy
import sys
import opt_einsum as oe

unique_blocks = ["vvvv", "vvvo", "vovv", "vvoo", "vovo", "oovv", "vooo", "ooov", "oooo"]
block_map = {}
block_map["vvov"] = ["vvvo", -1.0, (0, 1, 3, 2)]
block_map["ovvv"] = ["vovv", -1.0, (1, 0, 2, 3)]
block_map["voov"] = ["vovo", -1.0, (0, 1, 3, 2)]
block_map["ovvo"] = ["vovo", -1.0, (1, 0, 2, 3)]
block_map["ovov"] = ["vovo", 1.0, (1, 0, 3, 2)]
block_map["oovo"] = ["ooov", -1.0, (0, 1, 3, 2)]
block_map["ovoo"] = ["vooo", -1.0, (1, 0, 2, 3)]

breakline = "*" * 80

index_map = {}
oinds = ["i", "j", "k", "l", "m", "n", "o"]
vinds = ["a", "b", "c", "d", "e", "f", "g"]
pinds = ["I", "J", "K", "L", "M", "N", "O"]

for s in oinds:
    index_map[s] = "o"
for s in vinds:
    index_map[s] = "v"
for s in pinds:
    index_map[s] = "p"


def check_ovblock(indices, block):
    """
    rewrite the two_e_blocks with unique blocks (listed above)
    """
    postfix = ""
    for x in indices:
        if ord(x) >= 105:
            postfix += "o"
        elif ord(x) >= 97:
            postfix += "v"
        elif ord(x) >= 73:
            postfix += "p"
    if postfix in unique_blocks:
        return indices, 1.0, block
    else:
        newblock, sign, permutation = block_map[postfix]
        newindices = indices[permutation]
        return newindices, sign, newblock


def gen_name(nspace, A, B, einsum_str):
    # nameA = tmp + A.replace(".","") + B.replace(".","")
    # nameB = tmp + B.replace(".","") + A.replace(".","")

    oldpostfixA = ""
    oldpostfixB = ""
    if len(A.split(".")) > 1:
        oldpostfixA += A.split(".")[1]
    if len(B.split(".")) > 1:
        oldpostfixB += B.split(".")[1]

    oldpostfixAB = oldpostfixA + oldpostfixB
    oldpostfixBA = oldpostfixB + oldpostfixA

    tmp = " " * nspace
    A = A.split(".")[0]
    B = B.split(".")[0]
    nameAB = tmp + A + B
    nameBA = tmp + B + A

    tmp = einsum_str.strip().split("->")[1]
    # a ascii value: 97,(i is 105)
    # I ascii value: 73 (A is 65)
    postfix = ""
    for x in tmp:
        if ord(x) >= 105:
            postfix += "o"
        elif ord(x) >= 97:
            postfix += "v"
        elif ord(x) >= 73:
            postfix += "p"

    postfixAB = postfix
    postfixBA = postfix
    if oldpostfixAB != postfix:
        postfixAB = oldpostfixAB + "2" + postfix
    if oldpostfixBA != postfix:
        postfixBA = oldpostfixBA + "2" + postfix

    # print(tmp, postfixAB)

    nameAB += postfixAB
    nameBA += postfixBA
    return nameAB, nameBA


def is_einsum_str_same(str1, str2):
    res = False
    print(str1, str2)
    str1.strip()
    str2.strip()
    prefix1 = str1.split("->")
    prefix2 = str2.split("->")

    if len(str1.strip()) == len(str2.strip()):
        res = True
        for i in range(len(str1)):
            if str1[i] in index_map:
                if index_map[str1[i]] != index_map[str2[i]]:
                    res = False
    return res


# this is code to optimize the auto-generated code of each equation
def optimize(expression, size_dict, imds, lhs, coeff, prefactor=False, prescan=True):
    # expression should only contain einsum("ab, bc, xx->", a, b, c, )

    # should do it iteratively: starting from higher order to lower ones

    data = expression.split("einsum('")[1]
    data = data.split(")")[0]

    einsum_string = data.split("',")[0]
    einsum_string = einsum_string.replace(" ", "")

    unique_inds = set(einsum_string) - {" ", ",", "-", ">"}
    unique_size = [size_dict[s] for s in unique_inds]

    tensors = data.split("',")[1].split(",")
    tensors = [s.strip() for s in tensors]
    tmp = dict(zip(unique_inds, unique_size))
    views = oe.helpers.build_views(einsum_string, tmp)

    # TODO: if the pattern is already found in preivous contraction, no need to do it again
    path, path_info = oe.contract_path(einsum_string, *views, optimize="optimal")

    indices = einsum_string.split("->")[0].split(",")
    indices = [s.strip() for s in indices]

    # print('\neinsum_string: {}, tensors={}; Len(tensor)={}'.format(einsum_string, tensors, len(tensors)) )
    # print('\npath_info=\n',  path_info,'\n')
    # print('\npath_info.path=\n',  path_info.path,'\n')

    optimized_equation = []
    if len(tensors) > 2 or prefactor:
        inds, idx_rm, einsum_str, remaining, do_blas = path_info.contraction_list[0]
        i, j = inds  # path[0]

        nspace = len(lhs) - len(lhs.lstrip())
        nameA, nameB = gen_name(nspace, tensors[i], tensors[j], einsum_str)

        # ===============================================================
        ### known issues: same name may have different construction,
        # for example:
        # in U11:  A = T1oldIoovv2ov = 1.0 * einsum('bj,jkbc->kc', T1old, I.oovv)
        # in S1:   B = T1oldIoovv2ov = 1.0 * einsum('bi,ijab->ja', T1old, I.oovv)
        # these two imds are different, recyle one for the other will result in error
        # we should check the permutation and adjust the sign accordingly.
        # In the above case, It should be B = - A. as A is (bi,ijba-> ja)
        #                                             B is (bi,ijab-> ja)
        # and ijab = - ijba #
        # to be fixed
        # ===============================================================

        imdname_found = False

        """
        if True:
        """
        if nameA in imds:
            # check if the two immediates have the same contruction rule
            imdname_found = is_einsum_str_same(imds[nameA][0], einsum_str)
            imdname = nameA
            imds[imdname].append(einsum_str)

        ##elif nameB in imds:
        ##    imdname = nameB
        if not imdname_found:
            imdname = nameA
            """
            # todo: here, we should compare the contruction rule, instead of string. 
            # for example: "ia,ijba -> jb" is the same as "ic,ijdc-> jd", just the labels 
            #used are slightly different
            """
            # if imdname in imds and check_contraction(einsum_str, imds[imdname]):
            #    print('imds found, but the contration is different:', imds[imdname], einsum_str)

            # here, we track how many times of the intermediate varaibles appeared,
            # if it only appears once, we don't need to construct the intermediate varaibles.

            imds[imdname] = []
            imds[imdname].append(einsum_str)

            # TODO: this expression cannot distinguish different blocks, ov, oo, vv, vo. Need to add postfix to reflect different blocks
            # print('indx=', inds, einsum_str, i, j)
            if not prefactor:
                newexpr = "".join("%s = 1.0 * einsum('") % imdname + "%s', %s, %s)" % (
                    einsum_str,
                    tensors[i],
                    tensors[j],
                )
                print("\n a new intermedite variable is added: ", newexpr)
                if not newexpr in optimized_equation:
                    optimized_equation.append(newexpr)

        if prefactor:
            print(
                "imdname = ", imdname, imds[imdname], " len=", len(imds[imdname])
            )  # [len(imds[imdname])-1])
        # print('\ninds: {}; idx_rm: {}; einsum_str: {}; remaining: {}.'.format( inds, idx_rm, einsum_str, remaining) )

        # -------------------------------------------
        if prefactor:
            if prescan:
                return None

            nimds = len(imds[imdname])
            if nimds > 1:
                print("relation between new and first intermediate var")
                lhs0, rhs0 = imds[imdname][0].split("->")
                for k in range(1, nimds):
                    lhs, rhs = imds[imdname][k].split("->")
                    print(lhs0, rhs0, lhs, rhs)
                sys.exit()
            return None
        # -------------------------------------------

        tmptensors = copy.deepcopy(tensors)
        remaining_tensors = [x for k, x in enumerate(tensors) if k not in inds]

        if remaining is not None:
            if len(remaining) > 1:
                remaining_str = (
                    remaining[-1]
                    + ", "
                    + ",".join(remaining[:-1])
                    + "->"
                    + path_info.output_subscript
                )
            else:
                remaining_str = ",".join(remaining) + "->" + path_info.output_subscript
            # print('remaing_expr={}'.format(remaining_str))

        # position of imd
        imd_pos = inds[1]
        remaining_tensors.insert(imd_pos, imdname.strip())
        # print('imd_pos=', imd_pos)
        # print('reminaing tensors=', remaining_tensors)

        newexpr = (
            "".join("%s+= %s * einsum('") % (lhs, coeff)
            + remaining_str
            + "',"
            + ",".join(remaining_tensors)
            + ")"
        )
        print("new equation is: {}".format(newexpr))
        optimized_equation.append(newexpr)

        ## this is wrong, the other terms are not appended correctly (todo)!!!!!
        # if k == 1:
        #    newexpr = "".join('%s = %s * einsum("') % (lhs, coeff) + '%s", %s, %s)' % (einsum_str,  tmptensors[i],  tmptensors[j])
        #    print('new expression is:', newexpr)
        #    optimized_equation.append(newexpr)
        # tmptensors[i] = imd #tmptensors[j]

    return optimized_equation


class codeoptimizer(object):
    def __init__(self, equations, no=10, nv=15, nm=2, fname="optimized_equations_opt"):
        """
        kwargs: dimensions of no, nv, nm
        """
        self.equations = equations
        self.optimized_eqs = []
        self.dims = [no, nv, nm]

        inds = []
        self.oinds = ["i", "j", "k", "l", "m", "n", "o"]
        self.vinds = ["a", "b", "c", "d", "e", "f", "g"]
        self.pinds = ["I", "J", "K", "L", "M", "N", "O"]
        inds.append(self.oinds)  # occupied
        inds.append(self.vinds)  # virtual
        inds.append(self.pinds)  # photon

        index_size = []
        for k in range(3):
            for s in inds[k]:
                index_size.append(self.dims[k])

        # size of each index
        inds = sum(inds, [])
        self.size_dict = dict(zip(inds, index_size))
        print(self.size_dict)

        self.imds = {}
        self.outputf = fname

    # break  D = alpha * einsum('experssion', A, B, C..) into
    # D, alpha, einsum('experssion', A, B, C..)

    def breakequation(self, equation):
        if not "einsum" in equation or not "=" in equation:
            return None, None, None

        if "+=" in equation:
            lhs, rhs = equation.split("+=")
        else:
            lhs, rhs = equation.split("=")

        if "*" in rhs:
            coeff, rhs = rhs.split("*")
        else:
            coeff = str(1.0)
        return lhs, coeff.strip(), rhs.strip()
        # return lhs.strip(), coeff.strip(), rhs.strip()

    def break_einsum(self, eq):
        """
        break einsum('expression', a, b, c) into
        expression, [a,b,c...]
        """
        expr, tensors = eq.split("',")
        expr = expr.split("('")[1]

        tensors = tensors.split(")")[0]
        return expr, tensors

    def checkorder(self, equations=None):
        """
        check highest order in the equations (order = number of tensors in einsum)
        """
        if equations == None:
            equations = self.equations

        maxorder = 0
        orders = []
        for eq in equations:
            if not "einsum" in eq or not "=" in eq or "backend.einsum" in eq:
                orders.append(0)
                continue

            lhs, coeff, rhs = self.breakequation(eq)
            expr, tensors = self.break_einsum(rhs)

            order = len(tensors.split(","))
            maxorder = max(order, maxorder)
            # print('original einsum ', rhs, 'is broken into\n', expr, tensors, 'order=', order)
            orders.append(order)

        # print('\n max order of the equations = ', maxorder)
        return maxorder, orders

    def check_scaling(self, equations=None):
        """
        return the scaling of the equations
        """

        if equations is None:
            equations = self.equations

        scaling = None
        e_scaling_max = 0
        p_scaling_max = 0
        for k, eq in enumerate(equations):
            if not "einsum(" in eq:
                continue
            lhs, coeff, rhs = self.breakequation(eq)
            expr, tensors = self.break_einsum(rhs)
            indices = expr.split("->")[0].split(",")
            indices = "".join(indices)
            indices = "".join(set(indices))
            e_scaling = 0
            p_scaling = 0
            for x in indices:
                if x in self.vinds or x in self.oinds:
                    e_scaling += 1
                elif x in self.pinds:
                    p_scaling += 1
            if e_scaling + p_scaling > (e_scaling_max + p_scaling_max):
                e_scaling_max = max(e_scaling_max, e_scaling)
                p_scaling_max = max(p_scaling_max, p_scaling)

            # print(expr, indices, e_scaling, p_scaling)
        scaling = "O(N^%d P^%d)" % (e_scaling_max, p_scaling_max)

        return scaling

    def rewrite_twoe_blocks(self):
        """
        rewrite twoe_blocks
        if following blocks presents, we use symmetry to construct the eq (todo)
        using symmetry will save 7/16 ~43% memory
        vvov
        ovvv
        voov
        ovvo
        ovov
        oovo
        ovoo
        """

        print("\n==============rewrite twoe blocks================")
        for k, eq in enumerate(equations):
            if not "einsum(" in eq:
                continue

            # 1) find two_e_blocks
            if "I.o" in eq or "I.v" in eq:
                print("two_e_blocks found:", eq)
                # then extract indices and blocks from the eq (todo)
                lhs, coeff, rhs = self.breakequation(eq)
                expr, tensors = self.break_einsum(rhs)

                expr = expr.split("->")[0].split(",")
                # expr = expr.split("'")[1].split("->")[0].split(",")
                tensors = tensors.split(",")
                for j in range(len(tensors)):
                    if "I.o" in tensors[j] or "I.v" in tensors[j]:
                        idx = j
                indices = expr[idx]
                block = tensors[idx].split(".")[1]

                # 2) check_ovblock
                newindices, sign, newblock = check_ovblock(indices, block)
                print(indices, block)
                if newblock != block or newindices != indices:
                    print(newindices, sign, newblock)

                # rewrite it if not the unique blocks (todo)
        sys.exit()

    def prefactor_reduction(self):
        """
        code reduction: reduce the prefactor, i.e., consider the symmetry
        in I and other intermediate variables to 1) combine similar terms
        together, 2) recycle intermediate var,

        for example: 1) D += einsum(A,B) and einsum(A,C) --> D += einsum(A,B+C);
                     2) D += einsum(A,B) and E += einsum(A,C); if C = permutation(A),
                     then, we dont' need to evaluate C, should use A instead.
        """

        print("")
        print(breakline)
        print(" prefactor reduction ")
        print(breakline, "\n")

        maxorder, orders = self.checkorder()
        output = open("%s_final.py" % (self.outputf), "w")

        self.optimized_eqs = []
        count = 0
        nqs = 0

        print("\n pre-scaning the all the imds\n")
        # pre-scan the imds
        recyclable_imds = []
        self.imds = {}
        for k, eq in enumerate(self.equations):
            if "backend" in eq:
                continue

            order = orders[k]
            lhs, coeff, rhs = self.breakequation(eq)
            if lhs is None:
                continue

            # print('\n original equation is: {} in order {:d}'.format(eq, order))
            expr_opt = optimize(
                rhs, self.size_dict, self.imds, lhs, coeff, prefactor=True
            )

        print("\n" + breakline)
        print(" prefactor reduction begins ")
        print(breakline + "\n")
        for k, eq in enumerate(self.equations):
            if "backend" in eq:
                continue

            order = orders[k]
            lhs, coeff, rhs = self.breakequation(eq)
            if lhs is None:
                continue

            # only optimzie these equations have similared imds:

            count += 1
            print("\n original equation is: {} ".format(eq))
            expr_opt = optimize(
                rhs,
                self.size_dict,
                self.imds,
                lhs,
                coeff,
                prefactor=True,
                prescan=False,
            )

        return None

    def scaling_reduction(self):
        """
        code optimization: 1) reduce the scaling
        """

        print("\nNo. of equations in original equations=", len(self.equations))
        scaling = self.check_scaling()
        print("\nnavie scaling of original equaiton", scaling)

        # replace the redudant two_e_blocks
        # self.rewrite_twoe_blocks()

        # check order
        maxorder, orders = self.checkorder()
        iteration = 0
        while maxorder > 2:
            iteration += 1
            print("\n===========iteration %d ===============\n" % iteration)
            maxorder, orders = self.checkorder()
            output = open("%s%d.py" % (self.outputf, maxorder - 1), "w")
            if iteration == 2:
                print("no. of equations =", len(self.equations))

            self.optimized_eqs = []
            count = 0
            nqs = 0
            for k, eq in enumerate(self.equations):
                order = orders[k]
                if order > 2:
                    # if order == maxorder:

                    lhs, coeff, rhs = self.breakequation(eq)
                    if lhs is None:
                        continue

                    count += 1

                    print("\n", "=" * 100)
                    print("\n ====original equation is: \n{} ".format(eq))
                    expr_opt = optimize(rhs, self.size_dict, self.imds, lhs, coeff)

                    print("\n ==== new equation is:")
                    if len(expr_opt) > 2:
                        print("errr: at lost 2 equations should be generated!!!")
                        sys.exit()

                    nqs += len(expr_opt)
                    for neweq in expr_opt:
                        print(neweq)
                        self.optimized_eqs.append(neweq)
                        output.write("%s\n" % neweq)

                    print("\n", "=" * 100)

                    # print(lhs, coeff, rhs, expr_opt)
                    print("\n {} equations left!!".format(len(self.equations) - k))
                else:
                    self.optimized_eqs.append(eq)
                    output.write("%s\n" % eq.rstrip())
                    nqs += 1

            output.close()

            print(
                "\n%d equations of order %d are optimized!"
                % (
                    count,
                    maxorder,
                )
            )
            print("No. of equations in optimized_eps=", len(self.optimized_eqs), nqs)

            self.equations = []
            self.equations = copy.deepcopy(self.optimized_eqs)

            maxorder, orders = self.checkorder(self.optimized_eqs)
            print("max order of optimized equations =", maxorder)

        scaling = self.check_scaling(self.optimized_eqs)
        self.equations = copy.deepcopy(self.optimized_eqs)
        print("\nscaling of optimized equaiton", scaling)

    def equation_optimization(self):
        # in the future, we need to combine the two steps together
        # step 1) reduce scaling
        self.scaling_reduction()

        # step 2) reduce prefactor
        self.prefactor_reduction()

        return self.optimized_eqs


if __name__ == "__main__":
    fname = "eom_epcc_equations_slow.py"
    fname = "epcc_equations_gen.py"
    lines = open(fname, "r").readlines()
    equations = []

    # for line in lines:
    #    if "einsum(" in line:
    #        equations.append(line)
    equations = lines

    print("\n%d equations in original format" % len(equations))
    outputfname = fname[:-3] + "_opt"
    print(outputfname)

    # change the function name
    for k, eq in enumerate(equations):
        if "def qed" in eq:
            data = eq.split("(")
            data[0] += "_opt("
            equations[k] = "".join(data)
            print("new function name=", equations[k])
    code = codeoptimizer(equations, fname=outputfname)

    optimized_eqs = code.equation_optimization()
    # print('\n%d equations after optimization' % len(optimized_eqs))

    # analyse scaling of equations before and after optimization (todo)

    # analyse the computational time (todo)

