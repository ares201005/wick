import sys
import numpy
import time
import copy

# from numpy import allclose
from epcc_equations_gen import *
from epcc_equations_gen_shared import *
from epcc_equations_gen_opt2 import *

# from epcc_equations_opt import*

# from pyscf import lib as pyscflib
from opt_einsum import contract
import opt_einsum as oe

# code to test opt_einsum contraction

from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks, two_e_blocks_full

import torch
from pyscf.polaritonlib.backend import NumpyBackend, TorchBackend
from pyscf.polaritonlib.backend import backend as bd
from pyscf.polaritonlib.backend import set_backend
from pyscf import gto, scf, cc

breakline = "*" * 90
# Set desired formatting options
float_formatter = lambda x: f"  {x:.4f}"
numpy.set_printoptions(formatter={"float_kind": float_formatter})


def torch_speedup(bd, n):
    start = time.time()
    a = numpy.random.rand(n, n)
    b = numpy.random.rand(n, n)
    c = numpy.einsum("ik,kj->ij", a, b)
    # print('c=', c)
    end = time.time()
    t1 = end - start

    if isinstance(bd, TorchBackend):
        c = torch.Tensor(c)
        start = time.time()
        a = torch.Tensor(a)
        b = torch.Tensor(b)
        d = bd.einsum("ik,kj->ij", a, b)
        end = time.time()
        t2 = end - start

        d = d - c
        # print('\nd=', d)
        print(
            "size {} T1= {:.6f} T2= {:.6f} speedup= {:.6f}".format(n, t1, t2, t1 / t2)
        )


def test_torch_speedup():
    print("if distributed available?", torch.distributed.is_available())

    print("test speedup on torch gpu")
    set_backend("torch")
    # set_backend("numpy")
    einsum = bd.einsum

    # bd.set_backend("torch")
    for k in range(500, 4000, 500):
        torch_speedup(bd, k)


def genImd(no, nv, backend=NumpyBackend):
    # Note ijkl = - jikl = jilk.
    #
    # for a general matrix A, A+A^T is symmetric matrix
    #                         A-A^T is antisymmetic
    nmo = nv + no
    nao = nmo // 2

    # Cmo = backend.rand(nmo)
    # from pyscf import ao2mo
    # eri = ao2mo.general(mol, [C,]*4, compact=False).reshape([nmo,]*4)

    eri = backend.rand(nmo, nmo, nmo, nmo)
    eri[:nao, nao:] = eri[nao:, :nao] = eri[:, :, :nao, nao:] = eri[
        :, :, nao:, :nao
    ] = 0
    Ua_mo = backend.transpose(eri, [0, 2, 1, 3]) - backend.transpose(eri, [0, 2, 3, 1])

    temp = [i for i in range(nmo)]
    oidx = temp[:no]
    vidx = temp[no:nmo]

    vvvv = Ua_mo[numpy.ix_(vidx, vidx, vidx, vidx)]
    vvvo = Ua_mo[numpy.ix_(vidx, vidx, vidx, oidx)]
    vovv = Ua_mo[numpy.ix_(vidx, oidx, vidx, vidx)]
    vvoo = Ua_mo[numpy.ix_(vidx, vidx, oidx, oidx)]
    vovo = Ua_mo[numpy.ix_(vidx, oidx, vidx, oidx)]
    oovv = Ua_mo[numpy.ix_(oidx, oidx, vidx, vidx)]
    vooo = Ua_mo[numpy.ix_(vidx, oidx, oidx, oidx)]
    ooov = Ua_mo[numpy.ix_(oidx, oidx, oidx, vidx)]
    oooo = Ua_mo[numpy.ix_(oidx, oidx, oidx, oidx)]

    # enforce the symmetry
    """
    vvov = -vvvo.transpose(0,1,3,2)
    ovvv = -vovv.transpose(1,0,2,3)
    voov = -vovo.transpose(0,1,3,2)
    ovvo = -vovo.transpose(1,0,2,3)
    ovov =  vovo.transpose(1,0,3,2)
    oovo = -ooov.transpose(0,1,3,2)
    ovoo = -vooo.transpose(1,0,2,3)
    """

    vvov = None  # -backend.transpose(vvvo,[0,1,3,2])
    ovvv = None  # -backend.transpose(vovv,[1,0,2,3])
    voov = None  # -backend.transpose(vovo,[0,1,3,2])
    ovvo = None  # -backend.transpose(vovo,[1,0,2,3])
    ovov = None  # backend.transpose(vovo,[1,0,3,2])
    oovo = None  # -backend.transpose(ooov,[0,1,3,2])
    ovoo = None  # -backend.transpose(vooo,[1,0,2,3])

    return two_e_blocks_full(
        vvvv=vvvv,
        vvvo=vvvo,
        vovv=vovv,
        voov=voov,
        ovvv=ovvv,
        ovoo=ovoo,
        oovo=oovo,
        vvov=vvov,
        ovvo=ovvo,
        ovov=ovov,
        vvoo=vvoo,
        oovv=oovv,
        vovo=vovo,
        vooo=vooo,
        ooov=ooov,
        oooo=oooo,
    )


def gen_fock(no, nv, backend=NumpyBackend):
    # ensure symmetry (todo)
    Foo = backend.rand(no, no)
    Fvv = backend.rand(nv, nv)
    Fov = backend.rand(no, nv)
    Fvo = backend.transpose(Fov)
    for i in range(no):
        for j in range(i + 1, no):
            Foo[i, j] = Foo[j, i]
    for i in range(nv):
        for j in range(i + 1, nv):
            Fov[i, j] = Fov[j, i]

    return one_e_blocks(Foo, Fov, Fvo, Fvv)


def g_int(no, nv, nm, backend=NumpyBackend):
    # ensure symmetry; ov = vo
    oo = backend.rand(nm, no, no)
    ov = backend.rand(nm, no, nv)
    vo = backend.rand(nm, nv, no)
    vv = backend.rand(nm, nv, nv)

    return one_e_blocks(oo, ov, vo, vv)


def benchmark_ccsd(norb=10, nm=2, nrun=5, backend=NumpyBackend):
    no = norb // 2
    nv = norb // 2

    print("no. of orbital = ", norb, no, nv)
    print("no. of photon modes  = ", nm)
    print("backend is", backend)

    w = backend.rand(nm)
    G = backend.rand(nm)
    H = copy.copy(G)

    I = genImd(no, nv, backend)
    F = gen_fock(no, nv, backend)

    g = g_int(no, nv, nm, backend)
    h = copy.copy(g)

    # ======================================================
    if False:
        R = 3.2
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 " + str(R),  # in au
            basis="631g",
            unit="Bohr",
            symmetry=True,
        )
        method = "uhf"
        mol.build()
        xc = "b3lyp"
        nm = 1
        omega = numpy.zeros(nm)
        vec = numpy.zeros((nm, 3))
        omega[0] = 0.1  # 2.7/27.211
        vec[0, :] = [1.0, 1.0, 1.0]

        from pyscf.polariton.eom_epcc import eom_ee_epccsd_1_s1
        from pyscf.polariton.eom_epcc import eom_ee_epccsd_n_sn
        from pyscf.polariton.polaritoncc import epcc, epcc_nfock
        from pyscf.polariton.abinit import Model

        gfac = 0.005  # 5
        model = Model(mol, method, xc, omega, vec, gfac, shift=False)
        print("omega=", model.omega())

        F = model.g_fock()
        eo = F.oo.diagonal()
        ev = F.vv.diagonal()

        no = eo.shape[0]
        nv = ev.shape[0]

        I = model.g_aint()
        w = model.omega()
        g, h = model.gint()
        G, H = model.mfG()
    # ======================================================

    nfock = 1
    nfock2 = 1

    T1 = backend.rand(nv, no)
    T2 = backend.rand(nv, nv, no, no)

    # ssn and su1n
    ssn = [None] * nfock
    su1n = [None] * nfock2

    Snold = [None] * nfock
    U1nold = [None] * nfock2

    eo = backend.rand(no)
    ev = backend.rand(nv)

    D1p = eo[None, :, None] - ev[None, None, :] - w[:, None, None]
    D1p0 = copy.copy(D1p)

    # check if torch transpose is same as the numpy transpose
    if isinstance(bd, TorchBackend):
        D1p_numpy = D1p.numpy()
        D1ptrans_numpy = numpy.transpose(D1p_numpy, [0, 2, 1])
        # print('debug-D1transpose_numpy=', D1ptrans_numpy)
        D1ptrans = backend.transpose(D1p, [0, 2, 1])
        # print('debug-D1transpose_tensor=', D1ptrans.numpy())

    for k in range(nfock):
        if k == 0:
            Snold[k] = -H / w
        else:
            shape = [nm for j in range(k + 1)]
            Snold[k] = backend.zeros(tuple(shape))

    for k in range(nfock2):
        if k == 0:
            U1nold[k] = h.vo / backend.transpose(D1p, [0, 2, 1])
            # U1nold[k] =  h.vo/D1p.transpose(0,2,1)
        else:
            shape = [nm for j in range(k + 1)]
            U1nold[k] = backend.zeros(tuple(shape))

    amps = (T1, T2, Snold, U1nold)

    # ccsd
    start = time.time()
    # einsum = numpy.einsum
    einsum = backend.einsum

    # T1_0, T2_0, Sn_0, U1n_0 = qed_ccsd_sn_u1n_gen_opt(F, I, w, g, h, G, H, nfock, nfock2, amps, backend)
    T1_0, T2_0, Sn_0, U1n_0 = qed_ccsd_sn_u1n_gen(
        F, I, w, g, h, G, H, nfock, nfock2, amps, backend
    )

    end = time.time()
    time0 = end - start

    """
    # benchmark eom
    start = time.time()
    einsum = pyscflib.einsum
    amps = (T1, T2, Sn, U1n)
    #initialize RS, RD, Rn, Rsn 
    RS, RD, Rn, RSn = T1, T2, Sn, U1n
    sigS0, sigD0, sign0, sigSn0 = eom_ee_epccsd_n_sn_sigma_slow(nfock1, nfock2, 
            RS, RD, Rn, RSn, amps, F, I, w, g, h, G, H)
    end = time.time()
    time3 = end - start 
    """
    nmethod = 5
    times = numpy.zeros(nmethod)
    for irun in range(nrun):
        print(
            "\n========================== %d-th run ==================================\n"
            % irun
        )
        # method 1): unoptimized code
        start = time.time()
        T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen(
            F, I, w, g, h, G, H, nfock, nfock2, amps, backend
        )
        end = time.time()
        times[0] += end - start
        print("computational time", end - start)
        print(
            "is amps close?    ",
            backend.allclose(T1_0, T1),
            backend.allclose(T2_0, T2),
            [backend.allclose(Sn_0[k], Sn[k]) for k in range(nfock)],
            [backend.allclose(U1n_0[k], U1n[k]) for k in range(nfock2)],
            "\n",
        )
        print(
            "amps difference is ",
            backend.norm(T1_0 - T1),
            backend.norm(T2_0 - T2),
            [backend.norm(Sn_0[k] - Sn[k]) for k in range(nfock)],
            [backend.norm(U1n_0[k] - U1n[k]) for k in range(nfock2)],
            "\n",
        )
        # method 1): optimized code
        start = time.time()
        # T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen(F, I, w, g, h, G, H, nfock, nfock2, amps, backend)
        T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen_opt(
            F, I, w, g, h, G, H, nfock, nfock2, amps, backend
        )
        end = time.time()
        times[1] += end - start
        print("computational time", end - start)
        print(
            "is amps close?    ",
            backend.allclose(T1_0, T1),
            backend.allclose(T2_0, T2),
            [backend.allclose(Sn_0[k], Sn[k]) for k in range(nfock)],
            [backend.allclose(U1n_0[k], U1n[k]) for k in range(nfock2)],
            "\n",
        )
        print(
            "amps difference is ",
            backend.norm(T1_0 - T1),
            backend.norm(T2_0 - T2),
            [backend.norm(Sn_0[k] - Sn[k]) for k in range(nfock)],
            [backend.norm(U1n_0[k] - U1n[k]) for k in range(nfock2)],
            "\n",
        )

        # method 2) unoptimized code using opt_einsum
        start = time.time()
        einsum = contract
        T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen(
            F, I, w, g, h, G, H, nfock, nfock2, amps, backend, einsum
        )
        end = time.time()

        times[2] += end - start
        print("computational time", end - start)
        print(
            "is amps close?    ",
            backend.allclose(T1_0, T1),
            backend.allclose(T2_0, T2),
            [backend.allclose(Sn_0[k], Sn[k]) for k in range(nfock)],
            [backend.allclose(U1n_0[k], U1n[k]) for k in range(nfock2)],
            "\n",
        )
        print(
            "amps difference is ",
            backend.norm(T1_0 - T1),
            backend.norm(T2_0 - T2),
            [backend.norm(Sn_0[k] - Sn[k]) for k in range(nfock)],
            [backend.norm(U1n_0[k] - U1n[k]) for k in range(nfock2)],
            "\n",
        )

        # method 3): unoptimized code using shared intermediates
        # start = time.time()
        # this is wrong
        # with oe.shared_intermediates() as shared_context:
        #    T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen_shared(F, I, w, g, h, G, H, nfock, nfock2, amps, backend, shared_context)
        # end = time.time()
        # times[3] += end - start
        times[3] = times[0]
        print("computational time", end - start)
        print(
            "is amps close?    ",
            backend.allclose(T1_0, T1),
            backend.allclose(T2_0, T2),
            [backend.allclose(Sn_0[k], Sn[k]) for k in range(nfock)],
            [backend.allclose(U1n_0[k], U1n[k]) for k in range(nfock2)],
            "\n",
        )
        print(
            "amps difference is ",
            backend.norm(T1_0 - T1),
            backend.norm(T2_0 - T2),
            [backend.norm(Sn_0[k] - Sn[k]) for k in range(nfock)],
            [backend.norm(U1n_0[k] - U1n[k]) for k in range(nfock2)],
            "\n",
        )

        # method 4): optimized code using opteinsum
        start = time.time()
        einsum = contract
        T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen_opt(
            F, I, w, g, h, G, H, nfock, nfock2, amps, backend
        )
        end = time.time()
        times[4] += end - start
        print("computational time:", end - start)
        print(
            "is amps close?    :",
            backend.allclose(T1_0, T1),
            backend.allclose(T2_0, T2),
            [backend.allclose(Sn_0[k], Sn[k]) for k in range(nfock)],
            [backend.allclose(U1n_0[k], U1n[k]) for k in range(nfock2)],
            "\n",
        )
        print(
            "amps difference is:",
            backend.norm(T1_0 - T1),
            backend.norm(T2_0 - T2),
            [backend.norm(Sn_0[k] - Sn[k]) for k in range(nfock)],
            [backend.norm(U1n_0[k] - U1n[k]) for k in range(nfock2)],
            "\n",
        )

    for i in range(nmethod):
        times[i] /= nrun
    print("\n" + breakline)
    # Set desired formatting options

    # Convert the array into a formatted string
    formatted_times = numpy.array2string(times)
    speedup = numpy.array([times[0] / times[i] for i in range(1, nmethod)])
    formatted_speedup = numpy.array2string(speedup)

    print(f"times of orbital {norb} photon mode {nm}=", formatted_times)
    print(f"speedup is   ", formatted_speedup)
    print(breakline + "\n")
    print("")

    sys.stdout.flush()

    # pyscf einsum lib is actually doing the contraction
    # start = time.time()
    # einsum = pyscflib.einsum
    # T1, T2, Sn, U1n = qed_ccsd_sn_u1n_gen(F, I, w, g, h, G, H, nfock, nfock2, amps, einsum)
    # end = time.time()
    # print('amps got, time=', end - start, numpy.allclose(T1_0, T1), '\n')
    #

    return times


def benchmark_gputensors(norb=10, nm=2, nrun=5, backend=NumpyBackend):
    A = backend.rand(norb, norb)
    B = backend.rand(norb, norb)
    C = backend.rand(norb, norb)
    I = backend.rand(norb, norb, norb, norb)

    # D = A*B*C*I
    start = time.time()
    D = backend.einsum("ij,jp,mn,pqmn->iq", A, B, C, I)
    t1 = time.time() - start

    start = time.time()
    E = backend.einsum("mn,mnpq->pq", C, I)
    # del I

    F = backend.einsum("ij,jp->ip", A, B)
    G = backend.einsum("ip,pq->iq", E, F)
    t2 = time.time() - start
    print("times comparision=", t1, t2, t1 / t2)

    return t1, t2, t2


# ----------------------------------------------
if __name__ == "__main__":
    norb = 20
    nm = 5

    """
    if len(sys.argv) > 1:
        norb = int(sys.argv[1])
    if len(sys.argv) > 2:
        nm = int(sys.argv[2])
    benchmark_ccsd(norb,nm)
    """

    # set backend
    for back in ["torch", "numpy"]:
        set_backend(back)

        # ------------------
        # CCSD
        # ------------------
        orbitals = range(20, 40, 10)
        orbitals = range(100, 320, 100)
        orbitals = range(20, 210, 20)
        orbitals = range(20, 201, 20)
        nm = 1

        n = len(orbitals)
        nrun = 10
        if bd == NumpyBackend:
            nrun = 5

        times = numpy.zeros((n, 5))
        for i, norb in enumerate(orbitals):
            # times[i,:] = benchmark_gputensors(norb,nm,nrun=1, backend=bd)
            times[i, :] = benchmark_ccsd(norb, nm, nrun=nrun, backend=bd)
            print(times[i, :])

        # list times
        print("\n" + breakline + f"\nsummary of cpu times with {back}:")
        print(
            "Orbitals    unopt        opt         unopt(oe)     unopt(oeshare)    opt(oe)     speedup"
        )
        print(breakline)

        for i in range(n):
            print(
                orbitals[i],
                "{:12.4f}   {:12.4f}  {:12.4f}   {:12.4f}   {:12.4f}   {:9.3f}".format(
                    times[i, 0],
                    times[i, 1],
                    times[i, 2],
                    times[i, 3],
                    times[i, 4],
                    times[i, 0] / times[i, 1],
                ),
            )
        print(breakline + "\n")

        # ------------------
        # eom-ccsd
        # ------------------

        # --------------------------
        # gpu (via torch speedup)
        # --------------------------
        """
        if True:
            test_torch_speedup()
        """
