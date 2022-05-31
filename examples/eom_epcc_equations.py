
def eom_ee_epcc1s1_sigma_slow(xx):

    T1, T2, S1, U11 = amps

    sigS = numpy.zeros(T1.shape)
    sigD = numpy.zeros(T2.shape)
    sig1 = numpy.zeros(S1.shape)
    sigS1 = numpy.zeros(U11.shape)

    '''
    EOM-CCSD sigS part
    
     + 1.0\sum_{I}G_{I}R11_{Iai}
     - 1.0\sum_{j}f_{ji}RS_{aj}
     + 1.0\sum_{b}f_{ab}RS_{bi}
     + 1.0\sum_{I}g_{Iai}R1_{I}
     - 1.0\sum_{jb}f_{jb}RD_{abji}
     - 1.0\sum_{Ij}g_{Iji}R11_{Iaj}
     + 1.0\sum_{Ib}g_{Iab}R11_{Ibi}
     - 1.0\sum_{jb}I_{jaib}RS_{bj}
     + 0.5\sum_{jkb}I_{jkib}RD_{abkj}
     - 0.5\sum_{jbc}I_{jabc}RD_{cbji}
     - 1.0\sum_{jb}f_{jb}T1_{aj}RS_{bi}
     - 1.0\sum_{jb}f_{jb}T1_{bi}RS_{aj}
     - 1.0\sum_{Ij}g_{Iji}S1_{I}RS_{aj}
     - 1.0\sum_{Ij}g_{Iji}T1_{aj}R1_{I}
     + 1.0\sum_{Ib}g_{Iab}S1_{I}RS_{bi}
     + 1.0\sum_{Ib}g_{Iab}T1_{bi}R1_{I}
     - 1.0\sum_{Ijb}g_{Ijb}S1_{I}RD_{abji}
     - 1.0\sum_{Ijb}g_{Ijb}T1_{aj}R11_{Ibi}
     - 1.0\sum_{Ijb}g_{Ijb}T1_{bi}R11_{Iaj}
     + 1.0\sum_{Ijb}g_{Ijb}T1_{bj}R11_{Iai}
     + 1.0\sum_{Ijb}g_{Ijb}U11_{Iai}RS_{bj}
     - 1.0\sum_{Ijb}g_{Ijb}U11_{Iaj}RS_{bi}
     - 1.0\sum_{Ijb}g_{Ijb}U11_{Ibi}RS_{aj}
     - 1.0\sum_{Ijb}g_{Ijb}T2_{abji}R1_{I}
     - 1.0\sum_{jkb}I_{jkib}T1_{aj}RS_{bk}
     + 1.0\sum_{jkb}I_{jkib}T1_{bj}RS_{ak}
     + 1.0\sum_{jbc}I_{jabc}T1_{ci}RS_{bj}
     - 1.0\sum_{jbc}I_{jabc}T1_{cj}RS_{bi}
     - 0.5\sum_{jkbc}I_{jkbc}T1_{aj}RD_{cbki}
     - 0.5\sum_{jkbc}I_{jkbc}T1_{ci}RD_{abkj}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{cj}RD_{abki}
     + 1.0\sum_{jkbc}I_{jkbc}T2_{acji}RS_{bk}
     + 0.5\sum_{jkbc}I_{jkbc}T2_{ackj}RS_{bi}
     + 0.5\sum_{jkbc}I_{jkbc}T2_{cbji}RS_{ak}
     - 1.0\sum_{Ijb}g_{Ijb}T1_{aj}S1_{I}RS_{bi}
     - 1.0\sum_{Ijb}g_{Ijb}T1_{bi}S1_{I}RS_{aj}
     - 1.0\sum_{Ijb}g_{Ijb}T1_{bi}T1_{aj}R1_{I}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{aj}T1_{ck}RS_{bi}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{ci}T1_{aj}RS_{bk}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{ci}T1_{bj}RS_{ak}
    '''
    SigS += -1.0*einsum('ji,aj->ai', f, RS)
    SigS += 1.0*einsum('ab,bi->ai', f, RS)
    SigS += -1.0*einsum('jaib,bj->ai', I, RS)

    SigS += -1.0*einsum('jb,aj,bi->ai', f, T1, RS)
    SigS += -1.0*einsum('jb,bi,aj->ai', f, T1, RS)

    SigS += -1.0*einsum('Iji,I,aj->ai', g, S1, RS)
    SigS += 1.0*einsum('Iab,I,bi->ai', g, S1, RS)

    SigS += 1.0*einsum('Ijb,Iai,bj->ai', g, U11, RS)
    SigS += -1.0*einsum('Ijb,Iaj,bi->ai', g, U11, RS)
    SigS += -1.0*einsum('Ijb,Ibi,aj->ai', g, U11, RS)
    SigS += -1.0*einsum('jkib,aj,bk->ai', I, T1, RS)
    SigS += 1.0*einsum('jkib,bj,ak->ai', I, T1, RS)
    SigS += 1.0*einsum('jabc,ci,bj->ai', I, T1, RS)
    SigS += -1.0*einsum('jabc,cj,bi->ai', I, T1, RS)
    SigS += 1.0*einsum('jkbc,acji,bk->ai', I, T2, RS)
    SigS += 0.5*einsum('jkbc,ackj,bi->ai', I, T2, RS)
    SigS += 0.5*einsum('jkbc,cbji,ak->ai', I, T2, RS)
    SigS += -1.0*einsum('Ijb,aj,I,bi->ai', g, T1, S1, RS)
    SigS += -1.0*einsum('Ijb,bi,I,aj->ai', g, T1, S1, RS)
    SigS += -1.0*einsum('jkbc,aj,ck,bi->ai', I, T1, T1, RS)
    SigS += 1.0*einsum('jkbc,ci,aj,bk->ai', I, T1, T1, RS)
    SigS += -1.0*einsum('jkbc,ci,bj,ak->ai', I, T1, T1, RS)

    SigS += 1.0*einsum('I,Iai->ai', G, R11)
    SigS += 1.0*einsum('Iai,I->ai', g, R1)
    SigS += -1.0*einsum('jb,abji->ai', f, RD)
    SigS += -1.0*einsum('Iji,Iaj->ai', g, R11)
    SigS += 1.0*einsum('Iab,Ibi->ai', g, R11)
    SigS += 0.5*einsum('jkib,abkj->ai', I, RD)
    SigS += -0.5*einsum('jabc,cbji->ai', I, RD)
    SigS += -1.0*einsum('Iji,aj,I->ai', g, T1, R1)
    SigS += 1.0*einsum('Iab,bi,I->ai', g, T1, R1)
    SigS += -1.0*einsum('Ijb,I,abji->ai', g, S1, RD)
    SigS += -1.0*einsum('Ijb,aj,Ibi->ai', g, T1, R11)
    SigS += -1.0*einsum('Ijb,bi,Iaj->ai', g, T1, R11)
    SigS += 1.0*einsum('Ijb,bj,Iai->ai', g, T1, R11)
    SigS += -1.0*einsum('Ijb,abji,I->ai', g, T2, R1)
    SigS += -0.5*einsum('jkbc,aj,cbki->ai', I, T1, RD)
    SigS += -0.5*einsum('jkbc,ci,abkj->ai', I, T1, RD)
    SigS += 1.0*einsum('jkbc,cj,abki->ai', I, T1, RD)
    SigS += -1.0*einsum('Ijb,bi,aj,I->ai', g, T1, T1, R1)

    '''
    EOM-CCSD sigD part
    
     + 1.0f_{ai}RS_{bj}
     - 1.0f_{aj}RS_{bi}
     - 1.0f_{bi}RS_{aj}
     + 1.0f_{bj}RS_{ai}
     + 1.0\sum_{k}f_{ki}RD_{bakj}
     - 1.0\sum_{k}f_{kj}RD_{baki}
     + 1.0\sum_{c}f_{ac}RD_{bcji}
     - 1.0\sum_{c}f_{bc}RD_{acji}
     + 1.0\sum_{I}g_{Iai}R11_{Ibj}
     - 1.0\sum_{I}g_{Iaj}R11_{Ibi}
     - 1.0\sum_{I}g_{Ibi}R11_{Iaj}
     + 1.0\sum_{I}g_{Ibj}R11_{Iai}
     - 1.0\sum_{k}I_{kaji}RS_{bk}
     + 1.0\sum_{k}I_{kbji}RS_{ak}
     - 1.0\sum_{c}I_{baic}RS_{cj}
     + 1.0\sum_{c}I_{bajc}RS_{ci}
     - 0.5\sum_{kl}I_{klji}RD_{balk}
     + 1.0\sum_{kc}I_{kaic}RD_{bckj}
     - 1.0\sum_{kc}I_{kajc}RD_{bcki}
     - 1.0\sum_{kc}I_{kbic}RD_{ackj}
     + 1.0\sum_{kc}I_{kbjc}RD_{acki}
     - 0.5\sum_{cd}I_{bacd}RD_{dcji}
     + 1.0\sum_{I}G_{I}U11_{Iai}RS_{bj}
     - 1.0\sum_{I}G_{I}U11_{Iaj}RS_{bi}
     - 1.0\sum_{I}G_{I}U11_{Ibi}RS_{aj}
     + 1.0\sum_{I}G_{I}U11_{Ibj}RS_{ai}
     - 1.0\sum_{k}f_{ki}T1_{ak}RS_{bj}
     + 1.0\sum_{k}f_{ki}T1_{bk}RS_{aj}
     + 1.0\sum_{k}f_{kj}T1_{ak}RS_{bi}
     - 1.0\sum_{k}f_{kj}T1_{bk}RS_{ai}
     + 1.0\sum_{c}f_{ac}T1_{ci}RS_{bj}
     - 1.0\sum_{c}f_{ac}T1_{cj}RS_{bi}
     - 1.0\sum_{c}f_{bc}T1_{ci}RS_{aj}
     + 1.0\sum_{c}f_{bc}T1_{cj}RS_{ai}
     + 1.0\sum_{I}g_{Iai}S1_{I}RS_{bj}
     - 1.0\sum_{I}g_{Iaj}S1_{I}RS_{bi}
     - 1.0\sum_{I}g_{Ibi}S1_{I}RS_{aj}
     + 1.0\sum_{I}g_{Ibj}S1_{I}RS_{ai}
     - 1.0\sum_{kc}f_{kc}T1_{ak}RD_{bcji}
     + 1.0\sum_{kc}f_{kc}T1_{bk}RD_{acji}
     + 1.0\sum_{kc}f_{kc}T1_{ci}RD_{bakj}
     - 1.0\sum_{kc}f_{kc}T1_{cj}RD_{baki}
     + 1.0\sum_{kc}f_{kc}T2_{acji}RS_{bk}
     - 1.0\sum_{kc}f_{kc}T2_{acki}RS_{bj}
     + 1.0\sum_{kc}f_{kc}T2_{ackj}RS_{bi}
     - 1.0\sum_{kc}f_{kc}T2_{baki}RS_{cj}
     + 1.0\sum_{kc}f_{kc}T2_{bakj}RS_{ci}
     - 1.0\sum_{kc}f_{kc}T2_{bcji}RS_{ak}
     + 1.0\sum_{kc}f_{kc}T2_{bcki}RS_{aj}
     - 1.0\sum_{kc}f_{kc}T2_{bckj}RS_{ai}
     + 1.0\sum_{Ik}g_{Iki}S1_{I}RD_{bakj}
     - 1.0\sum_{Ik}g_{Iki}T1_{ak}R11_{Ibj}
     + 1.0\sum_{Ik}g_{Iki}T1_{bk}R11_{Iaj}
     + 1.0\sum_{Ik}g_{Iki}U11_{Iaj}RS_{bk}
     - 1.0\sum_{Ik}g_{Iki}U11_{Iak}RS_{bj}
     - 1.0\sum_{Ik}g_{Iki}U11_{Ibj}RS_{ak}
     + 1.0\sum_{Ik}g_{Iki}U11_{Ibk}RS_{aj}
     + 1.0\sum_{Ik}g_{Iki}T2_{bakj}R1_{I}
     - 1.0\sum_{Ik}g_{Ikj}S1_{I}RD_{baki}
     + 1.0\sum_{Ik}g_{Ikj}T1_{ak}R11_{Ibi}
     - 1.0\sum_{Ik}g_{Ikj}T1_{bk}R11_{Iai}
     - 1.0\sum_{Ik}g_{Ikj}U11_{Iai}RS_{bk}
     + 1.0\sum_{Ik}g_{Ikj}U11_{Iak}RS_{bi}
     + 1.0\sum_{Ik}g_{Ikj}U11_{Ibi}RS_{ak}
     - 1.0\sum_{Ik}g_{Ikj}U11_{Ibk}RS_{ai}
     - 1.0\sum_{Ik}g_{Ikj}T2_{baki}R1_{I}
     + 1.0\sum_{Ic}g_{Iac}S1_{I}RD_{bcji}
     + 1.0\sum_{Ic}g_{Iac}T1_{ci}R11_{Ibj}
     - 1.0\sum_{Ic}g_{Iac}T1_{cj}R11_{Ibi}
     - 1.0\sum_{Ic}g_{Iac}U11_{Ibi}RS_{cj}
     + 1.0\sum_{Ic}g_{Iac}U11_{Ibj}RS_{ci}
     + 1.0\sum_{Ic}g_{Iac}U11_{Ici}RS_{bj}
     - 1.0\sum_{Ic}g_{Iac}U11_{Icj}RS_{bi}
     + 1.0\sum_{Ic}g_{Iac}T2_{bcji}R1_{I}
     - 1.0\sum_{Ic}g_{Ibc}S1_{I}RD_{acji}
     - 1.0\sum_{Ic}g_{Ibc}T1_{ci}R11_{Iaj}
     + 1.0\sum_{Ic}g_{Ibc}T1_{cj}R11_{Iai}
     + 1.0\sum_{Ic}g_{Ibc}U11_{Iai}RS_{cj}
     - 1.0\sum_{Ic}g_{Ibc}U11_{Iaj}RS_{ci}
     - 1.0\sum_{Ic}g_{Ibc}U11_{Ici}RS_{aj}
     + 1.0\sum_{Ic}g_{Ibc}U11_{Icj}RS_{ai}
     - 1.0\sum_{Ic}g_{Ibc}T2_{acji}R1_{I}
     - 1.0\sum_{kl}I_{klji}T1_{ak}RS_{bl}
     + 1.0\sum_{kl}I_{klji}T1_{bk}RS_{al}
     + 1.0\sum_{kc}I_{kaic}T1_{bk}RS_{cj}
     + 1.0\sum_{kc}I_{kaic}T1_{cj}RS_{bk}
     - 1.0\sum_{kc}I_{kaic}T1_{ck}RS_{bj}
     - 1.0\sum_{kc}I_{kajc}T1_{bk}RS_{ci}
     - 1.0\sum_{kc}I_{kajc}T1_{ci}RS_{bk}
     + 1.0\sum_{kc}I_{kajc}T1_{ck}RS_{bi}
     - 1.0\sum_{kc}I_{kbic}T1_{ak}RS_{cj}
     - 1.0\sum_{kc}I_{kbic}T1_{cj}RS_{ak}
     + 1.0\sum_{kc}I_{kbic}T1_{ck}RS_{aj}
     + 1.0\sum_{kc}I_{kbjc}T1_{ak}RS_{ci}
     + 1.0\sum_{kc}I_{kbjc}T1_{ci}RS_{ak}
     - 1.0\sum_{kc}I_{kbjc}T1_{ck}RS_{ai}
     + 1.0\sum_{cd}I_{bacd}T1_{di}RS_{cj}
     - 1.0\sum_{cd}I_{bacd}T1_{dj}RS_{ci}
     - 1.0\sum_{Ikc}g_{Ikc}U11_{Iai}RD_{bckj}
     + 1.0\sum_{Ikc}g_{Ikc}U11_{Iaj}RD_{bcki}
     - 1.0\sum_{Ikc}g_{Ikc}U11_{Iak}RD_{bcji}
     + 1.0\sum_{Ikc}g_{Ikc}U11_{Ibi}RD_{ackj}
     - 1.0\sum_{Ikc}g_{Ikc}U11_{Ibj}RD_{acki}
     + 1.0\sum_{Ikc}g_{Ikc}U11_{Ibk}RD_{acji}
     + 1.0\sum_{Ikc}g_{Ikc}U11_{Ici}RD_{bakj}
     - 1.0\sum_{Ikc}g_{Ikc}U11_{Icj}RD_{baki}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{acji}R11_{Ibk}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{acki}R11_{Ibj}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{ackj}R11_{Ibi}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{baki}R11_{Icj}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{bakj}R11_{Ici}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{bcji}R11_{Iak}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{bcki}R11_{Iaj}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{bckj}R11_{Iai}
     + 1.0\sum_{klc}I_{klic}T1_{ak}RD_{bclj}
     - 1.0\sum_{klc}I_{klic}T1_{bk}RD_{aclj}
     + 0.5\sum_{klc}I_{klic}T1_{cj}RD_{balk}
     - 1.0\sum_{klc}I_{klic}T1_{ck}RD_{balj}
     + 1.0\sum_{klc}I_{klic}T2_{ackj}RS_{bl}
     + 0.5\sum_{klc}I_{klic}T2_{aclk}RS_{bj}
     + 1.0\sum_{klc}I_{klic}T2_{bakj}RS_{cl}
     + 0.5\sum_{klc}I_{klic}T2_{balk}RS_{cj}
     - 1.0\sum_{klc}I_{klic}T2_{bckj}RS_{al}
     - 0.5\sum_{klc}I_{klic}T2_{bclk}RS_{aj}
     - 1.0\sum_{klc}I_{kljc}T1_{ak}RD_{bcli}
     + 1.0\sum_{klc}I_{kljc}T1_{bk}RD_{acli}
     - 0.5\sum_{klc}I_{kljc}T1_{ci}RD_{balk}
     + 1.0\sum_{klc}I_{kljc}T1_{ck}RD_{bali}
     - 1.0\sum_{klc}I_{kljc}T2_{acki}RS_{bl}
     - 0.5\sum_{klc}I_{kljc}T2_{aclk}RS_{bi}
     - 1.0\sum_{klc}I_{kljc}T2_{baki}RS_{cl}
     - 0.5\sum_{klc}I_{kljc}T2_{balk}RS_{ci}
     + 1.0\sum_{klc}I_{kljc}T2_{bcki}RS_{al}
     + 0.5\sum_{klc}I_{kljc}T2_{bclk}RS_{ai}
     + 0.5\sum_{kcd}I_{kacd}T1_{bk}RD_{dcji}
     - 1.0\sum_{kcd}I_{kacd}T1_{di}RD_{bckj}
     + 1.0\sum_{kcd}I_{kacd}T1_{dj}RD_{bcki}
     - 1.0\sum_{kcd}I_{kacd}T1_{dk}RD_{bcji}
     + 1.0\sum_{kcd}I_{kacd}T2_{bdji}RS_{ck}
     - 1.0\sum_{kcd}I_{kacd}T2_{bdki}RS_{cj}
     + 1.0\sum_{kcd}I_{kacd}T2_{bdkj}RS_{ci}
     + 0.5\sum_{kcd}I_{kacd}T2_{dcji}RS_{bk}
     - 0.5\sum_{kcd}I_{kacd}T2_{dcki}RS_{bj}
     + 0.5\sum_{kcd}I_{kacd}T2_{dckj}RS_{bi}
     - 0.5\sum_{kcd}I_{kbcd}T1_{ak}RD_{dcji}
     + 1.0\sum_{kcd}I_{kbcd}T1_{di}RD_{ackj}
     - 1.0\sum_{kcd}I_{kbcd}T1_{dj}RD_{acki}
     + 1.0\sum_{kcd}I_{kbcd}T1_{dk}RD_{acji}
     - 1.0\sum_{kcd}I_{kbcd}T2_{adji}RS_{ck}
     + 1.0\sum_{kcd}I_{kbcd}T2_{adki}RS_{cj}
     - 1.0\sum_{kcd}I_{kbcd}T2_{adkj}RS_{ci}
     - 0.5\sum_{kcd}I_{kbcd}T2_{dcji}RS_{ak}
     + 0.5\sum_{kcd}I_{kbcd}T2_{dcki}RS_{aj}
     - 0.5\sum_{kcd}I_{kbcd}T2_{dckj}RS_{ai}
     + 0.5\sum_{klcd}I_{klcd}T2_{adji}RD_{bclk}
     - 1.0\sum_{klcd}I_{klcd}T2_{adki}RD_{bclj}
     + 1.0\sum_{klcd}I_{klcd}T2_{adkj}RD_{bcli}
     + 0.5\sum_{klcd}I_{klcd}T2_{adlk}RD_{bcji}
     - 0.5\sum_{klcd}I_{klcd}T2_{baki}RD_{dclj}
     + 0.5\sum_{klcd}I_{klcd}T2_{bakj}RD_{dcli}
     + 0.25\sum_{klcd}I_{klcd}T2_{balk}RD_{dcji}
     - 0.5\sum_{klcd}I_{klcd}T2_{bdji}RD_{aclk}
     + 1.0\sum_{klcd}I_{klcd}T2_{bdki}RD_{aclj}
     - 1.0\sum_{klcd}I_{klcd}T2_{bdkj}RD_{acli}
     - 0.5\sum_{klcd}I_{klcd}T2_{bdlk}RD_{acji}
     + 0.25\sum_{klcd}I_{klcd}T2_{dcji}RD_{balk}
     - 0.5\sum_{klcd}I_{klcd}T2_{dcki}RD_{balj}
     + 0.5\sum_{klcd}I_{klcd}T2_{dckj}RD_{bali}
     - 1.0\sum_{kc}f_{kc}T1_{ci}T1_{ak}RS_{bj}
     + 1.0\sum_{kc}f_{kc}T1_{ci}T1_{bk}RS_{aj}
     + 1.0\sum_{kc}f_{kc}T1_{cj}T1_{ak}RS_{bi}
     - 1.0\sum_{kc}f_{kc}T1_{cj}T1_{bk}RS_{ai}
     - 1.0\sum_{Ik}g_{Iki}T1_{ak}S1_{I}RS_{bj}
     + 1.0\sum_{Ik}g_{Iki}T1_{bk}S1_{I}RS_{aj}
     + 1.0\sum_{Ik}g_{Ikj}T1_{ak}S1_{I}RS_{bi}
     - 1.0\sum_{Ik}g_{Ikj}T1_{bk}S1_{I}RS_{ai}
     + 1.0\sum_{Ic}g_{Iac}T1_{ci}S1_{I}RS_{bj}
     - 1.0\sum_{Ic}g_{Iac}T1_{cj}S1_{I}RS_{bi}
     - 1.0\sum_{Ic}g_{Ibc}T1_{ci}S1_{I}RS_{aj}
     + 1.0\sum_{Ic}g_{Ibc}T1_{cj}S1_{I}RS_{ai}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ak}S1_{I}RD_{bcji}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ak}U11_{Ibi}RS_{cj}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ak}U11_{Ibj}RS_{ci}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ak}U11_{Ici}RS_{bj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ak}U11_{Icj}RS_{bi}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ak}T2_{bcji}R1_{I}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{bk}S1_{I}RD_{acji}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{bk}U11_{Iai}RS_{cj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{bk}U11_{Iaj}RS_{ci}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{bk}U11_{Ici}RS_{aj}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{bk}U11_{Icj}RS_{ai}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{bk}T2_{acji}R1_{I}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ci}S1_{I}RD_{bakj}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ci}T1_{ak}R11_{Ibj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ci}T1_{bk}R11_{Iaj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ci}U11_{Iaj}RS_{bk}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ci}U11_{Iak}RS_{bj}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ci}U11_{Ibj}RS_{ak}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ci}U11_{Ibk}RS_{aj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ci}T2_{bakj}R1_{I}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{cj}S1_{I}RD_{baki}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{cj}T1_{ak}R11_{Ibi}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{cj}T1_{bk}R11_{Iai}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{cj}U11_{Iai}RS_{bk}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{cj}U11_{Iak}RS_{bi}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{cj}U11_{Ibi}RS_{ak}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{cj}U11_{Ibk}RS_{ai}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{cj}T2_{baki}R1_{I}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ck}U11_{Iai}RS_{bj}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ck}U11_{Iaj}RS_{bi}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ck}U11_{Ibi}RS_{aj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ck}U11_{Ibj}RS_{ai}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{acji}S1_{I}RS_{bk}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{acki}S1_{I}RS_{bj}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{ackj}S1_{I}RS_{bi}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{baki}S1_{I}RS_{cj}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{bakj}S1_{I}RS_{ci}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{bcji}S1_{I}RS_{ak}
     + 1.0\sum_{Ikc}g_{Ikc}T2_{bcki}S1_{I}RS_{aj}
     - 1.0\sum_{Ikc}g_{Ikc}T2_{bckj}S1_{I}RS_{ai}
     - 1.0\sum_{klc}I_{klic}T1_{ak}T1_{cl}RS_{bj}
     - 1.0\sum_{klc}I_{klic}T1_{bk}T1_{al}RS_{cj}
     + 1.0\sum_{klc}I_{klic}T1_{bk}T1_{cl}RS_{aj}
     + 1.0\sum_{klc}I_{klic}T1_{cj}T1_{ak}RS_{bl}
     - 1.0\sum_{klc}I_{klic}T1_{cj}T1_{bk}RS_{al}
     + 1.0\sum_{klc}I_{kljc}T1_{ak}T1_{cl}RS_{bi}
     + 1.0\sum_{klc}I_{kljc}T1_{bk}T1_{al}RS_{ci}
     - 1.0\sum_{klc}I_{kljc}T1_{bk}T1_{cl}RS_{ai}
     - 1.0\sum_{klc}I_{kljc}T1_{ci}T1_{ak}RS_{bl}
     + 1.0\sum_{klc}I_{kljc}T1_{ci}T1_{bk}RS_{al}
     - 1.0\sum_{kcd}I_{kacd}T1_{di}T1_{bk}RS_{cj}
     - 1.0\sum_{kcd}I_{kacd}T1_{di}T1_{cj}RS_{bk}
     + 1.0\sum_{kcd}I_{kacd}T1_{di}T1_{ck}RS_{bj}
     + 1.0\sum_{kcd}I_{kacd}T1_{dj}T1_{bk}RS_{ci}
     - 1.0\sum_{kcd}I_{kacd}T1_{dj}T1_{ck}RS_{bi}
     + 1.0\sum_{kcd}I_{kbcd}T1_{di}T1_{ak}RS_{cj}
     + 1.0\sum_{kcd}I_{kbcd}T1_{di}T1_{cj}RS_{ak}
     - 1.0\sum_{kcd}I_{kbcd}T1_{di}T1_{ck}RS_{aj}
     - 1.0\sum_{kcd}I_{kbcd}T1_{dj}T1_{ak}RS_{ci}
     + 1.0\sum_{kcd}I_{kbcd}T1_{dj}T1_{ck}RS_{ai}
     - 1.0\sum_{klcd}I_{klcd}T1_{ak}T1_{dl}RD_{bcji}
     + 1.0\sum_{klcd}I_{klcd}T1_{ak}T2_{bdji}RS_{cl}
     - 1.0\sum_{klcd}I_{klcd}T1_{ak}T2_{bdli}RS_{cj}
     + 1.0\sum_{klcd}I_{klcd}T1_{ak}T2_{bdlj}RS_{ci}
     + 0.5\sum_{klcd}I_{klcd}T1_{ak}T2_{dcji}RS_{bl}
     - 0.5\sum_{klcd}I_{klcd}T1_{ak}T2_{dcli}RS_{bj}
     + 0.5\sum_{klcd}I_{klcd}T1_{ak}T2_{dclj}RS_{bi}
     - 0.5\sum_{klcd}I_{klcd}T1_{bk}T1_{al}RD_{dcji}
     + 1.0\sum_{klcd}I_{klcd}T1_{bk}T1_{dl}RD_{acji}
     - 1.0\sum_{klcd}I_{klcd}T1_{bk}T2_{adji}RS_{cl}
     + 1.0\sum_{klcd}I_{klcd}T1_{bk}T2_{adli}RS_{cj}
     - 1.0\sum_{klcd}I_{klcd}T1_{bk}T2_{adlj}RS_{ci}
     - 0.5\sum_{klcd}I_{klcd}T1_{bk}T2_{dcji}RS_{al}
     + 0.5\sum_{klcd}I_{klcd}T1_{bk}T2_{dcli}RS_{aj}
     - 0.5\sum_{klcd}I_{klcd}T1_{bk}T2_{dclj}RS_{ai}
     - 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{ak}RD_{bclj}
     + 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{bk}RD_{aclj}
     - 0.5\sum_{klcd}I_{klcd}T1_{di}T1_{cj}RD_{balk}
     + 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{ck}RD_{balj}
     - 1.0\sum_{klcd}I_{klcd}T1_{di}T2_{ackj}RS_{bl}
     - 0.5\sum_{klcd}I_{klcd}T1_{di}T2_{aclk}RS_{bj}
     - 1.0\sum_{klcd}I_{klcd}T1_{di}T2_{bakj}RS_{cl}
     - 0.5\sum_{klcd}I_{klcd}T1_{di}T2_{balk}RS_{cj}
     + 1.0\sum_{klcd}I_{klcd}T1_{di}T2_{bckj}RS_{al}
     + 0.5\sum_{klcd}I_{klcd}T1_{di}T2_{bclk}RS_{aj}
     + 1.0\sum_{klcd}I_{klcd}T1_{dj}T1_{ak}RD_{bcli}
     - 1.0\sum_{klcd}I_{klcd}T1_{dj}T1_{bk}RD_{acli}
     - 1.0\sum_{klcd}I_{klcd}T1_{dj}T1_{ck}RD_{bali}
     + 1.0\sum_{klcd}I_{klcd}T1_{dj}T2_{acki}RS_{bl}
     + 0.5\sum_{klcd}I_{klcd}T1_{dj}T2_{aclk}RS_{bi}
     + 1.0\sum_{klcd}I_{klcd}T1_{dj}T2_{baki}RS_{cl}
     + 0.5\sum_{klcd}I_{klcd}T1_{dj}T2_{balk}RS_{ci}
     - 1.0\sum_{klcd}I_{klcd}T1_{dj}T2_{bcki}RS_{al}
     - 0.5\sum_{klcd}I_{klcd}T1_{dj}T2_{bclk}RS_{ai}
     - 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{acji}RS_{bl}
     + 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{acli}RS_{bj}
     - 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{aclj}RS_{bi}
     + 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{bali}RS_{cj}
     - 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{balj}RS_{ci}
     + 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{bcji}RS_{al}
     - 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{bcli}RS_{aj}
     + 1.0\sum_{klcd}I_{klcd}T1_{dk}T2_{bclj}RS_{ai}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{ci}T1_{ak}S1_{I}RS_{bj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{ci}T1_{bk}S1_{I}RS_{aj}
     + 1.0\sum_{Ikc}g_{Ikc}T1_{cj}T1_{ak}S1_{I}RS_{bi}
     - 1.0\sum_{Ikc}g_{Ikc}T1_{cj}T1_{bk}S1_{I}RS_{ai}
     + 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{ak}T1_{cl}RS_{bj}
     + 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{bk}T1_{al}RS_{cj}
     - 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{bk}T1_{cl}RS_{aj}
     - 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{cj}T1_{ak}RS_{bl}
     + 1.0\sum_{klcd}I_{klcd}T1_{di}T1_{cj}T1_{bk}RS_{al}
     - 1.0\sum_{klcd}I_{klcd}T1_{dj}T1_{ak}T1_{cl}RS_{bi}
     - 1.0\sum_{klcd}I_{klcd}T1_{dj}T1_{bk}T1_{al}RS_{ci}
     + 1.0\sum_{klcd}I_{klcd}T1_{dj}T1_{bk}T1_{cl}RS_{ai}
    
     einsum format=
    '''
    
    #SigD += 1.0*einsum('ai,bj->abij', f, RS)
    #SigD += -1.0*einsum('aj,bi->abij', f, RS)
    #SigD += -1.0*einsum('bi,aj->abij', f, RS)
    #SigD += 1.0*einsum('bj,ai->abij', f, RS)
    #SigD += 1.0*einsum('ki,bakj->abij', f, RD)
    #SigD += -1.0*einsum('ki,ak,bj->abij', f, T1, RS)
    #SigD += 1.0*einsum('ki,bk,aj->abij', f, T1, RS)
    #SigD += 1.0*einsum('kj,ak,bi->abij', f, T1, RS)
    #SigD += -1.0*einsum('kj,bk,ai->abij', f, T1, RS)
    #SigD += 1.0*einsum('ac,ci,bj->abij', f, T1, RS)
    #SigD += -1.0*einsum('ac,cj,bi->abij', f, T1, RS)
    #SigD += -1.0*einsum('bc,ci,aj->abij', f, T1, RS)
    #SigD += 1.0*einsum('bc,cj,ai->abij', f, T1, RS)
    SigD += -1.0*einsum('kj,baki->abij', f, RD)
    SigD += 1.0*einsum('ac,bcji->abij', f, RD)
    SigD += -1.0*einsum('bc,acji->abij', f, RD)
    SigD += 1.0*einsum('Iai,Ibj->abij', g, R11)
    SigD += -1.0*einsum('Iaj,Ibi->abij', g, R11)
    SigD += -1.0*einsum('Ibi,Iaj->abij', g, R11)
    SigD += 1.0*einsum('Ibj,Iai->abij', g, R11)
    SigD += -1.0*einsum('kaji,bk->abij', I, RS)
    SigD += 1.0*einsum('kbji,ak->abij', I, RS)
    SigD += -1.0*einsum('baic,cj->abij', I, RS)
    SigD += 1.0*einsum('bajc,ci->abij', I, RS)
    SigD += -0.5*einsum('klji,balk->abij', I, RD)
    SigD += 1.0*einsum('kaic,bckj->abij', I, RD)
    SigD += -1.0*einsum('kajc,bcki->abij', I, RD)
    SigD += -1.0*einsum('kbic,ackj->abij', I, RD)
    SigD += 1.0*einsum('kbjc,acki->abij', I, RD)
    SigD += -0.5*einsum('bacd,dcji->abij', I, RD)
    SigD += 1.0*einsum('I,Iai,bj->abij', G, U11, RS)
    SigD += -1.0*einsum('I,Iaj,bi->abij', G, U11, RS)
    SigD += -1.0*einsum('I,Ibi,aj->abij', G, U11, RS)
    SigD += 1.0*einsum('I,Ibj,ai->abij', G, U11, RS)
    SigD += 1.0*einsum('Iai,I,bj->abij', g, S1, RS)
    SigD += -1.0*einsum('Iaj,I,bi->abij', g, S1, RS)
    SigD += -1.0*einsum('Ibi,I,aj->abij', g, S1, RS)
    SigD += 1.0*einsum('Ibj,I,ai->abij', g, S1, RS)
    SigD += -1.0*einsum('kc,ak,bcji->abij', f, T1, RD)
    SigD += 1.0*einsum('kc,bk,acji->abij', f, T1, RD)
    SigD += 1.0*einsum('kc,ci,bakj->abij', f, T1, RD)
    SigD += -1.0*einsum('kc,cj,baki->abij', f, T1, RD)
    SigD += 1.0*einsum('kc,acji,bk->abij', f, T2, RS)
    SigD += -1.0*einsum('kc,acki,bj->abij', f, T2, RS)
    SigD += 1.0*einsum('kc,ackj,bi->abij', f, T2, RS)
    SigD += -1.0*einsum('kc,baki,cj->abij', f, T2, RS)
    SigD += 1.0*einsum('kc,bakj,ci->abij', f, T2, RS)
    SigD += -1.0*einsum('kc,bcji,ak->abij', f, T2, RS)
    SigD += 1.0*einsum('kc,bcki,aj->abij', f, T2, RS)
    SigD += -1.0*einsum('kc,bckj,ai->abij', f, T2, RS)
    SigD += 1.0*einsum('Iki,I,bakj->abij', g, S1, RD)
    SigD += -1.0*einsum('Iki,ak,Ibj->abij', g, T1, R11)
    SigD += 1.0*einsum('Iki,bk,Iaj->abij', g, T1, R11)
    SigD += 1.0*einsum('Iki,Iaj,bk->abij', g, U11, RS)
    SigD += -1.0*einsum('Iki,Iak,bj->abij', g, U11, RS)
    SigD += -1.0*einsum('Iki,Ibj,ak->abij', g, U11, RS)
    SigD += 1.0*einsum('Iki,Ibk,aj->abij', g, U11, RS)
    SigD += 1.0*einsum('Iki,bakj,I->abij', g, T2, R1)
    SigD += -1.0*einsum('Ikj,I,baki->abij', g, S1, RD)
    SigD += 1.0*einsum('Ikj,ak,Ibi->abij', g, T1, R11)
    SigD += -1.0*einsum('Ikj,bk,Iai->abij', g, T1, R11)
    SigD += -1.0*einsum('Ikj,Iai,bk->abij', g, U11, RS)
    SigD += 1.0*einsum('Ikj,Iak,bi->abij', g, U11, RS)
    SigD += 1.0*einsum('Ikj,Ibi,ak->abij', g, U11, RS)
    SigD += -1.0*einsum('Ikj,Ibk,ai->abij', g, U11, RS)
    SigD += -1.0*einsum('Ikj,baki,I->abij', g, T2, R1)
    SigD += 1.0*einsum('Iac,I,bcji->abij', g, S1, RD)
    SigD += 1.0*einsum('Iac,ci,Ibj->abij', g, T1, R11)
    SigD += -1.0*einsum('Iac,cj,Ibi->abij', g, T1, R11)
    SigD += -1.0*einsum('Iac,Ibi,cj->abij', g, U11, RS)
    SigD += 1.0*einsum('Iac,Ibj,ci->abij', g, U11, RS)
    SigD += 1.0*einsum('Iac,Ici,bj->abij', g, U11, RS)
    SigD += -1.0*einsum('Iac,Icj,bi->abij', g, U11, RS)
    SigD += 1.0*einsum('Iac,bcji,I->abij', g, T2, R1)
    SigD += -1.0*einsum('Ibc,I,acji->abij', g, S1, RD)
    SigD += -1.0*einsum('Ibc,ci,Iaj->abij', g, T1, R11)
    SigD += 1.0*einsum('Ibc,cj,Iai->abij', g, T1, R11)
    SigD += 1.0*einsum('Ibc,Iai,cj->abij', g, U11, RS)
    SigD += -1.0*einsum('Ibc,Iaj,ci->abij', g, U11, RS)
    SigD += -1.0*einsum('Ibc,Ici,aj->abij', g, U11, RS)
    SigD += 1.0*einsum('Ibc,Icj,ai->abij', g, U11, RS)
    SigD += -1.0*einsum('Ibc,acji,I->abij', g, T2, R1)
    SigD += -1.0*einsum('klji,ak,bl->abij', I, T1, RS)
    SigD += 1.0*einsum('klji,bk,al->abij', I, T1, RS)
    SigD += 1.0*einsum('kaic,bk,cj->abij', I, T1, RS)
    SigD += 1.0*einsum('kaic,cj,bk->abij', I, T1, RS)
    SigD += -1.0*einsum('kaic,ck,bj->abij', I, T1, RS)
    SigD += -1.0*einsum('kajc,bk,ci->abij', I, T1, RS)
    SigD += -1.0*einsum('kajc,ci,bk->abij', I, T1, RS)
    SigD += 1.0*einsum('kajc,ck,bi->abij', I, T1, RS)
    SigD += -1.0*einsum('kbic,ak,cj->abij', I, T1, RS)
    SigD += -1.0*einsum('kbic,cj,ak->abij', I, T1, RS)
    SigD += 1.0*einsum('kbic,ck,aj->abij', I, T1, RS)
    SigD += 1.0*einsum('kbjc,ak,ci->abij', I, T1, RS)
    SigD += 1.0*einsum('kbjc,ci,ak->abij', I, T1, RS)
    SigD += -1.0*einsum('kbjc,ck,ai->abij', I, T1, RS)
    SigD += 1.0*einsum('bacd,di,cj->abij', I, T1, RS)
    SigD += -1.0*einsum('bacd,dj,ci->abij', I, T1, RS)
    SigD += -1.0*einsum('Ikc,Iai,bckj->abij', g, U11, RD)
    SigD += 1.0*einsum('Ikc,Iaj,bcki->abij', g, U11, RD)
    SigD += -1.0*einsum('Ikc,Iak,bcji->abij', g, U11, RD)
    SigD += 1.0*einsum('Ikc,Ibi,ackj->abij', g, U11, RD)
    SigD += -1.0*einsum('Ikc,Ibj,acki->abij', g, U11, RD)
    SigD += 1.0*einsum('Ikc,Ibk,acji->abij', g, U11, RD)
    SigD += 1.0*einsum('Ikc,Ici,bakj->abij', g, U11, RD)
    SigD += -1.0*einsum('Ikc,Icj,baki->abij', g, U11, RD)
    SigD += 1.0*einsum('Ikc,acji,Ibk->abij', g, T2, R11)
    SigD += -1.0*einsum('Ikc,acki,Ibj->abij', g, T2, R11)
    SigD += 1.0*einsum('Ikc,ackj,Ibi->abij', g, T2, R11)
    SigD += -1.0*einsum('Ikc,baki,Icj->abij', g, T2, R11)
    SigD += 1.0*einsum('Ikc,bakj,Ici->abij', g, T2, R11)
    SigD += -1.0*einsum('Ikc,bcji,Iak->abij', g, T2, R11)
    SigD += 1.0*einsum('Ikc,bcki,Iaj->abij', g, T2, R11)
    SigD += -1.0*einsum('Ikc,bckj,Iai->abij', g, T2, R11)
    SigD += 1.0*einsum('klic,ak,bclj->abij', I, T1, RD)
    SigD += -1.0*einsum('klic,bk,aclj->abij', I, T1, RD)
    SigD += 0.5*einsum('klic,cj,balk->abij', I, T1, RD)
    SigD += -1.0*einsum('klic,ck,balj->abij', I, T1, RD)
    SigD += 1.0*einsum('klic,ackj,bl->abij', I, T2, RS)
    SigD += 0.5*einsum('klic,aclk,bj->abij', I, T2, RS)
    SigD += 1.0*einsum('klic,bakj,cl->abij', I, T2, RS)
    SigD += 0.5*einsum('klic,balk,cj->abij', I, T2, RS)
    SigD += -1.0*einsum('klic,bckj,al->abij', I, T2, RS)
    SigD += -0.5*einsum('klic,bclk,aj->abij', I, T2, RS)
    SigD += -1.0*einsum('kljc,ak,bcli->abij', I, T1, RD)
    SigD += 1.0*einsum('kljc,bk,acli->abij', I, T1, RD)
    SigD += -0.5*einsum('kljc,ci,balk->abij', I, T1, RD)
    SigD += 1.0*einsum('kljc,ck,bali->abij', I, T1, RD)
    SigD += -1.0*einsum('kljc,acki,bl->abij', I, T2, RS)
    SigD += -0.5*einsum('kljc,aclk,bi->abij', I, T2, RS)
    SigD += -1.0*einsum('kljc,baki,cl->abij', I, T2, RS)
    SigD += -0.5*einsum('kljc,balk,ci->abij', I, T2, RS)
    SigD += 1.0*einsum('kljc,bcki,al->abij', I, T2, RS)
    SigD += 0.5*einsum('kljc,bclk,ai->abij', I, T2, RS)
    SigD += 0.5*einsum('kacd,bk,dcji->abij', I, T1, RD)
    SigD += -1.0*einsum('kacd,di,bckj->abij', I, T1, RD)
    SigD += 1.0*einsum('kacd,dj,bcki->abij', I, T1, RD)
    SigD += -1.0*einsum('kacd,dk,bcji->abij', I, T1, RD)
    SigD += 1.0*einsum('kacd,bdji,ck->abij', I, T2, RS)
    SigD += -1.0*einsum('kacd,bdki,cj->abij', I, T2, RS)
    SigD += 1.0*einsum('kacd,bdkj,ci->abij', I, T2, RS)
    SigD += 0.5*einsum('kacd,dcji,bk->abij', I, T2, RS)
    SigD += -0.5*einsum('kacd,dcki,bj->abij', I, T2, RS)
    SigD += 0.5*einsum('kacd,dckj,bi->abij', I, T2, RS)
    SigD += -0.5*einsum('kbcd,ak,dcji->abij', I, T1, RD)
    SigD += 1.0*einsum('kbcd,di,ackj->abij', I, T1, RD)
    SigD += -1.0*einsum('kbcd,dj,acki->abij', I, T1, RD)
    SigD += 1.0*einsum('kbcd,dk,acji->abij', I, T1, RD)
    SigD += -1.0*einsum('kbcd,adji,ck->abij', I, T2, RS)
    SigD += 1.0*einsum('kbcd,adki,cj->abij', I, T2, RS)
    SigD += -1.0*einsum('kbcd,adkj,ci->abij', I, T2, RS)
    SigD += -0.5*einsum('kbcd,dcji,ak->abij', I, T2, RS)
    SigD += 0.5*einsum('kbcd,dcki,aj->abij', I, T2, RS)
    SigD += -0.5*einsum('kbcd,dckj,ai->abij', I, T2, RS)
    SigD += 0.5*einsum('klcd,adji,bclk->abij', I, T2, RD)
    SigD += -1.0*einsum('klcd,adki,bclj->abij', I, T2, RD)
    SigD += 1.0*einsum('klcd,adkj,bcli->abij', I, T2, RD)
    SigD += 0.5*einsum('klcd,adlk,bcji->abij', I, T2, RD)
    SigD += -0.5*einsum('klcd,baki,dclj->abij', I, T2, RD)
    SigD += 0.5*einsum('klcd,bakj,dcli->abij', I, T2, RD)
    SigD += 0.25*einsum('klcd,balk,dcji->abij', I, T2, RD)
    SigD += -0.5*einsum('klcd,bdji,aclk->abij', I, T2, RD)
    SigD += 1.0*einsum('klcd,bdki,aclj->abij', I, T2, RD)
    SigD += -1.0*einsum('klcd,bdkj,acli->abij', I, T2, RD)
    SigD += -0.5*einsum('klcd,bdlk,acji->abij', I, T2, RD)
    SigD += 0.25*einsum('klcd,dcji,balk->abij', I, T2, RD)
    SigD += -0.5*einsum('klcd,dcki,balj->abij', I, T2, RD)
    SigD += 0.5*einsum('klcd,dckj,bali->abij', I, T2, RD)
    SigD += -1.0*einsum('kc,ci,ak,bj->abij', f, T1, T1, RS)
    SigD += 1.0*einsum('kc,ci,bk,aj->abij', f, T1, T1, RS)
    SigD += 1.0*einsum('kc,cj,ak,bi->abij', f, T1, T1, RS)
    SigD += -1.0*einsum('kc,cj,bk,ai->abij', f, T1, T1, RS)
    SigD += -1.0*einsum('Iki,ak,I,bj->abij', g, T1, S1, RS)
    SigD += 1.0*einsum('Iki,bk,I,aj->abij', g, T1, S1, RS)
    SigD += 1.0*einsum('Ikj,ak,I,bi->abij', g, T1, S1, RS)
    SigD += -1.0*einsum('Ikj,bk,I,ai->abij', g, T1, S1, RS)
    SigD += 1.0*einsum('Iac,ci,I,bj->abij', g, T1, S1, RS)
    SigD += -1.0*einsum('Iac,cj,I,bi->abij', g, T1, S1, RS)
    SigD += -1.0*einsum('Ibc,ci,I,aj->abij', g, T1, S1, RS)
    SigD += 1.0*einsum('Ibc,cj,I,ai->abij', g, T1, S1, RS)
    SigD += -1.0*einsum('Ikc,ak,I,bcji->abij', g, T1, S1, RD)
    SigD += 1.0*einsum('Ikc,ak,Ibi,cj->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ak,Ibj,ci->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ak,Ici,bj->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,ak,Icj,bi->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ak,bcji,I->abij', g, T1, T2, R1)
    SigD += 1.0*einsum('Ikc,bk,I,acji->abij', g, T1, S1, RD)
    SigD += -1.0*einsum('Ikc,bk,Iai,cj->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,bk,Iaj,ci->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,bk,Ici,aj->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,bk,Icj,ai->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,bk,acji,I->abij', g, T1, T2, R1)
    SigD += 1.0*einsum('Ikc,ci,I,bakj->abij', g, T1, S1, RD)
    SigD += -1.0*einsum('Ikc,ci,ak,Ibj->abij', g, T1, T1, R11)
    SigD += 1.0*einsum('Ikc,ci,bk,Iaj->abij', g, T1, T1, R11)
    SigD += 1.0*einsum('Ikc,ci,Iaj,bk->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ci,Iak,bj->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ci,Ibj,ak->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,ci,Ibk,aj->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,ci,bakj,I->abij', g, T1, T2, R1)
    SigD += -1.0*einsum('Ikc,cj,I,baki->abij', g, T1, S1, RD)
    SigD += 1.0*einsum('Ikc,cj,ak,Ibi->abij', g, T1, T1, R11)
    SigD += -1.0*einsum('Ikc,cj,bk,Iai->abij', g, T1, T1, R11)
    SigD += -1.0*einsum('Ikc,cj,Iai,bk->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,cj,Iak,bi->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,cj,Ibi,ak->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,cj,Ibk,ai->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,cj,baki,I->abij', g, T1, T2, R1)
    SigD += 1.0*einsum('Ikc,ck,Iai,bj->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ck,Iaj,bi->abij', g, T1, U11, RS)
    SigD += -1.0*einsum('Ikc,ck,Ibi,aj->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,ck,Ibj,ai->abij', g, T1, U11, RS)
    SigD += 1.0*einsum('Ikc,acji,I,bk->abij', g, T2, S1, RS)
    SigD += -1.0*einsum('Ikc,acki,I,bj->abij', g, T2, S1, RS)
    SigD += 1.0*einsum('Ikc,ackj,I,bi->abij', g, T2, S1, RS)
    SigD += -1.0*einsum('Ikc,baki,I,cj->abij', g, T2, S1, RS)
    SigD += 1.0*einsum('Ikc,bakj,I,ci->abij', g, T2, S1, RS)
    SigD += -1.0*einsum('Ikc,bcji,I,ak->abij', g, T2, S1, RS)
    SigD += 1.0*einsum('Ikc,bcki,I,aj->abij', g, T2, S1, RS)
    SigD += -1.0*einsum('Ikc,bckj,I,ai->abij', g, T2, S1, RS)
    SigD += -1.0*einsum('klic,ak,cl,bj->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('klic,bk,al,cj->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('klic,bk,cl,aj->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('klic,cj,ak,bl->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('klic,cj,bk,al->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kljc,ak,cl,bi->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kljc,bk,al,ci->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kljc,bk,cl,ai->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kljc,ci,ak,bl->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kljc,ci,bk,al->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kacd,di,bk,cj->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kacd,di,cj,bk->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kacd,di,ck,bj->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kacd,dj,bk,ci->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kacd,dj,ck,bi->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kbcd,di,ak,cj->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kbcd,di,cj,ak->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kbcd,di,ck,aj->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('kbcd,dj,ak,ci->abij', I, T1, T1, RS)
    SigD += 1.0*einsum('kbcd,dj,ck,ai->abij', I, T1, T1, RS)
    SigD += -1.0*einsum('klcd,ak,dl,bcji->abij', I, T1, T1, RD)
    SigD += 1.0*einsum('klcd,ak,bdji,cl->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,ak,bdli,cj->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,ak,bdlj,ci->abij', I, T1, T2, RS)
    SigD += 0.5*einsum('klcd,ak,dcji,bl->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,ak,dcli,bj->abij', I, T1, T2, RS)
    SigD += 0.5*einsum('klcd,ak,dclj,bi->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,bk,al,dcji->abij', I, T1, T1, RD)
    SigD += 1.0*einsum('klcd,bk,dl,acji->abij', I, T1, T1, RD)
    SigD += -1.0*einsum('klcd,bk,adji,cl->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,bk,adli,cj->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,bk,adlj,ci->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,bk,dcji,al->abij', I, T1, T2, RS)
    SigD += 0.5*einsum('klcd,bk,dcli,aj->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,bk,dclj,ai->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,di,ak,bclj->abij', I, T1, T1, RD)
    SigD += 1.0*einsum('klcd,di,bk,aclj->abij', I, T1, T1, RD)
    SigD += -0.5*einsum('klcd,di,cj,balk->abij', I, T1, T1, RD)
    SigD += 1.0*einsum('klcd,di,ck,balj->abij', I, T1, T1, RD)
    SigD += -1.0*einsum('klcd,di,ackj,bl->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,di,aclk,bj->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,di,bakj,cl->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,di,balk,cj->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,di,bckj,al->abij', I, T1, T2, RS)
    SigD += 0.5*einsum('klcd,di,bclk,aj->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,dj,ak,bcli->abij', I, T1, T1, RD)
    SigD += -1.0*einsum('klcd,dj,bk,acli->abij', I, T1, T1, RD)
    SigD += -1.0*einsum('klcd,dj,ck,bali->abij', I, T1, T1, RD)
    SigD += 1.0*einsum('klcd,dj,acki,bl->abij', I, T1, T2, RS)
    SigD += 0.5*einsum('klcd,dj,aclk,bi->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,dj,baki,cl->abij', I, T1, T2, RS)
    SigD += 0.5*einsum('klcd,dj,balk,ci->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,dj,bcki,al->abij', I, T1, T2, RS)
    SigD += -0.5*einsum('klcd,dj,bclk,ai->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,dk,acji,bl->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,dk,acli,bj->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,dk,aclj,bi->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,dk,bali,cj->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,dk,balj,ci->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,dk,bcji,al->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('klcd,dk,bcli,aj->abij', I, T1, T2, RS)
    SigD += 1.0*einsum('klcd,dk,bclj,ai->abij', I, T1, T2, RS)
    SigD += -1.0*einsum('Ikc,ci,ak,I,bj->abij', g, T1, T1, S1, RS)
    SigD += 1.0*einsum('Ikc,ci,bk,I,aj->abij', g, T1, T1, S1, RS)
    SigD += 1.0*einsum('Ikc,cj,ak,I,bi->abij', g, T1, T1, S1, RS)
    SigD += -1.0*einsum('Ikc,cj,bk,I,ai->abij', g, T1, T1, S1, RS)

    SigD += 1.0*einsum('klcd,di,ak,cl,bj->abij', I, T1, T1, T1, RS)
    SigD += 1.0*einsum('klcd,di,bk,al,cj->abij', I, T1, T1, T1, RS)
    SigD += -1.0*einsum('klcd,di,bk,cl,aj->abij', I, T1, T1, T1, RS)
    SigD += -1.0*einsum('klcd,di,cj,ak,bl->abij', I, T1, T1, T1, RS)
    SigD += 1.0*einsum('klcd,di,cj,bk,al->abij', I, T1, T1, T1, RS)
    SigD += -1.0*einsum('klcd,dj,ak,cl,bi->abij', I, T1, T1, T1, RS)
    SigD += -1.0*einsum('klcd,dj,bk,al,ci->abij', I, T1, T1, T1, RS)
    SigD += 1.0*einsum('klcd,dj,bk,cl,ai->abij', I, T1, T1, T1, RS)
    
    '''
    EOM-CCSD sig1 part
    
     + 1.0\sum_{J}w_{IJ}R1_{J}
     + 1.0\sum_{ia}f_{ia}R11_{Iai}
     + 1.0\sum_{ia}g_{Iia}RS_{ai}
     + 1.0\sum_{Jia}g_{Jia}S1_{J}R11_{Iai}
     + 1.0\sum_{Jia}g_{Jia}U11_{Iai}R1_{J}
     - 1.0\sum_{ijab}I_{ijab}T1_{bi}R11_{Iaj}
     - 1.0\sum_{ijab}I_{ijab}U11_{Ibi}RS_{aj}
    
     einsum format=
    '''
    
    Sig1 += 1.0*einsum('IJ,J->I', w, R1)
    Sig1 += 1.0*einsum('ia,Iai->I', f, R11)
    Sig1 += 1.0*einsum('Iia,ai->I', g, RS)
    Sig1 += 1.0*einsum('Jia,J,Iai->I', g, S1, R11)
    Sig1 += 1.0*einsum('Jia,Iai,J->I', g, U11, R1)
    Sig1 += -1.0*einsum('ijab,bi,Iaj->I', I, T1, R11)
    Sig1 += -1.0*einsum('ijab,Ibi,aj->I', I, U11, RS)
    
    '''
    EOM-CCSD sigU1 part
    
     + 1.0G_{I}RS_{ai}
     + 1.0f_{ai}R1_{I}
     - 1.0\sum_{j}f_{ji}R11_{Iaj}
     + 1.0\sum_{b}f_{ab}R11_{Ibi}
     + 1.0\sum_{J}w_{IJ}R11_{Jai}
     - 1.0\sum_{j}g_{Iji}RS_{aj}
     + 1.0\sum_{b}g_{Iab}RS_{bi}
     - 1.0\sum_{jb}g_{Ijb}RD_{abji}
     - 1.0\sum_{jb}I_{jaib}R11_{Ibj}
     + 1.0\sum_{J}G_{J}U11_{Jai}R1_{I}
     - 1.0\sum_{j}f_{ji}T1_{aj}R1_{I}
     + 1.0\sum_{b}f_{ab}T1_{bi}R1_{I}
     + 1.0\sum_{J}w_{IJ}S1_{J}RS_{ai}
     + 1.0\sum_{J}g_{Jai}S1_{J}R1_{I}
     - 1.0\sum_{jb}f_{jb}T1_{aj}R11_{Ibi}
     - 1.0\sum_{jb}f_{jb}T1_{bi}R11_{Iaj}
     - 1.0\sum_{jb}f_{jb}U11_{Iaj}RS_{bi}
     - 1.0\sum_{jb}f_{jb}U11_{Ibi}RS_{aj}
     + 1.0\sum_{jb}f_{jb}U11_{Ibj}RS_{ai}
     - 1.0\sum_{jb}f_{jb}T2_{abji}R1_{I}
     - 1.0\sum_{jb}g_{Ijb}T1_{aj}RS_{bi}
     - 1.0\sum_{jb}g_{Ijb}T1_{bi}RS_{aj}
     + 1.0\sum_{jb}g_{Ijb}T1_{bj}RS_{ai}
     - 1.0\sum_{Jj}g_{Jji}S1_{J}R11_{Iaj}
     - 1.0\sum_{Jj}g_{Jji}U11_{Iaj}R1_{J}
     - 1.0\sum_{Jj}g_{Jji}U11_{Jaj}R1_{I}
     + 1.0\sum_{Jb}g_{Jab}S1_{J}R11_{Ibi}
     + 1.0\sum_{Jb}g_{Jab}U11_{Ibi}R1_{J}
     + 1.0\sum_{Jb}g_{Jab}U11_{Jbi}R1_{I}
     - 1.0\sum_{jb}I_{jaib}T1_{bj}R1_{I}
     - 1.0\sum_{Jjb}g_{Jjb}U11_{Iaj}R11_{Jbi}
     - 1.0\sum_{Jjb}g_{Jjb}U11_{Ibi}R11_{Jaj}
     + 1.0\sum_{Jjb}g_{Jjb}U11_{Ibj}R11_{Jai}
     + 1.0\sum_{Jjb}g_{Jjb}U11_{Jai}R11_{Ibj}
     - 1.0\sum_{Jjb}g_{Jjb}U11_{Jaj}R11_{Ibi}
     - 1.0\sum_{Jjb}g_{Jjb}U11_{Jbi}R11_{Iaj}
     - 1.0\sum_{jkb}I_{jkib}T1_{aj}R11_{Ibk}
     + 1.0\sum_{jkb}I_{jkib}T1_{bj}R11_{Iak}
     - 1.0\sum_{jkb}I_{jkib}U11_{Iaj}RS_{bk}
     + 1.0\sum_{jkb}I_{jkib}U11_{Ibj}RS_{ak}
     + 0.5\sum_{jkb}I_{jkib}T2_{abkj}R1_{I}
     + 1.0\sum_{jbc}I_{jabc}T1_{ci}R11_{Ibj}
     - 1.0\sum_{jbc}I_{jabc}T1_{cj}R11_{Ibi}
     + 1.0\sum_{jbc}I_{jabc}U11_{Ici}RS_{bj}
     - 1.0\sum_{jbc}I_{jabc}U11_{Icj}RS_{bi}
     - 0.5\sum_{jbc}I_{jabc}T2_{cbji}R1_{I}
     - 0.5\sum_{jkbc}I_{jkbc}U11_{Iaj}RD_{cbki}
     - 0.5\sum_{jkbc}I_{jkbc}U11_{Ici}RD_{abkj}
     + 1.0\sum_{jkbc}I_{jkbc}U11_{Icj}RD_{abki}
     + 1.0\sum_{jkbc}I_{jkbc}T2_{acji}R11_{Ibk}
     + 0.5\sum_{jkbc}I_{jkbc}T2_{ackj}R11_{Ibi}
     + 0.5\sum_{jkbc}I_{jkbc}T2_{cbji}R11_{Iak}
     - 1.0\sum_{jb}f_{jb}T1_{bi}T1_{aj}R1_{I}
     - 1.0\sum_{Jj}g_{Jji}T1_{aj}S1_{J}R1_{I}
     + 1.0\sum_{Jb}g_{Jab}T1_{bi}S1_{J}R1_{I}
     - 1.0\sum_{Jjb}g_{Jjb}S1_{J}U11_{Iaj}RS_{bi}
     - 1.0\sum_{Jjb}g_{Jjb}S1_{J}U11_{Ibi}RS_{aj}
     + 1.0\sum_{Jjb}g_{Jjb}S1_{J}U11_{Ibj}RS_{ai}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{aj}S1_{J}R11_{Ibi}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{aj}U11_{Ibi}R1_{J}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{aj}U11_{Jbi}R1_{I}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{bi}S1_{J}R11_{Iaj}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{bi}U11_{Iaj}R1_{J}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{bi}U11_{Jaj}R1_{I}
     + 1.0\sum_{Jjb}g_{Jjb}T1_{bj}U11_{Jai}R1_{I}
     - 1.0\sum_{Jjb}g_{Jjb}T2_{abji}S1_{J}R1_{I}
     - 1.0\sum_{jkb}I_{jkib}T1_{aj}T1_{bk}R1_{I}
     + 1.0\sum_{jbc}I_{jabc}T1_{ci}T1_{bj}R1_{I}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{aj}T1_{ck}R11_{Ibi}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{aj}U11_{Ici}RS_{bk}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{aj}U11_{Ick}RS_{bi}
     - 0.5\sum_{jkbc}I_{jkbc}T1_{aj}T2_{cbki}R1_{I}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{ci}T1_{aj}R11_{Ibk}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{ci}T1_{bj}R11_{Iak}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{ci}U11_{Iaj}RS_{bk}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{ci}U11_{Ibj}RS_{ak}
     - 0.5\sum_{jkbc}I_{jkbc}T1_{ci}T2_{abkj}R1_{I}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{cj}U11_{Iak}RS_{bi}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{cj}U11_{Ibi}RS_{ak}
     - 1.0\sum_{jkbc}I_{jkbc}T1_{cj}U11_{Ibk}RS_{ai}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{cj}T2_{abki}R1_{I}
     - 1.0\sum_{Jjb}g_{Jjb}T1_{bi}T1_{aj}S1_{J}R1_{I}
     + 1.0\sum_{jkbc}I_{jkbc}T1_{ci}T1_{aj}T1_{bk}R1_{I}
    
     einsum format=
    '''
    
    SigU1 += 1.0*einsum('I,ai->Iai', G, RS)
    SigU1 += 1.0*einsum('ai,I->Iai', f, R1)
    SigU1 += -1.0*einsum('ji,Iaj->Iai', f, R11)
    SigU1 += 1.0*einsum('ab,Ibi->Iai', f, R11)
    SigU1 += 1.0*einsum('IJ,Jai->Iai', w, R11)
    SigU1 += -1.0*einsum('Iji,aj->Iai', g, RS)
    SigU1 += 1.0*einsum('Iab,bi->Iai', g, RS)
    SigU1 += -1.0*einsum('Ijb,abji->Iai', g, RD)
    SigU1 += -1.0*einsum('jaib,Ibj->Iai', I, R11)
    SigU1 += 1.0*einsum('J,Jai,I->Iai', G, U11, R1)
    SigU1 += -1.0*einsum('ji,aj,I->Iai', f, T1, R1)
    SigU1 += 1.0*einsum('ab,bi,I->Iai', f, T1, R1)
    SigU1 += 1.0*einsum('IJ,J,ai->Iai', w, S1, RS)
    SigU1 += 1.0*einsum('Jai,J,I->Iai', g, S1, R1)
    SigU1 += -1.0*einsum('jb,aj,Ibi->Iai', f, T1, R11)
    SigU1 += -1.0*einsum('jb,bi,Iaj->Iai', f, T1, R11)
    SigU1 += -1.0*einsum('jb,Iaj,bi->Iai', f, U11, RS)
    SigU1 += -1.0*einsum('jb,Ibi,aj->Iai', f, U11, RS)
    SigU1 += 1.0*einsum('jb,Ibj,ai->Iai', f, U11, RS)
    SigU1 += -1.0*einsum('jb,abji,I->Iai', f, T2, R1)
    SigU1 += -1.0*einsum('Ijb,aj,bi->Iai', g, T1, RS)
    SigU1 += -1.0*einsum('Ijb,bi,aj->Iai', g, T1, RS)
    SigU1 += 1.0*einsum('Ijb,bj,ai->Iai', g, T1, RS)
    SigU1 += -1.0*einsum('Jji,J,Iaj->Iai', g, S1, R11)
    SigU1 += -1.0*einsum('Jji,Iaj,J->Iai', g, U11, R1)
    SigU1 += -1.0*einsum('Jji,Jaj,I->Iai', g, U11, R1)
    SigU1 += 1.0*einsum('Jab,J,Ibi->Iai', g, S1, R11)
    SigU1 += 1.0*einsum('Jab,Ibi,J->Iai', g, U11, R1)
    SigU1 += 1.0*einsum('Jab,Jbi,I->Iai', g, U11, R1)
    SigU1 += -1.0*einsum('jaib,bj,I->Iai', I, T1, R1)
    SigU1 += -1.0*einsum('Jjb,Iaj,Jbi->Iai', g, U11, R11)
    SigU1 += -1.0*einsum('Jjb,Ibi,Jaj->Iai', g, U11, R11)
    SigU1 += 1.0*einsum('Jjb,Ibj,Jai->Iai', g, U11, R11)
    SigU1 += 1.0*einsum('Jjb,Jai,Ibj->Iai', g, U11, R11)
    SigU1 += -1.0*einsum('Jjb,Jaj,Ibi->Iai', g, U11, R11)
    SigU1 += -1.0*einsum('Jjb,Jbi,Iaj->Iai', g, U11, R11)
    SigU1 += -1.0*einsum('jkib,aj,Ibk->Iai', I, T1, R11)
    SigU1 += 1.0*einsum('jkib,bj,Iak->Iai', I, T1, R11)
    SigU1 += -1.0*einsum('jkib,Iaj,bk->Iai', I, U11, RS)
    SigU1 += 1.0*einsum('jkib,Ibj,ak->Iai', I, U11, RS)
    SigU1 += 0.5*einsum('jkib,abkj,I->Iai', I, T2, R1)
    SigU1 += 1.0*einsum('jabc,ci,Ibj->Iai', I, T1, R11)
    SigU1 += -1.0*einsum('jabc,cj,Ibi->Iai', I, T1, R11)
    SigU1 += 1.0*einsum('jabc,Ici,bj->Iai', I, U11, RS)
    SigU1 += -1.0*einsum('jabc,Icj,bi->Iai', I, U11, RS)
    SigU1 += -0.5*einsum('jabc,cbji,I->Iai', I, T2, R1)
    SigU1 += -0.5*einsum('jkbc,Iaj,cbki->Iai', I, U11, RD)
    SigU1 += -0.5*einsum('jkbc,Ici,abkj->Iai', I, U11, RD)
    SigU1 += 1.0*einsum('jkbc,Icj,abki->Iai', I, U11, RD)
    SigU1 += 1.0*einsum('jkbc,acji,Ibk->Iai', I, T2, R11)
    SigU1 += 0.5*einsum('jkbc,ackj,Ibi->Iai', I, T2, R11)
    SigU1 += 0.5*einsum('jkbc,cbji,Iak->Iai', I, T2, R11)
    SigU1 += -1.0*einsum('jb,bi,aj,I->Iai', f, T1, T1, R1)
    SigU1 += -1.0*einsum('Jji,aj,J,I->Iai', g, T1, S1, R1)
    SigU1 += 1.0*einsum('Jab,bi,J,I->Iai', g, T1, S1, R1)
    SigU1 += -1.0*einsum('Jjb,J,Iaj,bi->Iai', g, S1, U11, RS)
    SigU1 += -1.0*einsum('Jjb,J,Ibi,aj->Iai', g, S1, U11, RS)
    SigU1 += 1.0*einsum('Jjb,J,Ibj,ai->Iai', g, S1, U11, RS)
    SigU1 += -1.0*einsum('Jjb,aj,J,Ibi->Iai', g, T1, S1, R11)
    SigU1 += -1.0*einsum('Jjb,aj,Ibi,J->Iai', g, T1, U11, R1)
    SigU1 += -1.0*einsum('Jjb,aj,Jbi,I->Iai', g, T1, U11, R1)
    SigU1 += -1.0*einsum('Jjb,bi,J,Iaj->Iai', g, T1, S1, R11)
    SigU1 += -1.0*einsum('Jjb,bi,Iaj,J->Iai', g, T1, U11, R1)
    SigU1 += -1.0*einsum('Jjb,bi,Jaj,I->Iai', g, T1, U11, R1)
    SigU1 += 1.0*einsum('Jjb,bj,Jai,I->Iai', g, T1, U11, R1)
    SigU1 += -1.0*einsum('Jjb,abji,J,I->Iai', g, T2, S1, R1)
    SigU1 += -1.0*einsum('jkib,aj,bk,I->Iai', I, T1, T1, R1)
    SigU1 += 1.0*einsum('jabc,ci,bj,I->Iai', I, T1, T1, R1)
    SigU1 += -1.0*einsum('jkbc,aj,ck,Ibi->Iai', I, T1, T1, R11)
    SigU1 += 1.0*einsum('jkbc,aj,Ici,bk->Iai', I, T1, U11, RS)
    SigU1 += -1.0*einsum('jkbc,aj,Ick,bi->Iai', I, T1, U11, RS)
    SigU1 += -0.5*einsum('jkbc,aj,cbki,I->Iai', I, T1, T2, R1)
    SigU1 += 1.0*einsum('jkbc,ci,aj,Ibk->Iai', I, T1, T1, R11)
    SigU1 += -1.0*einsum('jkbc,ci,bj,Iak->Iai', I, T1, T1, R11)
    SigU1 += 1.0*einsum('jkbc,ci,Iaj,bk->Iai', I, T1, U11, RS)
    SigU1 += -1.0*einsum('jkbc,ci,Ibj,ak->Iai', I, T1, U11, RS)
    SigU1 += -0.5*einsum('jkbc,ci,abkj,I->Iai', I, T1, T2, R1)
    SigU1 += 1.0*einsum('jkbc,cj,Iak,bi->Iai', I, T1, U11, RS)
    SigU1 += 1.0*einsum('jkbc,cj,Ibi,ak->Iai', I, T1, U11, RS)
    SigU1 += -1.0*einsum('jkbc,cj,Ibk,ai->Iai', I, T1, U11, RS)
    SigU1 += 1.0*einsum('jkbc,cj,abki,I->Iai', I, T1, T2, R1)
    SigU1 += -1.0*einsum('Jjb,bi,aj,J,I->Iai', g, T1, T1, S1, R1)
    SigU1 += 1.0*einsum('jkbc,ci,aj,bk,I->Iai', I, T1, T1, T1, R1)
