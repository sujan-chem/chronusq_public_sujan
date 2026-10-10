"""
Single-determinant (Kramers-unrestricted, spinor) X2C HF for the Co atom and
Co+ cation with a FIXED spinor occupation, in the spirit of
Zhang & Cheng, JCTC 18, 6732 (2022) and the socutils U/UO workflow.

Target states (high-|MJ| component of the ground J level):

  Co   3d7 4s2   4F_9/2, MJ = +9/2
       dominant spinor determinant:
       [Ar] 3d3/2^4 [3d5/2, 5/2]^1 [3d5/2, 3/2]^1 [3d5/2, 1/2]^1 4s1/2^2

  Co+  3d8       3F_4,   MJ = +4
       dominant spinor determinant:
       [Ar] 3d3/2^4 [3d5/2, 5/2]^1 [3d5/2, 3/2]^1 [3d5/2, 1/2]^1 [3d5/2,-1/2]^1

Occupation dictionary: key = m_j, value = number of occupied spinors of that
m_j in each angular block, using the same layout as the U / O examples in the
socutils workflow:  [ 0, s, p, d, f ].
(For O there: '1/2':[0,2,2] = 1s,2s and p1/2+p3/2; for U: '7/2':[0,0,0,0,1] = 4f7/2.)

The basis is read from ChronusQ's dyall_v2z.gbs so that the HF energy is
directly comparable to the ChronusQ CASSCF run.
"""
import argparse
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.somf import x2cmp

#  Co (27 e):  1s 2s 3s 4s | 2p 3p | 3d3/2^4 3d5/2^3 (mj = 5/2, 3/2, 1/2)
#     mj :        s  p  d  f
OCC_CO = {
     '1/2': [0, 4, 4, 2, 0],   # s: 1s-4s | p: 2p,3p (p1/2+p3/2) | d: 3d3/2, 3d5/2
    '-1/2': [0, 4, 4, 1, 0],   #                                  | d: 3d3/2 only
     '3/2': [0, 0, 2, 2, 0],   # p3/2 (2p,3p)                      | d: 3d3/2, 3d5/2
    '-3/2': [0, 0, 2, 1, 0],   #                                  | d: 3d3/2 only
     '5/2': [0, 0, 0, 1, 0],   # 3d5/2
    '-5/2': [0, 0, 0, 0, 0],
}   # total = 10 + 9 + 4 + 3 + 1 = 27,  MJ = 9/2

#  Co+ (26 e): 4s removed, one more 3d5/2 electron (mj = -1/2)
OCC_COP = {
     '1/2': [0, 3, 4, 2, 0],
    '-1/2': [0, 3, 4, 2, 0],
     '3/2': [0, 0, 2, 2, 0],
    '-3/2': [0, 0, 2, 1, 0],
     '5/2': [0, 0, 0, 1, 0],
    '-5/2': [0, 0, 0, 0, 0],
}   # total = 9 + 9 + 4 + 3 + 1 = 26,  MJ = 4

MOLECULES = {
    'Co':  {'occupation': OCC_CO,  'charge': 0, 'spin': 3},
    'Cop': {'occupation': OCC_COP, 'charge': 1, 'spin': 2},
}


def read_gbs(path, element):
    """Read one element from a Gaussian-format .gbs file into PySCF format."""
    lmap = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5}
    shells, inside = [], False
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if not inside:
            if tok and tok[0].upper() == element.upper() and len(tok) == 2:
                inside = True
            i += 1
            continue
        if not tok or tok[0] == '****':
            break
        lab, nprim = tok[0].upper(), int(tok[1])
        prims = [[float(x.replace('D', 'E')) for x in lines[i + 1 + k].split()]
                 for k in range(nprim)]
        if lab == 'SP':
            shells.append([0] + [[p[0], p[1]] for p in prims])
            shells.append([1] + [[p[0], p[2]] for p in prims])
        else:
            shells.append([lmap[lab]] + [[p[0], p[1]] for p in prims])
        i += 1 + nprim
    if not shells:
        raise ValueError(f'{element} not found in {path}')
    return shells


def check_occupation(occ, nelec):
    from fractions import Fraction
    n = sum(sum(v) for v in occ.values())
    mj = sum(Fraction(k) * sum(v) for k, v in occ.items())
    print(f'  occupation: {n} electrons (expected {nelec}), MJ = {mj}')
    assert n == nelec, 'occupation does not match the electron count'


def run(name, gbs, gaunt, breit, amf_type):
    info = MOLECULES[name]
    mol = gto.M(
        atom='Co 0.0 0.0 0.0',
        basis={'Co': read_gbs(gbs, 'Co')},
        nucmod='G',
        unit='Bohr',
        charge=info['charge'],
        spin=info['spin'],
        verbose=5)
    check_occupation(info['occupation'], mol.nelectron)

    mf = spinor_hf.SpinorSymmSCF(mol, symmetry='linear', occup=info['occupation'])
    mf.with_x2c = x2cmp.SpinorX2CMPHelper(mol, x2cmp=amf_type, with_gaunt=gaunt,
                                          with_breit=breit, with_aoc=False)
    scheme = 'dcb' if breit else ('dcg' if gaunt else 'dc')
    mf.chkfile = f'{name}_{amf_type}_{scheme}_frac.chk'
    mf.max_cycle = 200
    mf.kernel()
    return mf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gbs', required=True,
                    help='path to chronusq_public/basis/dyall_v2z.gbs')
    ap.add_argument('--gaunt', action='store_true')
    ap.add_argument('--breit', action='store_true')
    ap.add_argument('--amf', default='x2cmp')
    args = ap.parse_args()

    results = {}
    for name in ['Co', 'Cop']:
        results[name] = run(name, args.gbs, args.gaunt, args.breit, args.amf)

    with open('energy.dat', 'a+') as f:
        for name, mf in results.items():
            f.write(f'{name} {mf.energy_tot():.10f} converged={mf.converged}\n')
    ie = (results['Cop'].energy_tot() - results['Co'].energy_tot()) * 27.211386
    print(f'HF IE(Co) = {ie:.4f} eV')
