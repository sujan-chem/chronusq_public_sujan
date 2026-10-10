"""
Single-determinant spinor (Kramers-unrestricted) X2CAMF-HF for the neutral
Co atom with a FIXED spinor occupation, in the spirit of
Zhang & Cheng, JCTC 18, 6732 (2022) and the socutils U/UO workflow.

Two steps:
  1. Co(2+)  3d7        -- converges easily (no 4s/3d competition).
  2. Co      3d7 4s2    -- the hard neutral atom, started from the Co(2+)
                           density.  Co(2+) has exactly the same 3d
                           occupation, so only the 4s pair is added.

Target state: Co 4F_9/2, MJ = +9/2, dominant spinor determinant
    [Ar] 3d3/2^4 [3d5/2, 5/2]^1 [3d5/2, 3/2]^1 [3d5/2, 1/2]^1 4s1/2^2
(Co(2+) is the same without 4s2, also MJ = +9/2.)

Occupation dictionary (socutils SymmSpinorSCF, symmetry='linear'):
    key   = m_j
    value = [n0, s, p, d, f]
            n0 : occupy the n0 lowest spinors of this m_j regardless of type
            s/p/d/f : then occupy the lowest spinors of that dominant l-type

The basis is read from ChronusQ's dyall_v2z.gbs so the two codes use the same
basis.

Run (PYTHONPATH must contain cdsoc_bagh and cdsoc_bagh/socutils):
    python workflow_co.py --gbs /path/to/chronusq_public/basis/dyall_v2z.gbs
"""
import argparse
import resource
from fractions import Fraction

# The Breit part of the X2CAMF integrals needs a large stack for Co
# (segfaults with the common 8 MB default).  Same as `ulimit -s unlimited`.
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (_hard, _hard))
except (ValueError, OSError):
    pass
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.somf import x2cmp

#  Co(2+) (25 e):  1s 2s 3s | 2p 3p | 3d3/2^4 3d5/2^3 (mj = 5/2, 3/2, 1/2)
#     mj :          n0 s  p  d  f
OCC_CO2P = {
     '1/2': [0, 3, 4, 2, 0],   # s: 1s-3s | p: 2p,3p (p1/2+p3/2) | d: 3d3/2, 3d5/2
    '-1/2': [0, 3, 4, 1, 0],   #                                  | d: 3d3/2 only
     '3/2': [0, 0, 2, 2, 0],   # p3/2 (2p,3p)                      | d: 3d3/2, 3d5/2
    '-3/2': [0, 0, 2, 1, 0],   #                                  | d: 3d3/2 only
     '5/2': [0, 0, 0, 1, 0],   # 3d5/2
    '-5/2': [0, 0, 0, 0, 0],
}   # total = 9 + 8 + 4 + 3 + 1 = 25,  MJ = 9/2

#  Co (27 e): Co(2+) + 4s1/2^2
OCC_CO = {
     '1/2': [0, 4, 4, 2, 0],
    '-1/2': [0, 4, 4, 1, 0],
     '3/2': [0, 0, 2, 2, 0],
    '-3/2': [0, 0, 2, 1, 0],
     '5/2': [0, 0, 0, 1, 0],
    '-5/2': [0, 0, 0, 0, 0],
}   # total = 10 + 9 + 4 + 3 + 1 = 27,  MJ = 9/2


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
    n = sum(sum(v) for v in occ.values())
    mj = sum(Fraction(k) * sum(v) for k, v in occ.items())
    print(f'  occupation: {n} electrons (expected {nelec}), MJ = {mj}')
    assert n == nelec, 'occupation does not match the electron count'


def build(charge, spin, occ, basis, args):
    mol = gto.M(
        atom='Co 0.0 0.0 0.0',
        basis={'Co': basis},
        nucmod='G',
        unit='Bohr',
        charge=charge,
        spin=spin,
        verbose=args.verbose)
    check_occupation(occ, mol.nelectron)
    mf = spinor_hf.SpinorSymmSCF(mol, symmetry='linear', occup=occ)
    mf.with_x2c = x2cmp.SpinorX2CMPHelper(mol, x2cmp=args.amf,
                                          with_gaunt=args.gaunt,
                                          with_breit=args.breit,
                                          with_aoc=False)
    mf.max_cycle = args.max_cycle
    return mf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gbs', required=True,
                    help='path to chronusq_public/basis/dyall_v2z.gbs')
    ap.add_argument('--dc', action='store_true',
                    help='Dirac-Coulomb only (default: Gaunt + Breit)')
    ap.add_argument('--amf', default='x2cmp')
    ap.add_argument('--max-cycle', type=int, default=200)
    ap.add_argument('--verbose', type=int, default=4)
    args = ap.parse_args()
    args.gaunt = args.breit = not args.dc
    scheme = 'dc' if args.dc else 'dcb'

    basis = read_gbs(args.gbs, 'Co')

    # Step 1: Co(2+) 3d7
    print('\n===== Step 1: Co(2+) 3d7, 4F9/2 MJ=9/2 =====')
    ion = build(2, 3, OCC_CO2P, basis, args)
    ion.chkfile = f'Co2p_{args.amf}_{scheme}.chk'
    ion.kernel()
    if not ion.converged:
        raise RuntimeError('Co(2+) SCF did not converge')

    # Step 2: neutral Co 3d7 4s2 from the Co(2+) density
    print('\n===== Step 2: Co 3d7 4s2, 4F9/2 MJ=9/2 (guess: Co(2+)) =====')
    mf = build(0, 3, OCC_CO, basis, args)
    mf.chkfile = f'Co_{args.amf}_{scheme}.chk'
    mf.kernel(dm0=ion.make_rdm1())

    with open('energy.dat', 'a+') as f:
        f.write(f'Co2+ {scheme} {ion.e_tot:.10f} converged={ion.converged}\n')
        f.write(f'Co   {scheme} {mf.e_tot:.10f} converged={mf.converged}\n')
    print(f'\nE(Co2+) = {ion.e_tot:.10f}  converged = {ion.converged}')
    print(f'E(Co)   = {mf.e_tot:.10f}  converged = {mf.converged}')
