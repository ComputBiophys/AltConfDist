from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from pyrosetta import rosetta
from scipy import stats
from scipy.special import expit

DISTANCE_EPSILON = 1e-9
LOGGER = logging.getLogger(__name__)
__all__ = [
    "add_restraints",
    "generate_distance_restraints",
    "read_fasta_sequence",
    "remove_clash",
    "set_random_dihedral",
]


def generate_distance_restraints(
    npz: np.ndarray,
    u: float,
    dist_flag: np.ndarray,
    H: np.ndarray,
    tmpdir: str,
    params: dict[str, Any],
    pose: Any,
    p_last: float,
    p_nl: float,
) -> dict[str, list[list[Any]]]:

    flagged_pairs = {(int(a), int(b)) for a, b in np.argwhere(dist_flag)}
    LOGGER.debug("Flagged distance-map pairs: %d", len(flagged_pairs))
    dist = npz
    dist = dist.astype(np.float32) + DISTANCE_EPSILON
    rst = {"dist": [], "omega": [], "theta": [], "phi": [], "rep": []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT1 = params["PCUT1"]
    EBASE = params["EBASE"]
    EREP = params["EREP"]
    DREP = params["DREP"]
    MEFF = params["MEFF"]
    # DCUT  = params['DCUT']
    DCUT = 21.6875
    ALPHA = params["ALPHA"]
    MEFF = 0.0001
    EBASE = -0.5

    # DSTEP = params['DSTEP']
    DSTEP = 0.3125

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    ###original p<PCUT abandoned
    # for c in range(0,nres):
    # for d in range(0,nres):
    # vec = np.where(dist[c,d]>(PCUT+eps))
    # print(vec[0],dist[c,d][vec[0]])
    # for e in range(0,64):
    # if e not in vec[0]:
    # dist[c,d,e]=float(eps)
    bins = np.array([3.25 + DSTEP * i for i in range(61)])  # DSTEP=0.3125
    prob = np.sum(dist[:, :, 5:], axis=-1)  # p for 3.875 A further
    bkgr = np.array((bins / DCUT) ** ALPHA)  # DCUT=21.6875, ALPHA = 1.57
    LOGGER.debug("Original distance input: %s", dist[57, 177])
    # attr = -np.log((dist[:,:,5:]+MEFF)/(dist[:,:,-1][:,:,None]*bkgr[None,None,:]))+EBASE #MEFF=0.0001,EBASE=-0.5
    # attr = -np.log((dist[:,:,3:]+MEFF)/(np.sum(dist[:,:,-3:-1][:,:,None],axis=3)*bkgr[None,None,:]))+EBASE #MEFF=0.0001,EBASE=-0.5
    attr = (
        -np.log(
            (dist[:, :, 3:] + MEFF) / (dist[:, :, -2][:, :, None] * bkgr[None, None, :])
        )
        + EBASE
    )  # MEFF=0.0001,EBASE=-0.5
    LOGGER.debug("Distance energies: %s", attr[57, 177])
    repul = (
        np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None]
        + np.array(EREP)[None, None, :]
    )  # EREP= [10.0,3.0,0.5]
    dist = np.concatenate([repul, attr], axis=-1)
    LOGGER.debug("Processed distances: %s", dist[57, 177])
    bins = np.concatenate([DREP, bins])  # DREP = [0.0,2.0,3.5]---0,2,3
    LOGGER.debug("Distance bins: %s", bins)
    i, j = np.where(prob > PCUT1)
    prob = prob[i, j]
    # test_attr = dist[:,:,-1][:,:,None]*bkgr[None,None,:] #only last bin for all pairs*bkgr [l,l,1]*[61]=[l,l,61]
    # print("bkgr:",bkgr,"dist-1*bkgr", test_attr.shape)
    # print("i,j",i,j,len(i))
    # for n in range(0,len(i)):
    # x = np.where(dist[i[n],j[n]]<0)
    # if len(x[0])<=1 and int(x[0]) >=63:
    # print('d_distribution of last bin',i[n]+1,j[n]+1,len(x[0]),int(x[0]))
    # inew.append(i[n])
    # jnew.append(j[n])
    ##continue
    # else:
    # print('last bin',i[n]+1,j[n]+1,x[0])
    # inew.append(i[n])
    # jnew.append(j[n])
    # print("new i j:",len(inew),bins,len(bins))
    # prob = prob[inew,jnew]
    nbins = 64
    step = 0.3125
    # u = 0.1

    e_single = []
    min_edge = []
    min_edge_multi = []
    for a, b, _ in zip(i, j, prob, strict=True):
        if b > a + 6:
            name = str(Path(tmpdir) / f"{a + 1}.{b + 1}.txt")
            tag = 0
            resid_a = pose.residue(a + 1)
            a_CB = resid_a.xyz("CB")
            resid_b = pose.residue(b + 1)
            b_CB = resid_b.xyz("CB")
            a_b_vec = a_CB - b_CB
            dij = float(f"{a_b_vec.norm():.1f}")
            if (a, b) in flagged_pairs:
                tag = 1
                for distance_bin in range(0, 64):
                    if dist[a, b, distance_bin] >= 0:
                        dist[a, b, distance_bin] = math.tanh(dist[a, b, distance_bin])
                min_index = np.argmin(dist[a, b])
                if dij < 21.6875 and min_index > dij:
                    nleft = int((dij - 2) / 0.3125)
                    nright = np.where(dist[a, b] < 0)[0][0]
                    for distance_bin in range(nleft, nright):
                        dist[a, b, distance_bin] -= (1 / (nright - nleft)) * (
                            distance_bin - nleft
                        )
                elif dij < 21.6875 and min_index < dij:
                    nright = int((dij - 2) / 0.3125)
                    nleft = np.where(dist[a, b] < 0)[0][0]
                    for distance_bin in range(nleft, nright):
                        dist[a, b, distance_bin] -= (1 / (nright - nleft)) * (
                            distance_bin - nleft
                        )
                min_edge_multi.append(np.min(dist[a, b]))
            if tag != 1:
                min_edge.append(np.min(dist[a, b]))
                d_int = []
                d_int = [int(value) for value in dist[a, b] if value > 0]
                platform = stats.mode(d_int).mode
                # if a==147 and b==205:
                # print("single origin!!",a+1,b+1, platform,'\n',dist[a,b])
                for distance_bin in range(0, 64):
                    if dist[a, b, distance_bin] > platform - 1:
                        # if dist[a,b,bin]>=0:
                        dist[a, b, distance_bin] = (
                            platform
                            - 1
                            + math.tanh(dist[a, b, distance_bin] - platform + 1)
                        )
                e_single.append(np.mean(dist[a, b, 2:]))
                # print("single platform!!",a+1,b+1,'\n',dist[a,b])
                # print("e_single of 61bins",e_single[-1],a+1,b+1)

    LOGGER.debug(
        "Single-pair energy range: %.4f to %.4f (mean %.4f)",
        min(min_edge),
        max(min_edge),
        np.mean(min_edge),
    )
    LOGGER.debug(
        "Multi-pair energy range: %.4f to %.4f (mean %.4f)",
        min(min_edge_multi),
        max(min_edge_multi),
        np.mean(min_edge_multi),
    )
    LOGGER.debug(
        "Single-peak mean energies (q25/q50/q75): %.4f / %.4f / %.4f",
        np.percentile(e_single, 25),
        np.percentile(e_single, 50),
        np.percentile(e_single, 75),
    )
    if p_last > 0.5:
        # mag = p_nl*(1-expit(p_last))*(np.mean(min_edge_multi)/np.mean(min_edge))
        # mag =(1-expit(p_last))*(np.mean(min_edge_multi)/np.mean(min_edge))
        mag = p_nl * (1 - expit(p_last)) * max(min_edge_multi) / min(min_edge)
        # mag = (1-expit(p_last))*max(min_edge_multi)/min(min_edge)
        # mag = 0.001
    else:
        mag = p_nl * (1 - expit(p_last)) * (np.mean(min_edge_multi) / np.mean(min_edge))
        # mag = (1-expit(p_last))*(np.mean(min_edge_multi)/np.mean(min_edge))
        # mag = p_nl*(1-expit(p_last))*max(min_edge_multi)/min(min_edge)
        # mag = 1-expit(p_last)
        # mag = (1-expit(p_last))*max(min_edge_multi)/min(min_edge)
        # mag = 0.001
    LOGGER.debug("Restraint magnitude: %.6f", mag)
    # for x in range(0,len(dist)):
    # for y in range(0,len(dist[0])):
    # if (x not in flag[0]) or (y not in flag[1]):
    # dist[x,y]=mag*dist[x,y]
    # print("rest pairs like",np.min(dist[x,y]))
    for a, b, p in zip(i, j, prob, strict=True):
        if b > a + 6:
            name = str(Path(tmpdir) / f"{a + 1}.{b + 1}.txt")
            dist[a, b] = (H[a, b] + 1e-5) * mag * dist[a, b]
            if (a, b) in flagged_pairs:
                dist[a, b] = (1 / (mag * (H[a, b] + 1e-5))) * dist[a, b]

            Path(name).write_text(
                "x_axis"
                + "\t%.4f" * nbins % tuple(bins)
                + "\n"
                + "y_axis"
                + "\t%.4f" * nbins % tuple(u * dist[a, b])
                + "\n",
                encoding="utf-8",
            )
            rst_line = (
                f"AtomPair CB {a + 1} CB {b + 1} "
                f"SPLINE TAG {name} 1.0 {1.0:.3f} {step:.5f}"
            )
            rst["dist"].append([a, b, p, rst_line])

    LOGGER.info("Distance restraints: %d", len(rst["dist"]))

    return rst


def set_random_dihedral(pose: Any) -> Any:
    nres = pose.total_residue()
    for i in range(1, nres):
        phi, psi = random_dihedral()
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, 180)

    return pose


# pick phi/psi randomly from:
# -140  153 180 0.135 B
# -72  145 180 0.155 B
# -122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral() -> tuple[int, int]:
    phi = 0
    psi = 0
    r = random.random()
    if r <= 0.135:
        phi = -140
        psi = 153
    elif r > 0.135 and r <= 0.29:
        phi = -72
        psi = 145
    elif r > 0.29 and r <= 0.363:
        phi = -122
        psi = 117
    elif r > 0.363 and r <= 0.485:
        phi = -82
        psi = -14
    elif r > 0.485 and r <= 0.982:
        phi = -61
        psi = -41
    else:
        phi = 57
        psi = 39
    return (phi, psi)


def read_fasta_sequence(path: str | Path) -> str:
    with open(path) as handle:
        lines = (line.strip() for line in handle if not line.startswith(">"))
        return "".join(line for line in lines if line)


def remove_clash(scorefxn: Any, mover: Any, pose: Any) -> None:
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_restraints(
    pose: Any,
    rst: dict[str, list[list[Any]]],
    n: int,
    u: float,
    sep1: int,
    sep2: int,
    params: dict[str, Any],
    nogly: bool = False,
) -> None:
    pcut = params["PCUT"]
    sequence = params["seq"]
    lines = []

    def include(a, b, probability, threshold):
        in_range = sep1 <= abs(a - b) < sep2
        if not in_range or probability < threshold:
            return False
        return not nogly or (sequence[a] != "G" and sequence[b] != "G")

    lines.extend(
        line
        for a, b, probability, line in rst["dist"]
        if include(a, b, probability, pcut)
    )
    if nogly and params["USE_ORIENT"]:
        for restraint_type, threshold in (
            ("omega", pcut + 0.5),
            ("theta", pcut + 0.5),
            ("phi", pcut + 0.6),
        ):
            lines.extend(
                line
                for a, b, probability, line in rst[restraint_type]
                if include(a, b, probability, threshold)
            )

    if not lines:
        return

    random.shuffle(lines)
    constraint_path = Path(params["TDIR"]) / f"{n}_{u}minimize.cst"
    constraint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(str(constraint_path))
    constraints.add_constraints(True)
    constraints.apply(pose)
    constraint_path.unlink()
