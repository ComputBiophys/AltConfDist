from __future__ import annotations

import argparse
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, argrelmin

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    root: Path
    pdb_id: str
    chain_id: str
    target_dm: str
    gap_beg: int
    gap_end: int

    @property
    def signal_name(self) -> str:
        return f"alt-{self.pdb_id}"


@dataclass(slots=True)
class SignalAnalysisResult:
    name: str
    signals: pd.DataFrame
    alternative_distance_map: np.ndarray
    alternative_flags: np.ndarray
    entropy: np.ndarray
    p_last: float
    p_nl: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract alternative distance signals from AlphaFold2"
    )
    parser.add_argument(
        "PDBid", type=str, help="name of protein structure file for analysis"
    )
    parser.add_argument("ChainID", type=str, help="protein chain ID")
    parser.add_argument(
        "targetDM", type=str, help="name of AF2 predicted DM---.pickle file"
    )
    parser.add_argument(
        "gap_beg",
        type=int,
        nargs="?",
        default=0,
        help=(
            "number of missing residues at the beginning of the structure "
            "relative to the FASTA sequence used for DM prediction"
        ),
    )
    parser.add_argument(
        "gap_end",
        type=int,
        nargs="?",
        default=0,
        help=(
            "number of missing residues at the end of the structure "
            "relative to the FASTA sequence used for DM prediction"
        ),
    )
    return parser


def parse_config(argv: list[str] | None = None) -> AnalysisConfig:
    args = build_parser().parse_args(argv)
    return AnalysisConfig(
        root=Path(__file__).resolve().parent,
        pdb_id=args.PDBid,
        chain_id=args.ChainID,
        target_dm=args.targetDM,
        gap_beg=args.gap_beg,
        gap_end=args.gap_end,
    )


def gauss1(x, a, u, sig):
    return a * np.exp(-((x - u) ** 2) / (2 * sig**2)) / (sig * math.sqrt(2 * math.pi))


def gaussian2(x, a1, a2, u1, u2, sig1, sig2):
    return a1 * np.exp(-((x - u1) ** 2) / (2 * sig1**2)) / (
        sig1 * math.sqrt(2 * math.pi)
    ) + a2 * np.exp(-((x - u2) ** 2) / (2 * sig2**2)) / (
        sig2 * math.sqrt(2 * math.pi)
    )


def get_cb_coords(chain, **kwargs):
    return np.array(
        [_get_real_or_virtual_cb(res) for res in chain], dtype=np.float32, **kwargs
    )


def _get_real_or_virtual_cb(res, gly_ca: bool = True):
    if res.name != "GLY":
        try:
            return res["CB"][0].pos.tolist()
        except Exception:
            pass
    else:
        if gly_ca:
            return res["CA"][0].pos.tolist()
    Ca = res["CA"][0].pos
    b = Ca - res["N"][0].pos
    c = res["C"][0].pos - Ca
    a = b.cross(c)
    return (-0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca).tolist()


def prepare_structure(
    path: Path,
    chain_id: str,
    model_id: int = 0,
    begin: int = 0,
    end: int | None = None,
):
    st = gemmi.read_structure(str(path))
    chain = list(st[model_id][chain_id].get_polymer().first_conformer())[begin:end]
    three_letter_seq = [aa.name for aa in chain]
    one_letter_seq = gemmi.one_letter_code(three_letter_seq)
    bb_coords = get_cb_coords(chain)
    return bb_coords, one_letter_seq


def find_peak(
    distance: float,
    breadths: list[str | tuple[float, float]],
) -> tuple[bool, int | str]:
    if distance > 21.6875:
        return ("last" in breadths, "last" if "last" in breadths else "none")

    for index, breadth in enumerate(breadths):
        if breadth == "last":
            continue
        if breadth[0] <= distance <= breadth[-1]:
            return True, index + 1
    return False, "none"


def load_predicted_distogram(
    path: Path, gap_beg: int, gap_end: int
) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)

    logits = np.asarray(data["distogram"]["logits"], dtype=np.float32)
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    bin_edges = np.asarray(data["distogram"]["bin_edges"], dtype=np.float32)

    if gap_beg > 0 or gap_end > 0:
        seq_end = -gap_end if gap_end > 0 else None
        probabilities = probabilities[gap_beg:seq_end, gap_beg:seq_end, :]

    return probabilities, bin_edges


def analyze_signals(config: AnalysisConfig) -> SignalAnalysisResult:
    name = config.signal_name
    probs, bin_edges = load_predicted_distogram(
        config.root / "AF2_DM" / config.target_dm,
        config.gap_beg,
        config.gap_end,
    )
    probs_apo = np.array(probs)
    entropy = np.zeros((probs.shape[0], probs.shape[0]))
    protein_length = probs.shape[0]
    bins_64 = np.append(bin_edges, 22.0)
    x2 = bins_64
    apo_predictions = []
    pairs = []
    peak_counts = []
    apo_distances = []
    breadths = []
    peak_locations = []
    last_signal_count = 0
    single_peak_signal_count = 0
    peak_locations_apo = []
    apo_flags = np.zeros((protein_length, protein_length))

    structure_path = config.root / "pdb" / f"{config.pdb_id}.cif"
    bb_coords_apo, one_letter_seq_apo = prepare_structure(
        structure_path, config.chain_id
    )
    LOGGER.info(
        "Analyzing %s chain %s: %d residues",
        config.pdb_id,
        config.chain_id,
        protein_length,
    )

    for i in range(0, protein_length - 6):
        for j in range(i + 6, protein_length):
            atom1_apo = bb_coords_apo[i]
            atom2_apo = bb_coords_apo[j]
            d0_apo = float(np.linalg.norm(np.asarray(atom1_apo) - atom2_apo))
            # atom1_holo = bb_coords_holo[i]
            # atom2_holo = bb_coords_holo[j]
            # d_sub = [(atom1_holo[x]-atom2_holo[x])**2 for x in range(0,len(atom1_holo))]
            # d0_holo = math.sqrt(sum(d_sub))

            H = -sum(probs[i, j] * np.log(probs[i, j]))
            entropy[i, j] = H
            n_max = 0
            breadth = []
            uu = []
            signal_types = []
            maxvector = argrelextrema(np.array(probs[i][j][:-1]), np.greater)[0]
            minvector = argrelmin(np.array(probs[i][j][:-1]))[0]
            maxv = []
            # print("original max,min:",i+1,j+1,maxvector,minvector,len(maxvector),len(minvector))

            if (len(maxvector) < 1) or (len(minvector) < 1):
                continue
            if float(max(probs[i, j, maxvector])) < 1e-4 or (
                len(maxvector) == 1 and probs[i][j][-1] <= 0.4 * max(probs[i][j])
            ):
                continue
            if len(maxvector) >= 2:
                for peak_index in range(0, len(maxvector) - 1):
                    if (
                        maxvector[peak_index + 1] - maxvector[peak_index] > 2
                        and probs[i][j][maxvector[peak_index]] > 1e-4
                    ):
                        maxv.append(maxvector[peak_index])
                    else:
                        continue
                if maxvector[-1] - maxvector[-2] > 2:
                    maxv.append(maxvector[-1])
                if len(maxv) >= 2:
                    tag = 0
                    p_peaks = list(np.array(probs[i][j][maxv]))
                    p_max1 = max(p_peaks)
                    if sum(p_peaks) <= 0.1:
                        continue  ##all p_peak_max too low
                    index_max1 = p_peaks.index(p_max1)

                    for x in maxv:
                        if x >= 57:
                            tag += 1

                    u1 = bins_64[maxv[index_max1]]
                    # print("max1",maxvector,maxv,p_peaks,p_max1,index_max1,maxv[index_max1],u1)
                    p_peaks[index_max1] = 0  ###max1=0,seek for second max
                    p_max2 = max(p_peaks)
                    index_max2 = p_peaks.index(p_max2)
                    u2 = bins_64[maxv[index_max2]]
                    if p_max2 < 0.1 * p_max1:
                        if (
                            float(probs[i][j][-1]) > 3 * p_max1
                            and float(probs[i][j][-1]) > 0.1
                            and float(probs[i][j][-1]) < 0.8
                        ):
                            breadth.append("last")
                            uu.append(23.0)
                            n_max += 1
                            signal_types.append("p2weak_last")
                            ##gaussian1 fit for 1 main peak
                            popt_solo, pcov_solo = curve_fit(
                                gauss1,
                                x2[:-1],
                                np.array(probs[i][j][:-1]),
                                maxfev=1000000,
                                p0=[0.1, u1, 1],
                            )
                            x_bin = np.arange(x2[0], x2[-1], 0.1)
                            if popt_solo[1] >= 20.75 or popt_solo[2] > 3:
                                continue
                            # print("Gauss1:",popt_solo)
                            y_solo = [
                                gauss1(xx, popt_solo[0], popt_solo[1], popt_solo[2])
                                for xx in x_bin
                            ]
                            if max(y_solo) / float(probs[i][j][-1]) < 0.1:
                                continue
                            ll_solo = round(
                                max(2.3125, popt_solo[1] - 3 * popt_solo[2]), 4
                            )
                            rr_solo = round(
                                min(popt_solo[1] + 3 * popt_solo[2], 21.6875), 4
                            )
                            if popt_solo[1] <= 20.125:
                                n_max += 1
                                breadth.append((ll_solo, rr_solo))
                                uu.append(popt_solo[1])
                                signal_types.append("p2weak_solofit")
                                # print("len(maxv)>=2, p2toolow only p1_maxv but last bin useful",i+1,j+1,p_max1,float(probs[i][j][-1]),ll_solo,rr_solo)
                        else:
                            continue  ###only 1 high peak berfore bins57
                        pass
                    elif (
                        (p_max1 - p_max2) / p_max1 < 0.05
                        and (u2 - u1) < 2
                        and len(maxv) >= 3
                    ):
                        # print("a.3 2maxv close",i+1,j+1,p_max1,p_max2)
                        p_peaks[index_max2] = 0
                        p_max2 = max(p_peaks)
                        index_max2 = p_peaks.index(p_max2)
                        u2 = bins_64[maxv[index_max2]]
                    if p_max2 > 0.1 * p_max1:
                        # print(i+1,j+1,maxv,"probs",p_max1,p_max2,"uuuuuuuuuu",u1,u2)
                        popt, pcov = curve_fit(
                            gaussian2,
                            x2[:-1],
                            np.array(probs[i][j][:-1]),
                            maxfev=1000000,
                            p0=[0.1, 0.1, u1, u2, 1, 1],
                        )
                        x_bin = np.arange(x2[0], x2[-1], 0.1)
                        u1_fit = min(popt[2], popt[3])
                        u2_fit = max(popt[2], popt[3])
                        # print("Params for Gaussian2:",popt,"fitted u",u1_fit,u2_fit)
                        if (popt[-1] > 3 or popt[-1] < -3) and (
                            popt[-2] > 3 or popt[-2] < -3
                        ):  ##两个峰展宽都大于3
                            # print("b.1 too wide",i+1,j+1)
                            continue
                        if u1_fit > 20.125 and u2_fit > 20.125:
                            # print("b.2 fitted too far",i+1,j+1)
                            continue
                        y = [
                            gaussian2(
                                xx,
                                popt[0],
                                popt[1],
                                popt[2],
                                popt[3],
                                popt[4],
                                popt[5],
                            )
                            for xx in x_bin
                        ]
                        fit_maxv = argrelextrema(np.array(y), np.greater)[0]
                        fit_minv = argrelmin(np.array(y))[0]
                        y_maxv = []
                        for e in fit_maxv:
                            y_maxv.append(y[int(e)])
                        if len(fit_maxv) > 1 and len(fit_minv) > 0:
                            # print("Fitted",i+1,j+1,"max_values",y_maxv,"min_values",y[int(fit_minv[0])])
                            y_minv = y[int(fit_minv[0])]
                            mid = round(float(x_bin[fit_minv[0]]), 4)
                            if popt[2] < popt[3]:
                                ll_1 = round(max(2.3125, popt[2] - 3 * popt[2 + 2]), 4)
                                rr_2 = round(min(popt[3] + 3 * popt[3 + 2], 20.125), 4)
                            else:
                                ll_1 = round(max(2.3125, popt[3] - 3 * popt[3 + 2]), 4)
                                rr_2 = round(min(popt[2] + 3 * popt[2 + 2], 20.125), 4)
                                # print(popt[2]+3*popt[2+2])
                            if (
                                abs(min(y_maxv) - max(y_minv, 0)) / min(y_maxv) < 0.05
                            ):  ##platform not 2 peaks!!!

                                if max(y_maxv) / float(probs[i][j][-1]) > 0.1:
                                    n_max += 1
                                    breadth.append((ll_1, rr_2))
                                    uu.append(float((popt[2] + popt[3]) / 2))
                                    signal_types.append("2p_platform_fit1")
                                # print("yyyyyyyyyyyyyyyyyyyy",y_maxv,y_minv)

                            elif x_bin[fit_maxv[1]] - x_bin[fit_maxv[0]] >= 1.5:
                                n_max += 2
                                breadth.append((ll_1, mid))
                                uu.append(min(popt[2], popt[3]))
                                signal_types.append("2p_2fit_1")
                                breadth.append((mid, rr_2))
                                uu.append(max(popt[2], popt[3]))
                                signal_types.append("2p_2fit_2")
                            elif max(y_maxv) / float(probs[i][j][-1]) > 0.1:
                                n_max += 1
                                breadth.append((ll_1, rr_2))
                                uu.append((popt[2] + popt[3]) / 2)
                                signal_types.append("2p_2fit_restin1p")

                        else:
                            if (
                                len(fit_maxv) == 1
                                and max(y_maxv) / float(probs[i][j][-1]) > 0.1
                            ):
                                n_max += 1
                                vector = np.where(np.array(y) > 0.005)[0]
                                breadth.append(
                                    (
                                        round(x_bin[vector[0]], 4),
                                        min(round(x_bin[vector[-1]], 4), 20.125),
                                    )
                                )
                                uu.append(float(x_bin[fit_maxv[0]]))
                                signal_types.append("2p_2fit_in1_over0.005")
                            # print("-----2 set of parameters used------------1 peak fitted---------------")

                        #### last bin counting!!!
                        if float(probs[i][j][-1]) > 0.1 and (
                            sum(probs[i][j][57:63]) < max(probs[i][j]) / 3
                        ):
                            LOGGER.debug(
                                "Sparse last-bin signal at %d:%d", i + 1, j + 1
                            )
                        if (
                            len(breadth) >= 1
                            and breadth[-1] != "last"
                            and breadth[-1][1] < 20.125
                            and float(probs[i][j][-1]) > 0.1
                            and (sum(probs[i][j][57:63]) > max(probs[i][j]) / 3)
                        ):
                            breadth.append((20.125, 21.6875))
                            uu.append(21.0)
                            n_max += 1
                            signal_types.append("2p_bins57-63")
                        if (
                            float(probs[i][j][-1]) > 0.8 * max(probs[i][j])
                            and float(probs[i][j][-1]) > 0.1
                            and float(probs[i][j][-1]) < 0.8
                        ):
                            n_max += 1
                            breadth.append("last")
                            uu.append(23.0)
                            signal_types.append("2p_last")
                elif len(maxv) == 1:  # 有效极大值点就一个
                    u1 = float(bins_64[maxv[0]])
                    p_peak = sum(
                        np.array(probs[i][j][int(maxv[0]) - 1 : int(maxv[0]) + 2])
                    )
                    if p_peak < 0.1:
                        continue
                    ##gaussian1 fit for 1 main peak
                    popt_solo, pcov_solo = curve_fit(
                        gauss1,
                        x2[:-1],
                        np.array(probs[i][j][:-1]),
                        maxfev=1000000,
                        p0=[0.1, u1, 1],
                    )
                    x_bin = np.arange(x2[0], x2[-1], 0.1)
                    if popt_solo[1] >= 20.75 or popt_solo[2] > 3:
                        continue
                    y_solo = [
                        gauss1(xx, popt_solo[0], popt_solo[1], popt_solo[2])
                        for xx in x_bin
                    ]
                    if max(y_solo) / float(probs[i][j][-1]) < 0.1:
                        continue
                    ll_solo = round(max(2.3125, popt_solo[1] - 3 * popt_solo[2]), 4)
                    rr_solo = round(min(popt_solo[1] + 3 * popt_solo[2], 21.6875), 4)
                    if (
                        float(probs[i][j][-1]) > 0.4
                        and float(probs[i][j][-1]) < 0.8
                        and p_peak > 0.1
                    ):
                        n_max += 1
                        breadth.append((ll_solo, rr_solo))
                        uu.append(popt_solo[1])
                        signal_types.append("1pmaxv=1_1fit")
                        LOGGER.debug(
                            "Single fitted peak with last-bin signal at %d:%d",
                            i + 1,
                            j + 1,
                        )
                    elif (
                        popt_solo[1] + popt_solo[2]
                    ) < 21.6875 and p_peak > 0.1:  ##last bin 进入u+1*sigma范围，不好区分最后一列是否是单独一个峰且是的化变化距离很小？
                        n_max += 1
                        breadth.append((ll_solo, rr_solo))
                        uu.append(popt_solo[1])
                        signal_types.append("1pmaxv=1_1fit_u+1sigma")
                    if (
                        float(probs[i][j][-1]) > p_peak
                        and float(probs[i][j][-1]) > 0.1
                        and float(probs[i][j][-1]) < 0.8
                    ):
                        # print("LLLLLLLLLLLLen(maxv)==1, and last bin effective!!!!!!!",i+1,j+1)
                        breadth.append("last")
                        uu.append(23.0)
                        n_max += 1
                        signal_types.append("1pmaxv=1_1fit_last")
                        # print("1_maxv but last bin useful",p_peak,float(probs[i][j][-1]),breadth)
                    else:
                        continue  ###only 1 high peak
            elif len(maxvector) == 1:  ##初始就一个极大值点
                p_maxvector = float(probs[i][j][maxvector[0]])
                p_peak = sum(
                    np.array(probs[i][j][int(maxvector[0]) - 1 : int(maxvector[0]) + 2])
                )
                if (
                    p_maxvector / float(probs[i][j][-1]) > 1
                    or p_maxvector / float(probs[i][j][-1]) < 0.1
                ):  ###one large peak / mainly on lastbin
                    continue
                elif p_peak > 0.1 and float(probs[i][j][-1]) < 0.8:  ##!!!!!
                    u = float(bins_64[maxvector[0]])
                    popt_solo, pcov_solo = curve_fit(
                        gauss1,
                        x2[:-1],
                        np.array(probs[i][j][:-1]),
                        maxfev=1000000,
                        p0=[0.1, u, 1],
                    )
                    if (
                        popt_solo[1] >= 20.75 or popt_solo[2] > 3
                    ):  ##拟合峰值所在距离不能与最后太近
                        continue
                    x_bin = np.arange(x2[0], x2[-1], 0.1)
                    y_solo = [
                        gauss1(xx, popt_solo[0], popt_solo[1], popt_solo[2])
                        for xx in x_bin
                    ]
                    ll_solo = round(max(2.3125, popt_solo[1] - 3 * popt_solo[2]), 4)
                    rr_solo = round(min(popt_solo[1] + 3 * popt_solo[2], 21.6875), 4)
                    if (
                        float(probs[i][j][-1]) > 0.4
                        and float(probs[i][j][-1]) < 0.8
                        and p_peak > 0.1
                    ):
                        n_max += 1
                        breadth.append((ll_solo, rr_solo))
                        uu.append(popt_solo[1])
                        signal_types.append("1pvector=1_1fit")
                        # print("LEN(MAXVECTOR==1) 1_maxv but last bin useful",p_maxvector,float(probs[i][j][-1]),ll_solo,rr_solo)
                    elif (
                        popt_solo[1] + popt_solo[2]
                    ) < 21.6875 and p_peak > 0.1:  ##last bin 进入u+1*sigma范围，不好区分最后一列是否是单独一个峰且是的化变化距离很小？
                        n_max += 1
                        breadth.append((ll_solo, rr_solo))
                        uu.append(popt_solo[1])
                        signal_types.append("1pvector=1_1fit_u+1sigma")
                        # print("11111sigma+u____LEN(MAXVECTOR==1) 1_maxv but last bin useful",p_maxvector,"lastbin",float(probs[i][j][-1]),ll_solo,rr_solo)
                    if (
                        float(probs[i][j][-1]) > p_peak
                        and float(probs[i][j][-1]) > 0.1
                        and float(probs[i][j][-1]) < 0.8
                    ):
                        n_max += 1
                        breadth.append("last")
                        uu.append(23.0)
                        signal_types.append("1pvector=1_1fit_last")

            if n_max >= 2:
                is_apo_signal, peak_location = find_peak(round(d0_apo, 4), breadth)
                apo_predictions.append("True" if is_apo_signal else "False")
                peak_locations_apo.append(peak_location)
                LOGGER.debug(
                    "Alternative signal %d:%d peaks=%s types=%s distance=%.2f",
                    i + 1,
                    j + 1,
                    uu,
                    signal_types,
                    d0_apo,
                )
                pairs.append((i + 1, j + 1))
                peak_counts.append(n_max)
                breadths.append(breadth)
                apo_distances.append(round(d0_apo, 2))
                peak_locations.append(uu)
                if is_apo_signal:
                    apo_flags[i, j] = 1
                    if peak_location != "last":
                        if "last" in breadth:
                            last_signal_count += 1
                        d_range = breadth[peak_location - 1]
                        ll = int((d_range[0] - 2.3125) / 0.3125)
                        rr = int((d_range[1] - 2.3125) / 0.3125)
                        probs_apo[i][j][ll:rr] = [0] * abs(ll - rr)
                        p_peak1 = sum(probs[i][j][ll:rr])
                        probs_apo[i][j] = (1 / (1 - p_peak1)) * probs_apo[i][j]
                    elif peak_location == "last":
                        probs_apo[i][j][-1] = 0
                        probs_apo[i][j] = (
                            1 / (1 - float(probs[i][j][-1]))
                        ) * probs_apo[i][j]
            else:  ##Single peak signal?
                probs_apo[i][j] = probs_apo[i][j]
                # probs_holo[i][j] = probs_holo[i][j]
                if n_max == 1:
                    if breadth[0] == "last":
                        if d0_apo < 21.6875:
                            single_peak_signal_count += 1
                            apo_flags[i, j] = 1
                            last_signal_count += 1
                    else:
                        single_peak_range = breadth[0]
                        if (
                            d0_apo < single_peak_range[0]
                            or d0_apo > single_peak_range[-1]
                        ):
                            single_peak_signal_count += 1
                            apo_flags[i, j] = 1
                            LOGGER.debug(
                                "Single-peak alternative signal %d:%d",
                                i + 1,
                                j + 1,
                            )
    maxh = np.max(entropy)
    minh = np.min(entropy[np.where(entropy > 0)])
    if maxh == minh:
        entropy.fill(0.0)
    else:
        entropy = (maxh - entropy) / (maxh - minh)

    signal_count = len(peak_counts) + single_peak_signal_count
    if signal_count == 0:
        raise RuntimeError("No alternative distance signals were identified")
    p_last = round(last_signal_count / signal_count, 2)
    p_nl = round(signal_count / protein_length, 2)
    LOGGER.info(
        "Identified %d alternative signals (single-peak: %d, multi-peak: %d)",
        signal_count,
        single_peak_signal_count,
        len(peak_counts),
    )

    signals = pd.DataFrame(
        {
            "res": pairs,
            "n_max": peak_counts,
            "d0_apo": apo_distances,
            "apo_peak_loc": peak_locations_apo,
            "Apo_pred": apo_predictions,
            "breadths": breadths,
        }
    )

    return SignalAnalysisResult(
        name=name,
        signals=signals,
        alternative_distance_map=probs_apo,
        alternative_flags=apo_flags,
        entropy=entropy,
        p_last=p_last,
        p_nl=p_nl,
    )


def write_analysis_result(output_dir: Path, result: SignalAnalysisResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p-parameters.txt").write_text(
        f"{result.name}\n{result.p_last}\n{result.p_nl}",
        encoding="utf-8",
    )
    result.signals.to_csv(
        output_dir / f"{result.name}alternative signals_rec0_msaAll_rank005.csv"
    )
    np.save(output_dir / f"{result.name}_flag.npy", result.alternative_flags)
    np.save(
        output_dir / f"{result.name}_altDM.npy",
        result.alternative_distance_map,
    )
    np.save(output_dir / f"Comentropy_{result.name}.npy", result.entropy)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_config(argv)
    result = analyze_signals(config)
    write_analysis_result(config.root / "distfile", result)
    LOGGER.info("Wrote analysis outputs to %s", config.root / "distfile")


if __name__ == "__main__":
    main()
