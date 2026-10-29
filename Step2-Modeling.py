import json
import multiprocessing
import os
import tempfile
import time
from argparse import Namespace
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import numpy as np
from pyrosetta import (
    MoveMap,
    Pose,
    RepeatMover,
    ScoreFunction,
    SwitchResidueTypeSetMover,
    create_score_function,
    init,
    pose_from_pdb,
    rosetta,
)
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

from arguments_AF2 import get_args
from dij_magH_noroll_Utils_cycle_ros_dist_npy import (
    add_restraints,
    generate_distance_restraints,
    read_fasta_sequence,
    remove_clash,
)

NCYCLES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
U_VALUES = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
]
WORKER_MEMORY_BYTES = 2 * 1024**3
NumberType = TypeVar("NumberType", int, float)
__all__ = ["main"]


@dataclass(frozen=True, slots=True)
class SignalParameters:
    name: str
    pdb_id: str
    p_last: float
    p_nl: float


@dataclass(frozen=True, slots=True)
class ModelTask:
    ncycle: int
    restraint_weight: float
    steps: int
    mode: int
    pcut: float
    use_orient: bool
    fastrelax: bool
    work_dir: str
    output_prefix: str
    seed: int


@dataclass(frozen=True, slots=True)
class ModelResult:
    ncycle: int
    restraint_weight: float
    seed: int
    output: Path
    elapsed: float


@dataclass(slots=True)
class WorkerContext:
    params: dict[str, Any]
    pdb_id: str
    p_last: float
    p_nl: float
    entropy: Any
    distance_map: Any
    distance_flags: Any
    sequence: str
    template_pose: Any
    sf_vdw: Any
    min_mover: Any
    min_mover1: Any
    min_mover_cart: Any


_WORKER_CONTEXT: WorkerContext | None = None


def load_signal_parameters(path: Path) -> SignalParameters:
    with path.open() as f:
        lines = [line.strip() for line in f]

    if len(lines) < 3:
        raise ValueError(f"Expected at least 3 lines in signal parameters: {path}")

    name = lines[0]
    return SignalParameters(
        name=name,
        pdb_id=name.removeprefix("alt-"),
        p_last=float(lines[1]),
        p_nl=float(lines[2]),
    )


def parse_number_list(
    raw_value: str | None,
    converter: Callable[[str], NumberType],
    option_name: str,
) -> list[NumberType] | None:
    if raw_value is None:
        return None

    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(converter(item))
        except ValueError as error:
            raise ValueError(f"Invalid value for {option_name}: {item}") from error

    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return list(dict.fromkeys(values))


def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def resolve_worker_count(requested_jobs: int, task_count: int) -> int:
    if requested_jobs < 0:
        raise ValueError("--jobs must be zero or a positive integer")

    cpu_limit = available_cpu_count()
    memory = available_memory_bytes()
    memory_limit = max(1, memory // WORKER_MEMORY_BYTES) if memory else cpu_limit

    if requested_jobs == 0:
        workers = min(cpu_limit, memory_limit)
        selection = "auto"
    else:
        workers = requested_jobs
        selection = "requested"

    worker_count = max(1, min(task_count, workers))
    print(
        f"Parallel mode ({selection}): {worker_count} workers "
        f"(CPU limit: {cpu_limit}, memory limit: {memory_limit}, tasks: {task_count})"
    )
    return worker_count


def initialize_worker(script_dir: str, fasta_path: str, distfile_dir: str) -> None:
    init(
        "-hb_cen_soft -relax:default_repeats 5 "
        "-default_max_cycles 200 -out:level 100"
    )

    params_path = Path(script_dir) / "data/params.json"
    with params_path.open() as jsonfile:
        params = json.load(jsonfile)

    signal_parameters = load_signal_parameters(Path(distfile_dir) / "p-parameters.txt")
    distfile_dir = Path(distfile_dir)
    entropy = np.load(distfile_dir / f"Comentropy_{signal_parameters.name}.npy")
    distance_map = np.load(distfile_dir / f"{signal_parameters.name}_altDM.npy")
    distance_flags = np.load(distfile_dir / f"{signal_parameters.name}_flag.npy")
    sequence = read_fasta_sequence(fasta_path)
    params["seq"] = sequence

    sf = ScoreFunction()
    sf.add_weights_from_file(script_dir + "/data/scorefxn.wts")

    sf1 = ScoreFunction()
    sf1.add_weights_from_file(script_dir + "/data/scorefxn1.wts")

    sf_vdw = ScoreFunction()
    sf_vdw.add_weights_from_file(script_dir + "/data/scorefxn_vdw.wts")

    sf_cart = ScoreFunction()
    sf_cart.add_weights_from_file(script_dir + "/data/scorefxn_cart.wts")

    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)

    min_mover = MinMover(movemap, sf, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover1 = MinMover(movemap, sf1, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover_cart = MinMover(
        movemap, sf_cart, "lbfgs_armijo_nonmonotone", 0.0001, True
    )
    min_mover_cart.cartesian(True)

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = WorkerContext(
        params=params,
        pdb_id=signal_parameters.pdb_id,
        p_last=signal_parameters.p_last,
        p_nl=signal_parameters.p_nl,
        entropy=entropy,
        distance_map=distance_map,
        distance_flags=distance_flags,
        sequence=sequence,
        template_pose=pose_from_pdb(
            str(Path(script_dir) / "pdb" / f"{signal_parameters.pdb_id}.pdb")
        ),
        sf_vdw=sf_vdw,
        min_mover=min_mover,
        min_mover1=min_mover1,
        min_mover_cart=min_mover_cart,
    )


def run_model_task(task: ModelTask) -> ModelResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker was not initialized")

    context = _WORKER_CONTEXT
    n = task.ncycle
    u = task.restraint_weight
    steps = task.steps
    start_time = time.monotonic()
    rosetta.basic.random.init_random_generators(task.seed, "mt19937")

    params = context.params.copy()
    params["PCUT"] = task.pcut
    params["USE_ORIENT"] = task.use_orient
    params["seq"] = context.sequence
    sequence = context.sequence
    context.min_mover.max_iter(steps)
    context.min_mover1.max_iter(steps)
    context.min_mover_cart.max_iter(steps)

    print(f"Starting ncycle={n}, U={u}, seed={task.seed}")
    with tempfile.TemporaryDirectory(
        prefix=f"altconf-{n}-{u}-", dir=task.work_dir
    ) as tmpdir:
        params["TDIR"] = tmpdir

        pose = Pose()
        pose.assign(context.template_pose)
        switch = SwitchResidueTypeSetMover("centroid")
        switch.apply(pose)

        # Mutate GLY to ALA during centroid minimization.
        for residue_index, residue in enumerate(sequence):
            if residue == "G":
                mutator = rosetta.protocols.simple_moves.MutateResidue(
                    residue_index + 1, "ALA"
                )
                mutator.apply(pose)

        restraints = generate_distance_restraints(
            context.distance_map,
            u,
            context.distance_flags,
            context.entropy,
            tmpdir,
            params,
            pose,
            context.p_last,
            context.p_nl,
        )

        repeat_mover = RepeatMover(context.min_mover, n)
        if task.mode == 0:
            add_restraints(pose, restraints, n, u, 1, 12, params)
            repeat_mover.apply(pose)
            context.min_mover_cart.apply(pose)
            remove_clash(context.sf_vdw, context.min_mover1, pose)

            add_restraints(pose, restraints, n, u, 12, 24, params)
            repeat_mover.apply(pose)
            context.min_mover_cart.apply(pose)
            remove_clash(context.sf_vdw, context.min_mover1, pose)

            add_restraints(pose, restraints, n, u, 24, len(sequence), params)
            repeat_mover.apply(pose)
            context.min_mover_cart.apply(pose)
            remove_clash(context.sf_vdw, context.min_mover1, pose)

        elif task.mode == 1:
            add_restraints(pose, restraints, n, u, 3, 24, params)
            repeat_mover.apply(pose)
            context.min_mover_cart.apply(pose)
            remove_clash(context.sf_vdw, context.min_mover1, pose)

            add_restraints(pose, restraints, n, u, 24, len(sequence), params)
            repeat_mover.apply(pose)
            context.min_mover_cart.apply(pose)
            remove_clash(context.sf_vdw, context.min_mover1, pose)

        elif task.mode == 2:
            add_restraints(pose, restraints, n, u, 1, len(sequence), params)
            repeat_mover.apply(pose)
            context.min_mover_cart.apply(pose)
            remove_clash(context.sf_vdw, context.min_mover1, pose)

        if task.fastrelax:
            sf_fa = create_score_function("ref2015")
            sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
            sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
            sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 1)

            relax_movemap = MoveMap()
            relax_movemap.set_bb(True)
            relax_movemap.set_chi(True)
            relax_movemap.set_jump(True)

            relax = rosetta.protocols.relax.FastRelax()
            relax.set_scorefxn(sf_fa)
            relax.max_iter(steps)
            relax.dualspace(True)
            relax.set_movemap(relax_movemap)

            pose.remove_constraints()
            switch = SwitchResidueTypeSetMover("fa_standard")
            switch.apply(pose)

            params["PCUT"] = 0.15
            add_restraints(pose, restraints, n, u, 1, len(sequence), params, True)
            relax.apply(pose)

            # Restore GLY after full-atom refinement.
            for residue_index, residue in enumerate(sequence):
                if residue == "G":
                    mutator = rosetta.protocols.simple_moves.MutateResidue(
                        residue_index + 1, "GLY"
                    )
                    mutator.apply(pose)

        else:
            # Restore GLY when no full-atom refinement is requested.
            for residue_index, residue in enumerate(sequence):
                if residue == "G":
                    mutator = rosetta.protocols.simple_moves.MutateResidue(
                        residue_index + 1, "GLY"
                    )
                    mutator.apply(pose)

        output_path = Path(f"{task.output_prefix}{n}_U{u}.pdb")
        pose.dump_pdb(str(output_path))

    elapsed = time.monotonic() - start_time
    print(f"Finished ncycle={n}, U={u}: {output_path} ({elapsed:.1f}s)")
    return ModelResult(
        ncycle=n,
        restraint_weight=u,
        seed=task.seed,
        output=output_path,
        elapsed=elapsed,
    )


def build_tasks(
    args: Namespace,
    ncycles: list[int],
    u_values: list[float],
) -> list[ModelTask]:
    tasks = []
    for n in ncycles:
        for u in u_values:
            task_index = len(tasks)
            tasks.append(
                ModelTask(
                    ncycle=n,
                    restraint_weight=u,
                    steps=args.steps,
                    mode=args.mode,
                    pcut=args.pcut,
                    use_orient=args.use_orient,
                    fastrelax=args.fastrelax,
                    work_dir=args.wdir,
                    output_prefix=args.OUT,
                    seed=(args.seed + task_index) % (2**31 - 1),
                )
            )
    return tasks


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    params_path = script_dir / "data/params.json"
    with params_path.open() as jsonfile:
        params = json.load(jsonfile)

    args = get_args(params)
    print(args)

    ncycles = parse_number_list(args.ncycles, int, "--ncycles") or NCYCLES
    u_values = parse_number_list(args.u_values, float, "--u-values") or U_VALUES
    if any(n < 1 for n in ncycles):
        raise ValueError("All --ncycles values must be positive")
    if any(u <= 0 for u in u_values):
        raise ValueError("All --u-values values must be positive")

    fasta_path = str(Path(args.FASTA).resolve())
    distfile_dir = str(Path("./distfile").resolve())
    signal_parameters = load_signal_parameters(Path(distfile_dir) / "p-parameters.txt")
    print("name", signal_parameters.name, "PDB", signal_parameters.pdb_id)

    required_inputs = [
        Path(fasta_path),
        script_dir / "pdb" / f"{signal_parameters.pdb_id}.pdb",
        Path(distfile_dir) / f"Comentropy_{signal_parameters.name}.npy",
        Path(distfile_dir) / f"{signal_parameters.name}_altDM.npy",
        Path(distfile_dir) / f"{signal_parameters.name}_flag.npy",
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(missing_inputs)
        )
    if not Path(args.wdir).is_dir():
        raise FileNotFoundError(f"Temporary directory does not exist: {args.wdir}")

    Path(args.OUT).parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args, ncycles, u_values)
    worker_count = resolve_worker_count(args.jobs, len(tasks))

    if worker_count == 1:
        initialize_worker(str(script_dir), fasta_path, distfile_dir)
        for task in tasks:
            run_model_task(task)
        return

    process_context = multiprocessing.get_context("spawn")
    completed = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=initialize_worker,
        initargs=(str(script_dir), fasta_path, distfile_dir),
    ) as executor:
        futures = {executor.submit(run_model_task, task): task for task in tasks}
        try:
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] ncycle={result.ncycle}, "
                    f"U={result.restraint_weight} -> {result.output} "
                    f"({result.elapsed:.1f}s, seed={result.seed})"
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise


if __name__ == "__main__":
    main()
