#!/usr/bin/env python
"""Build the ASVspoof 5 and In-the-Wild demo subsets used by the deepfake task.

Companion to prepare_asvspoof_la_subset.py. Same rules: nothing here is
committed (`Backend/.gitignore` ignores `/data`), the selection is
deterministic (fixed seed) and balanced, and each subset gets a protocol.txt
in the ASVspoof 2019 CM format (`SPEAKER FILE - SYSTEM KEY`) so the backend
reads all three datasets the same way.

In-the-Wild (Müller et al., Interspeech 2022; https://deepfake-total.com/in_the_wild):

    python scripts/prepare_deepfake_eval_subsets.py in-the-wild \\
        --source /path/to/release_in_the_wild

    -> data/deepfake/in_the_wild/wav/*.wav + protocol.txt
    100 bona fide + 100 spoof, each spread round-robin across speakers. The
    release names no generator, so spoof clips get the system id
    "unattributed".

ASVspoof 5 (Wang et al. 2025; https://zenodo.org/records/14498691, ODC-By):

    python scripts/prepare_deepfake_eval_subsets.py asvspoof5 \\
        --protocols /path/to/ASVspoof5_protocols.tar.gz \\
        --audio /path/to/flac_E_aa.tar [more flac_E_*.tar ...]

    -> data/deepfake/asvspoof5/flac/E_*.flac + protocol.txt
    100 bona fide + 100 spoof spread across attacks (A17-A32), chosen from
    whichever clips the tar(s) actually contain. A tar cut short on purpose
    (e.g. only its first 600 MB downloaded) works: every complete member
    before the cut is usable, and the truncated last one is skipped.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEST_ROOT = BACKEND_DIR / "data" / "deepfake"
DEFAULT_SEED = 20260818

# (speaker, file_id, system_id, key) -- the ASVspoof 2019 CM protocol fields.
Entry = tuple[str, str, str, str]


def pick_round_robin(entries: list[Entry], target: int, group_index: int, rng: random.Random) -> list[Entry]:
    """Round-robin across groups (attack or speaker) so none dominates."""
    groups: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        groups[entry[group_index]].append(entry)
    for members in groups.values():
        rng.shuffle(members)

    chosen: list[Entry] = []
    names = sorted(groups)
    while len(chosen) < target and any(groups[name] for name in names):
        for name in names:
            if len(chosen) >= target:
                break
            if groups[name]:
                chosen.append(groups[name].pop())
    return chosen


def select_balanced(entries: list[Entry], bonafide: int, spoof: int, seed: int, spoof_group: int) -> list[Entry]:
    rng = random.Random(seed)
    genuine = [entry for entry in entries if entry[3] == "bonafide"]
    fake = [entry for entry in entries if entry[3] == "spoof"]
    print(f"Candidates: {len(genuine)} bona fide, {len(fake)} spoof")
    chosen = pick_round_robin(genuine, bonafide, 0, rng) + pick_round_robin(fake, spoof, spoof_group, rng)
    if sum(e[3] == "bonafide" for e in chosen) < bonafide or sum(e[3] == "spoof" for e in chosen) < spoof:
        print("WARNING: not enough candidates for a full balanced subset", file=sys.stderr)
    # Sorted by file id so the on-disk order says nothing about the label.
    return sorted(chosen, key=lambda entry: entry[1])


def reset_dest(dest: Path, audio_subdir: str) -> Path:
    audio_dir = dest / audio_subdir
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)
    return audio_dir


def write_protocol(dest: Path, entries: list[Entry]) -> None:
    lines = [f"{speaker} {file_id} - {system} {key}" for speaker, file_id, system, key in entries]
    (dest / "protocol.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(dest: Path, audio_dir: Path, entries: list[Entry]) -> None:
    composition: dict[str, int] = defaultdict(int)
    for _speaker, _file, system, key in entries:
        composition["bonafide" if key == "bonafide" else system] += 1
    files = [path for path in audio_dir.iterdir() if path.is_file()]
    print(f"\nWrote {len(files)} clips to {audio_dir} ({sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")
    print(f"Wrote protocol to {dest / 'protocol.txt'}")
    print("Composition:", dict(sorted(composition.items())))


# ── In-the-Wild ──────────────────────────────────────────────────────────────


def prepare_in_the_wild(args: argparse.Namespace) -> int:
    meta = args.source / "meta.csv"
    if not meta.is_file():
        raise SystemExit(f"meta.csv not found under {args.source}")

    entries: list[Entry] = []
    with open(meta, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip().lower()
            key = "bonafide" if label in {"bona-fide", "bonafide"} else "spoof" if label == "spoof" else None
            if key is None:
                continue
            # Protocol fields are whitespace-separated; speakers are full names.
            speaker = "_".join(row["speaker"].split()) or "unknown"
            file_id = Path(row["file"]).stem
            entries.append((speaker, file_id, "-" if key == "bonafide" else "unattributed", key))

    # Spoofs have no attack id here, so spread them across speakers too.
    chosen = select_balanced(entries, args.bonafide, args.spoof, args.seed, spoof_group=0)

    dest = DEST_ROOT / "in_the_wild"
    audio_dir = reset_dest(dest, "wav")
    written: list[Entry] = []
    for entry in chosen:
        source = args.source / f"{entry[1]}.wav"
        if not source.is_file():
            print(f"WARNING: missing {source}", file=sys.stderr)
            continue
        shutil.copy2(source, audio_dir / source.name)
        written.append(entry)
    write_protocol(dest, written)
    attribution = args.source / "attribution.txt"
    if attribution.is_file():
        shutil.copy2(attribution, dest / "attribution.txt")
    report(dest, audio_dir, written)
    return 0


# ── ASVspoof 5 ───────────────────────────────────────────────────────────────


def read_asvspoof5_protocol(protocols_tgz: Path, partition: str) -> dict[str, Entry]:
    """{file_id: entry} from ASVspoof5.<partition>.track_1.tsv inside the tarball.

    Columns: SPEAKER FILE GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
    """
    wanted = f"ASVspoof5.{partition}.track_1.tsv"
    with tarfile.open(protocols_tgz, "r:gz") as archive:
        member = next((m for m in archive.getmembers() if Path(m.name).name == wanted), None)
        if member is None:
            raise SystemExit(f"{wanted} not found in {protocols_tgz}")
        text = archive.extractfile(member).read().decode("utf-8")

    entries: dict[str, Entry] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 9:
            continue
        speaker, file_id, attack, key = fields[0], fields[1], fields[7], fields[8]
        entries[file_id] = (speaker, file_id, "-" if key == "bonafide" else attack, key)
    return entries


def iter_complete_members(tar_path: Path):
    """Yield (archive, member) for every complete .flac in a possibly truncated tar."""
    try:
        with tarfile.open(tar_path, "r:") as archive:
            while True:
                try:
                    member = archive.next()
                except (tarfile.ReadError, EOFError, OSError):
                    return  # cut short mid-header
                if member is None:
                    return
                if member.isfile() and member.name.endswith(".flac"):
                    if member.offset_data + member.size > tar_path.stat().st_size:
                        return  # cut short mid-file
                    yield archive, member
    except tarfile.ReadError as error:
        raise SystemExit(f"{tar_path} is not a readable tar ({error}). Is it empty?") from error


def prepare_asvspoof5(args: argparse.Namespace) -> int:
    protocol = read_asvspoof5_protocol(args.protocols, args.partition)

    available: dict[str, tuple[Path, str]] = {}
    for tar_path in args.audio:
        if tar_path.stat().st_size == 0:
            raise SystemExit(f"{tar_path} is 0 bytes; the download did not complete.")
        count = 0
        for _archive, member in iter_complete_members(tar_path):
            file_id = Path(member.name).stem
            if file_id in protocol:
                available[file_id] = (tar_path, member.name)
                count += 1
        print(f"{tar_path.name}: {count} complete clips listed in the {args.partition} protocol")

    entries = [protocol[file_id] for file_id in available]
    chosen = select_balanced(entries, args.bonafide, args.spoof, args.seed, spoof_group=2)
    wanted = {entry[1] for entry in chosen}

    dest = DEST_ROOT / "asvspoof5"
    audio_dir = reset_dest(dest, "flac")
    for tar_path in args.audio:
        for archive, member in iter_complete_members(tar_path):
            file_id = Path(member.name).stem
            if file_id not in wanted or available[file_id][0] != tar_path:
                continue
            with archive.extractfile(member) as source, open(audio_dir / f"{file_id}.flac", "wb") as target:
                shutil.copyfileobj(source, target)
    write_protocol(dest, chosen)
    report(dest, audio_dir, chosen)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bonafide", type=int, default=100)
    parser.add_argument("--spoof", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    commands = parser.add_subparsers(dest="dataset", required=True)

    itw = commands.add_parser("in-the-wild")
    itw.add_argument("--source", required=True, type=Path, help="The extracted release_in_the_wild directory")
    itw.set_defaults(run=prepare_in_the_wild)

    asv5 = commands.add_parser("asvspoof5")
    asv5.add_argument("--protocols", required=True, type=Path, help="ASVspoof5_protocols.tar.gz")
    asv5.add_argument("--audio", required=True, type=Path, nargs="+", help="flac_E_*.tar file(s), complete or cut short")
    asv5.add_argument("--partition", choices=["eval", "dev"], default="eval")
    asv5.set_defaults(run=prepare_asvspoof5)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
