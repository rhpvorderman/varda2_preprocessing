import argparse
import shutil
import tempfile
import os.path
import heapq
import typing
from os.path import commonprefix
from typing import Iterable, Iterator

from natsort import natsorted, natsort_key

from pysam import VariantFile

# Strip the prefix from a string
def remove_prefix(text, pref):
    if text.startswith(pref):
        return text[len(pref):]
    return text


# Strip the suffix from a string
def remove_suffix(text, suf):
    if text.endswith(suf):
        return text[:len(text) - len(suf)]
    return text


# Return the common suffix of a list of strings
def common_suffix(entries):
    suf = commonprefix([entry[::-1] for entry in entries])
    return suf[::-1]


def trim(start, end, ref, alt):
    # Find the common prefix of the ref and the alt
    prefix = commonprefix([ref, alt])
    prefix_len = len(prefix)

    # Remove common prefix from both ref and alt
    ref_remain = remove_prefix(ref, prefix)
    alt_remain = remove_prefix(alt, prefix)

    # Now remove common suffix from ref and ald
    suffix = common_suffix([ref_remain, alt_remain])
    suffix_len = len(suffix)

    # Derive inserted string from the remaining alt string
    # If there is no inserted string (len==0), set it to '.'
    inserted = remove_suffix(alt_remain, suffix)

    # Trim start and end positions
    trim_start = start + prefix_len
    trim_end = end - suffix_len

    return trim_start, trim_end, inserted


def read_vcf(filename):
    with VariantFile(filename) as vcf_file:
        # Assume single sample VCF files for now (and forever)
        assert len(vcf_file.header.samples) == 1

        phase_sets = {}
        active_ps_id = 0
        for rec in vcf_file.fetch():
            entry = rec.samples[0]

            ps_id = entry.get("PS")
            if entry.phased:
                ps_id = ps_id or active_ps_id
            if not ps_id:
                ps_id = rec.pos
                active_ps_id = ps_id if entry.phased else 0

            key = rec.chrom, ps_id
            if key not in phase_sets:
                phase_sets[key] = [[] for _ in entry["GT"]]

            for idx, gt in enumerate(entry["GT"]):
                if gt:
                    alt = rec.alts[gt - 1]
                    if "<" in alt or ">" in alt:
                        raise ValueError("Cannot deal with symbolic alleles")

                    if alt == "*":
                        continue

                    variant = trim(rec.start, rec.stop, rec.ref, alt)
                    phase_sets[key][idx].append(variant)

        return {key: value for key, value in phase_sets.items() if any(value)}


class VardaEntry(typing.NamedTuple):
    chrom: str
    start: int
    end: int
    ploidy: int
    ps_id: int
    length: int
    sequence: str

    def to_line(self) -> str:
        return (f"{self.chrom}\t{self.start}\t{self.end}\t{self.ploidy}\t"
                f"{self.ps_id}\t{self.length}\t{self.sequence}\n")

    @classmethod
    def from_line(cls, line: str):
        chrom, start, end, ploidy, ps_id, length, sequence = line.strip().split("\t")
        return cls(
            chrom, int(start), int(end), int(ploidy), int(ps_id), int(length),
            sequence
        )


def to_varda(phase_sets):
    ps_ids = {}
    for (chrom, ps), alleles in phase_sets.items():
        if chrom not in ps_ids:
            ps_ids[chrom] = {(0, 0): 0}

        homozygotes = set.intersection(*(set(allele) for allele in alleles))
        heterozygotes = [set(allele) - homozygotes for allele in alleles]
        unique_variants = set.union(*(set(allele) for allele in heterozygotes))

        for start, end, sequence in homozygotes:
            yield VardaEntry(chrom, start, end, len(alleles), -1, len(sequence), sequence or ".")

        if len(unique_variants) == 1:
            # We are now a heterozygous unphased variant
            start, end, sequence = unique_variants.pop()
            ploidy = sum(len(allele) for allele in heterozygotes)
            yield VardaEntry(chrom, start, end, ploidy, ps_ids[chrom][(0, 0)], len(sequence), sequence or ".")
        else:
            for idx, allele in enumerate(heterozygotes):
                for variant in allele:
                    if (ps, idx) not in ps_ids[chrom]:
                        ps_ids[chrom][(ps, idx)] = len(ps_ids[chrom])

                    start, end, sequence = variant
                    yield VardaEntry(chrom, start, end, 1, ps_ids[chrom][(ps, idx)], len(sequence), sequence or ".")


def _tmp_file_to_varda_entry(tmp_file: str) -> Iterator[VardaEntry]:
    with open(tmp_file, "r") as tmp:
        for line in tmp:
            yield VardaEntry.from_line(line)


def batched_natural_sorting(
        varda_entries: Iterable[VardaEntry],
        max_batch_size: int = 500_000,  # Roughly 1 GiB of memory usage.
        work_folder: str = os.path.curdir
) -> Iterator[VardaEntry]:
    batches_folder = tempfile.mkdtemp(dir=work_folder)
    os.makedirs(batches_folder, exist_ok=True)
    filenames = []
    try:
        while True:
            batch: list[VardaEntry] = []
            for _, entry in zip(range(max_batch_size), varda_entries):
                batch.append(entry)
            sorted_batch = natsorted(batch)
            del batch  # Immediately cleanup memory here.
            if len(sorted_batch) < max_batch_size:
                # Keep the last batch in memory rather than writing to disk
                break
            # Write sorted batch to disk.
            with tempfile.NamedTemporaryFile(dir=batches_folder, delete=False, mode="wt") as tmp:
                filenames.append(tmp.name)
                for entry in sorted_batch:
                    tmp.write(entry.to_line())
            del sorted_batch
        if len(filenames) == 0:
            yield from sorted_batch
            return

        iterators = [_tmp_file_to_varda_entry(filename) for filename in filenames]
        iterators.append(iter(sorted_batch))
        yield from heapq.merge(*iterators, key=natsort_key)
    finally:
        shutil.rmtree(batches_folder)


def main():
    parser = argparse.ArgumentParser(description="Read phase sets from single sample VCF 4.3 file.")
    parser.add_argument("filename", help="VCF file")
    args = parser.parse_args()

    phase_sets = read_vcf(args.filename)
    for entry in batched_natural_sorting(to_varda(phase_sets)):
        print(*entry, sep="\t")


if __name__ == "__main__":
    main()
