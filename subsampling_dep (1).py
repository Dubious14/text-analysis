#!/usr/bin/env python3
import os
import time
import random
import argparse
import shelve
import tempfile

CHUNK_SIZE = 50000
NOTIFY_INTERVAL = 60


def count_frequencies_to_file(input_path, freq_raw_path):
    with open(input_path, 'r') as infile, open(freq_raw_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) == 2:
                word, context = parts
                outfile.write(f"{word}|||{context}\t1\n")


def sort_frequency_file(freq_raw_path, freq_sorted_path):
    print("Sorting frequency file with external 'sort'...")
    os.system(f"sort {freq_raw_path} > {freq_sorted_path}")


def merge_sorted_frequencies(freq_sorted_path, freq_merged_path):
    print("Merging sorted frequencies...")
    with open(freq_sorted_path, 'r') as infile, open(freq_merged_path, 'w') as outfile:
        last_key = None
        current_count = 0

        for line in infile:
            key, count = line.strip().split('\t')
            count = int(count)
            if key == last_key:
                current_count += count
            else:
                if last_key is not None:
                    outfile.write(f"{last_key}\t{current_count}\n")
                last_key = key
                current_count = count

        if last_key is not None:
            outfile.write(f"{last_key}\t{current_count}\n")


def load_frequencies_to_shelve(freq_merged_path, shelve_path):
    print("Loading frequencies into shelve DB...")
    with shelve.open(shelve_path) as db, open(freq_merged_path, 'r') as infile:
        for line in infile:
            key, count = line.strip().split('\t')
            db[key] = int(count)


def compute_discard_probability(freq, sample):
    gamma = sample / freq
    discard_prob = 1 - (gamma ** 0.5 + gamma ** 2 + gamma ** 3)
    return max(0.0, min(1.0, discard_prob))


def subsample_with_shelve(input_path, output_path, shelve_path, threshold):
    print("Starting final subsampling pass...")
    total_written = 0
    start_time = last_notify = time.time()

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile, shelve.open(shelve_path, 'r') as freq_db:
        for i, line in enumerate(infile):
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            word, context = parts
            key = f"{word}|||{context}"
            freq = freq_db.get(key, 1)

            discard_prob = compute_discard_probability(freq, threshold)
            if random.random() > discard_prob:
                outfile.write(f"{word} {context}\n")
                total_written += 1

            now = time.time()
            if now - last_notify >= NOTIFY_INTERVAL:
                print(f"[{int(now - start_time)}s] Written: {total_written:,} lines")
                last_notify = now

    print(f"Subsampling complete. Total lines written: {total_written:,}")


def clean_temp_files(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def main(input_file, output_file, threshold):
    with tempfile.TemporaryDirectory() as tmpdir:
        freq_raw_path = os.path.join(tmpdir, "freq_raw.txt")
        freq_sorted_path = os.path.join(tmpdir, "freq_sorted.txt")
        freq_merged_path = os.path.join(tmpdir, "freq_merged.txt")
        shelve_path = os.path.join(tmpdir, "freq_db")

        print("Pass 1: Counting raw frequencies...")
        count_frequencies_to_file(input_file, freq_raw_path)

        print("Pass 2: Sorting raw frequency data...")
        sort_frequency_file(freq_raw_path, freq_sorted_path)

        print("Pass 3: Merging counts...")
        merge_sorted_frequencies(freq_sorted_path, freq_merged_path)

        print("Pass 4: Loading merged counts into disk-based DB...")
        load_frequencies_to_shelve(freq_merged_path, shelve_path)

        print("Pass 5: Subsampling line by line...")
        subsample_with_shelve(input_file, output_file, shelve_path, threshold)

        print("Cleaning up temporary files...")
        clean_temp_files(freq_raw_path, freq_sorted_path, freq_merged_path)

        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAM-safe subsampling for massive .contexts files")
    parser.add_argument("input_file", help="Path to input .contexts file")
    parser.add_argument("output_file", help="Path to output file")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Subsampling threshold (default: 0.001)")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.threshold)
