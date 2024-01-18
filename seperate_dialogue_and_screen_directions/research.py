import os
from collections import defaultdict
import statistics

import matplotlib.pyplot as plt

EXTRACTED_SCRIPT_PATH = 'script_files/scripts_without_tags'
good_script_count = 0
total_script_count = 0
max_freq_cutoff = 0.92
max2_fre_cutoff = 0.3

movies_over_total_freq_threshold = []
for filename in os.listdir(EXTRACTED_SCRIPT_PATH):
    total_script_count += 1
    filepath = os.path.join(EXTRACTED_SCRIPT_PATH, filename)
    whitespace_lengths = defaultdict(lambda:0)
    with open(filepath) as script_file:
        title = next(script_file)
        url = next(script_file)
        for line in script_file.readlines():
            whitespace_length = len(line) - len(line.lstrip())
            whitespace_lengths[whitespace_length] += 1

        filtered_whitespace_length_count_pairs = []
        total_lines = 0
        for length, count in whitespace_lengths.items():
            if count > 5:
                total_lines += count
                filtered_whitespace_length_count_pairs.append((length, count))

        whitepsace_frequency_pairs = list(map(lambda x: (x[0], x[1]/total_lines), filtered_whitespace_length_count_pairs))
        average_whitespace = sum((whitespace * freq) for whitespace, freq in whitepsace_frequency_pairs)
        most_freq = max(whitepsace_frequency_pairs, key= lambda x: x[1])
        if most_freq[1] > max_freq_cutoff:
            total_script_count -= 1
            continue

        whitepsace_frequency_pairs.remove(most_freq)
        second_most_freq = max(whitepsace_frequency_pairs, key= lambda x: x[1])

        if most_freq[1] + second_most_freq[1] > 0.3:
            movies_over_total_freq_threshold.append((title.strip(), most_freq[1] + second_most_freq[1], average_whitespace))
movies_over_total_freq_threshold.sort(key= lambda x: x[1])
print(movies_over_total_freq_threshold[:10])
print(len(movies_over_total_freq_threshold), total_script_count)

