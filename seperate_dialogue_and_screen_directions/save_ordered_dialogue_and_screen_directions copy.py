import os
from collections import defaultdict


SCRIPTS_WITHOUT_TAGS_PATH = 'script_files/scripts_without_tags'
DESTINATION_PATH = 'script_files/ordered_dialogue_and_screen_directions'

MOST_FREQ_WHITESPACE_FREQ_UPPER_LIMIT = 0.92
TWO_MOST_FREQUENT_WHITESAPCE_LOWER_LIMIT = 0.3
ABSOLUTE_WHITESPACE_COUNT_LOWER_LIMIT = 5

if not os.path.exists(DESTINATION_PATH):
    os.mkdir(DESTINATION_PATH)

for filename in os.listdir(SCRIPTS_WITHOUT_TAGS_PATH):
    filepath = os.path.join(SCRIPTS_WITHOUT_TAGS_PATH, filename)
    whitespace_length_to_lines = defaultdict(lambda: [])
    with open(filepath) as script_file:
        title = next(script_file)
        url = next(script_file)
        for line in script_file.readlines():
            whitespace_length = len(line) - len(line.lstrip())
            whitespace_length_to_lines[whitespace_length].append(line.strip())

    filtered_whitespace_length_lines_pairs = []
    total_filtered_lines = 0
    total_filtered_whitespace = 0
    for whitespace, lines in whitespace_length_to_lines.items():
        if len(lines) > ABSOLUTE_WHITESPACE_COUNT_LOWER_LIMIT:
            filtered_whitespace_length_lines_pairs.append((whitespace, lines))
            total_filtered_lines += len(lines)
            total_filtered_whitespace += len(lines) * whitespace
    average_whitespace = total_filtered_whitespace / total_filtered_lines

    longest_whitespace_length_line_pair = filtered_whitespace_length_lines_pairs[0]
    second_longest_whitespace_length_line_pair = filtered_whitespace_length_lines_pairs[0]
    for pair in filtered_whitespace_length_lines_pairs:
        if len(pair[1]) > len(longest_whitespace_length_line_pair[1]):
            second_longest_whitespace_length_line_pair = longest_whitespace_length_line_pair
            longest_whitespace_length_line_pair = pair
        elif len(pair[1]) > len(second_longest_whitespace_length_line_pair[1]):
            second_longest_whitespace_length_line_pair = pair
    
    if len(longest_whitespace_length_line_pair[1]) / total_filtered_lines >= MOST_FREQ_WHITESPACE_FREQ_UPPER_LIMIT:
        continue

    if (len(longest_whitespace_length_line_pair[1]) 
        + len(second_longest_whitespace_length_line_pair[1])) / total_filtered_lines <= TWO_MOST_FREQUENT_WHITESAPCE_LOWER_LIMIT:
        continue

    labeled_data = []
    with open(filepath) as script_file:
            title = next(script_file)
            url = next(script_file)
            for line in script_file.readlines():
                whitespace_length = len(line) - len(line.lstrip())
                if whitespace_length > average_whitespace:
                    labeled_data.append('0 ' + line.strip())
                else:
                    labeled_data.append('1 ' + (line.strip()))

    with open(os.path.join(DESTINATION_PATH, filename), 'w') as destination_file:
        destination_file.writelines(map(lambda line: line + os.linesep, labeled_data))

        



        
        
