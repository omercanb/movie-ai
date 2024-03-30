import os

def get_dialogue_and_stage_directions(path):
    dialogue = []
    stage_directions = []
    with open(path) as f:
        current_type = dialogue
        for line in f.readlines():
            if line == '\n':
                current_type = stage_directions
                continue
            current_type.append(line.strip())
    return dialogue, stage_directions


def get_three_line_dialogue_and_stage_directions(path):
    dialogue, stage_directions = get_dialogue_and_stage_directions(path)
    for lines in (dialogue, stage_directions):
        for i in range(len(lines) - 2):
            lines[i] = ' '.join(lines[i:i+3])
    return dialogue[:-2], stage_directions[:-2]


def make_three_lines(lines):
    for i in range(len(lines) - 2):
        lines[i] = ' '.join(lines[i:i+3])
    return lines[-2]


def get_labeled_data(path, split):
    assert split == 'train' or split == 'test'
    labeled_data = []
    for line in open(os.path.join(path, split + '.txt')).readlines():
        labeled_data.append((line[2:].rstrip(), (int(line[0]))))
    return labeled_data
