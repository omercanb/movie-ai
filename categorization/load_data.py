import os

DATA_PATH = 'script_files/joined/raw.txt'

def get_dialogue_and_stage_directions():
    dialogue = []
    stage_directions = []
    with open(DATA_PATH) as f:
        current_type = dialogue
        for line in f.readlines():
            if line == '\n':
                current_type = stage_directions
                continue
            current_type.append(line.strip())
    return dialogue, stage_directions


def get_three_line_dialogue_and_stage_directions():
    dialogue, stage_directions = get_dialogue_and_stage_directions()
    for lines in (dialogue, stage_directions):
        for i in range(len(lines) - 2):
            lines[i] = ' '.join(lines[i:i+3])
    return dialogue[:-2], stage_directions[:-2]