"""
Find the amount of data I have for each category to find if there are imbalances.
Result: There may be imbalances depending on the type of tokenization used.
"""
import os

import load_dataset


CATEGORIZED_SRIPT_LINES_PATH = 'script_files/dialogue_and_screen_directions'

dialogue, stage_directions = load_dataset.get_all_dialogue_and_stage_directions()

dialogue_char_count = sum(len(line) for line in dialogue)
stage_directions_char_count = sum(len(line) for line in stage_directions)

print(f'Scripts: {load_dataset.get_script_count()}')
print()

print(f'Dialogue lines: {len(dialogue):15d}') # 1585051
print(f'Stage direction lines: {len(stage_directions):8d}') # 1353619
print(f'Ratio: {len(dialogue)/len(stage_directions):.2f}') # 1.17 
print()

print(f'Dialogue chars : {dialogue_char_count:15d}') # 42734931
print(f'Stage direction chars : {stage_directions_char_count:3d}') # 42734931
print(f'Ratio : {dialogue_char_count/stage_directions_char_count:.2f}') # 0.68