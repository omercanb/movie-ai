"""
Verb tenses seemed to be important for categorization.
Screen directions seemed to have much more use of past tense.
This module checks for these patterns.
Result:
The biggest difference in frequency of part of speeches was <0.1
So it will most likely have no help in decision making.
"""

import collections
import random

import nltk #Do not use nltk is too slow for the task

import load_dataset


WORD_SAMPLE_SIZE = 10000 # Can do 10000 samples in reasonable time


def get_tag_counts(lines):
    words = []
    while len(words) < WORD_SAMPLE_SIZE:
        words.extend(nltk.word_tokenize(random.choice(lines)))
    words = words[:WORD_SAMPLE_SIZE]
    tags = nltk.pos_tag(words)
    tag_counts = collections.defaultdict(lambda : 0)
    for word, tag in tags:
        tag_counts[tag] += 1
    return tag_counts

dialogue, screen_directions = load_dataset.get_all_dialogue_and_stage_directions()

dialogue_tag_counts = get_tag_counts(dialogue)
screen_direction_tag_counts = get_tag_counts(screen_directions)

tags = set([*dialogue_tag_counts.keys(), *screen_direction_tag_counts.keys()])

tag_frequency_differences = dict()
for tag in tags:
    dialogue_tag_frequency = dialogue_tag_counts[tag]/WORD_SAMPLE_SIZE
    stage_direction_tag_frequency = screen_direction_tag_counts[tag]/WORD_SAMPLE_SIZE
    tag_frequency_differences[tag] = dialogue_tag_frequency - stage_direction_tag_frequency
    print(tag, dialogue_tag_frequency, 
          stage_direction_tag_frequency)
    
most_important_tags = sorted([(tag, freq_difference) for tag, freq_difference in tag_frequency_differences.items()], 
                             key = lambda x: abs(tag_frequency_differences[x[0]]), reverse = True)
print(most_important_tags)
print('Positive means more common in dialogue, negative more common in stage direction')

with open('feature_extraction_research/important_tags.txt', 'w') as f:
    for tag, freq_difference in most_important_tags:
        f.write(tag + '\n')