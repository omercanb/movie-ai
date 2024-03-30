import os
import time
import nltk

CATEGORIZED_SRIPT_LINES_PATH = 'script_files/dialogue_and_screen_directions'
SAVE_PATH = 'script_files/joined'

def get_all_dialogue_and_stage_directions():
    raw_path = os.path.join(SAVE_PATH, 'raw.txt')
    already_saved = os.path.exists(raw_path)

    if already_saved:
        filepaths = (raw_path)
    else:
        filepaths = (os.path.join(CATEGORIZED_SRIPT_LINES_PATH, filename) 
                     for filename in os.listdir(CATEGORIZED_SRIPT_LINES_PATH))
        
    dialogue = []
    stage_directions = []
    for filepath in filepaths:
        file_dialogue, file_stage_directions = get_dialogue_and_stage_directions_of_file(filepath)
        dialogue.extend(file_dialogue)
        stage_directions.extend(file_stage_directions)

    if not already_saved:
        write_dialogue_and_stage_directions(dialogue, stage_directions, raw_path)
    return dialogue, stage_directions


def get_sentence_dialogue_and_stage_directions():
    sentences_path = os.path.join(SAVE_PATH, 'sentences.txt')
    already_saved = os.path.exists(sentences_path)

    if already_saved:
        return get_dialogue_and_stage_directions_of_file(sentences_path)

    dialogue, stage_directions = get_all_dialogue_and_stage_directions()
    dialogue = ' '.join(dialogue)
    stage_directions = ' '.join(stage_directions)
    dialogue_sentences = nltk.sent_tokenize(dialogue)
    stage_direction_sentences = nltk.sent_tokenize(stage_directions)

    write_dialogue_and_stage_directions(dialogue_sentences, stage_direction_sentences, sentences_path)
    return dialogue_sentences, stage_direction_sentences


def get_three_sentence_dialogue_and_screen_directions():
    dialogue, stage_directions = get_sentence_dialogue_and_stage_directions()
    for lines in (dialogue, stage_directions):
        for i in range(len(lines) - 2):
            lines[i] = ' '.join(lines[i:i+3])
    return dialogue[:-2], stage_directions[:-2]



def make_save_path():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)


def write_dialogue_and_stage_directions(dialogue, stage_directions, path):
    make_save_path()
    with open(path, 'w') as f:
        f.writelines(line.strip() + '\n' for line in dialogue)
        f.write('\n')
        f.writelines(line.strip() + '\n' for line in stage_directions)


def get_dialogue_and_stage_directions_of_file(path):
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


def get_script_count():
    return len(os.listdir(CATEGORIZED_SRIPT_LINES_PATH))


if __name__ == '__main__':
    dialogue, screen_directions =  get_three_sentence_dialogue_and_screen_directions()
    print(dialogue[10])
    print(screen_directions[15])