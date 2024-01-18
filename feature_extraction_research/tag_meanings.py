import nltk

with open('feature_extraction_research/important_tags.txt') as tag_file:
    tags = tag_file.readline()[:-1]

for tag in tags:
    nltk.help.upenn_tagset(tag)