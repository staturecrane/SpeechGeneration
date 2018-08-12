import json

speakers = open('all-data/SPEAKERS.TXT').readlines()
split = lambda x: [y.strip() for y in x.split('|')]
split_speakers = [split(speaker) for speaker in speakers]
speaker_dict = {x[0]: 1 if x[1] == 'M' else 0 for x in split_speakers}
with open('speaker_dict.json', 'w') as speaker_file:
    speaker_file.write(json.dumps(speaker_dict))
