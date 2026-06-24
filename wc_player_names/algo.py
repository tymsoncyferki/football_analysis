import numpy as np
from tqdm import tqdm
import pandas as pd
from unidecode import unidecode
import re

DEBUG = False

players = pd.read_csv('players_wc2026.csv', header=0)
names = players['LAST NAME(S)']

alphabet = 'abcdefghijklmnopqrstuvwxyz'
ALPHABET = dict()
for idx, l in enumerate(alphabet):
    ALPHABET[l] = idx

class PlayerName:
    def __init__(self, name):
        self.name = name
        self.name_length = len(name)
        self.letters_used = np.array([ALPHABET[l] for l in name[:3]])
        
    def check_next(self, letter_id, new_letter_id):
        if len(self.letters_used[self.letters_used > letter_id]) > 0:
            next = np.min(self.letters_used[self.letters_used > letter_id])
        else: 
            next = np.min(self.letters_used)
        if DEBUG:
            print(f"closest {alphabet[next]}")
        if new_letter_id < next:
            return True
        return False

    def check_last(self, letter_id, new_letter_id):
        if len(self.letters_used[self.letters_used < letter_id]) > 0:
            last = np.max(self.letters_used[self.letters_used < letter_id])
        else: 
            last = np.max(self.letters_used)
        if DEBUG:
            print(f"closest {alphabet[last]}")
        if new_letter_id > last:
            return True
        return False

def check_if_in_circle(name):
    regex = re.compile('[^a-zA-Z]')
    normalized_name = unidecode(regex.sub('', name.lower()))
    if len(normalized_name) < len(name):
        return False
    if len(normalized_name) < 4:
        return False
    player = PlayerName(normalized_name)
    for new_idx in range(3, player.name_length):
        new_letter = player.name[new_idx]
        new_letter_id = ALPHABET[new_letter]
        previous_letter = player.name[new_idx-1]
        previous_letter_id = ALPHABET[previous_letter]
        if DEBUG:
            print(f"new {new_letter}, previous {previous_letter}")
        if new_letter == previous_letter:
            return False
        elif new_letter_id > previous_letter_id:
            if DEBUG:
                print(f"checking next in {player.letters_used}")
            if player.check_next(previous_letter_id, new_letter_id):
                player.letters_used = np.append(player.letters_used, [new_letter_id])
            else:
                return False
        else:
            if DEBUG:
                print(f"checking last in {player.letters_used}")
            if player.check_last(previous_letter_id, new_letter_id):
                player.letters_used = np.append(player.letters_used, [new_letter_id])
            else:
                return False
    return True

good_names = []
for name in tqdm(names):
    if check_if_in_circle(name):
        good_names.append(name)

with open("names.txt", "w") as file:
    for name in good_names:
        file.writelines(f"{name}\n")

func = np.vectorize(len)
longest_name = good_names[np.argmax(func(np.array(good_names)))]

print(f"Longest Name: {longest_name}")        