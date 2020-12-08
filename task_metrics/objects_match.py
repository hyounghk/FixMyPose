import json
import numpy as np
import glob
from collections import Counter
import csv
import nltk
from nltk.corpus import wordnet
import sys

wordfile = open('objectwords.txt')
objects = wordfile.readlines()
wordfile.close()

wordlist = {}

final_word_list = []

object_list = []
for word in objects:
 word = word.strip().lower()

 if len(word) > 0:
   object_list.append(word)
   
   if word is not None and len(word) > 0 and word not in final_word_list:
     final_word_list.append(word)

object_list = list(set(object_list))
for word in object_list:
 syn = wordnet.synsets(word, pos=wordnet.NOUN)

 synlinst = [word]
 if word is not None and len(word) > 0 and word not in final_word_list:
 	final_word_list.append(word)
 for g in syn:
   w = g.name().split('.')[0]
   w = w.replace('_', ' ')
   synlinst.append(w)

   if w is not None and len(w) > 0 and w not in final_word_list:
     final_word_list.append(w)
   
   synlinst = list(set(synlinst))

 for s in synlinst:
  if wordlist.get(s) is not None:
   wordlist[s].extend(synlinst)
   wordlist[s] = list(set(wordlist[s]))
  else:
   wordlist[s] = synlinst


files = sys.argv[1:]

for f in files:
	print()
	image_with_caption = {}
	image_with_gt = {}

	gen_txt = f

	with open(gen_txt, 'r', encoding="utf-8") as f:
		gen = json.load(f)

		for k,v in gen.items():
			image_with_caption.update({k : v[0]})
			image_with_gt.update({k : v[1]})

	cnt_global = 0.0

	for image,c in image_with_caption.items():	
		cnt_local = 0.0
		words_done = []

		for word in final_word_list:
			if word in c and word not in words_done:
				if ' ' in word:
					words_done.append(word.split()[1])
				words_done.append(word)

				if word in image_with_gt[image][0].split(" "):
					cnt_local += 1
				if word in image_with_gt[image][1].split(" "):
					cnt_local += 1
				if word in image_with_gt[image][2].split(" "):
					cnt_local += 1
		
		cnt_global += (cnt_local / 3)

	print(gen_txt + ": ", cnt_global / len(image_with_caption))