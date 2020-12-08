import json
import numpy as np
import glob
from collections import Counter
import csv
import nltk
from nltk.corpus import wordnet
import sys

wordfile = open('objectlisthindi.txt', encoding='utf-8')
objects = wordfile.readlines()
wordfile.close()

wordlist = {}

object_list = []
for word in objects:
 word = word.strip().lower()
 if len(word) > 0:
   object_list.append(word)


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

		for word in object_list:
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