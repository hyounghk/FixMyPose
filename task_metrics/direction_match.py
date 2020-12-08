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

object_list = []
for word in objects:
 word = word.strip().lower()
 object_list.append(word)


object_list = list(set(object_list))
for word in object_list:
 syn = wordnet.synsets(word, pos=wordnet.NOUN)

 synlinst = [word]
 for g in syn:
   w = g.name().split('.')[0]
   w = w.replace('_', ' ')
   synlinst.append(w)
   
   synlinst = list(set(synlinst))

 for s in synlinst:
  if wordlist.get(s) is not None:
   wordlist[s].extend(synlinst)
   wordlist[s] = list(set(wordlist[s]))
  else:
   wordlist[s] = synlinst


body_parts = {
	"waist", "hip", "hips",
	"left leg", "leg","legs", "thigh","left thigh", "thighs",
	"left knee", "knee","knees",
	"left foot", "foot","feet","heels", "heel",
	"left toe", "toe","toes",
	"right leg", "leg", "thigh","right thigh", "thighs",
	"right knee", "knee",
	"right foot", "foot","heels", "heel",
	"right toe", "toe",
	"left arm", "arm","arms","forearms", "forearm",
	"left shoulder", "shoulder","shoulders",
	"left hand", "hand","hands", "left wrist", "wrist", "wrists", "left palm", "palm", "palms",
	"left finger", "finger","fingers",
	"left elbow", "elbow", "elbows",
	"right arm", "arm",	"forearms", "forearm",
	"right shoulder", "shoulder",
	"right hand", "hand", "right wrist", "wrist", "right palm", "palm",
	"right finger", "finger",
	"right elbow", "elbow",
	"head", "face", "eyes", "forehead",
	"torso", "navel", "chest", "body", "belly",
	"neck", "throat",
}

body_parts_syn = [
	["waist", "hip", "hips"],
	["left leg", "leg","legs", "thigh","left thigh", "thighs"],
	["left knee", "knee","knees"],
	["left foot", "foot","feet","heels", "heel"],
	["left toe", "toe","toes"],
	["right leg", "leg", "legs", "thigh","right thigh", "thighs"],
	["right knee", "knee", "knees"],
	["right foot", "foot","feet", "heels", "heel"],
	["right toe", "toe", "toes"],
	["left arm", "arm","arms","forearms", "forearm"],
	["left shoulder", "shoulder","shoulders"],
	["left hand", "hand","hands", "left wrist", "wrist", "wrists", "left palm", "palm", "palms"],
	["left finger", "finger","fingers"],
	["left elbow", "elbow", "elbows"],
	["right arm", "arm", "arms", "forearms", "forearm"],
	["right shoulder", "shoulder"],
	["right hand", "hand", "hands", "right wrist", "wrist", "wrists", "right palm", "palm", "palms"],
	["right finger", "finger", "fingers"],
	["right elbow", "elbow",  "elbows"],
	["head", "face", "eyes", "forehead"],
	["torso", "navel", "chest", "body", "belly"],
	["neck", "throat"]
]

directions = [
	"right",
	"left",
	"up",
	"upwards",
	"downwards",
	"down",
	"forward",
	"forwards",
	"front",
	"back",
	"backwards",
	"to",
	"towards",
	"toward",
	"beside",
	"besides",
	"out",
	"outwards",
	"inwards",
	"on",
	"behind",
	"facing",
	"points",
	"pointed",
	"at"
]

direction_syn = [
	["up", "upwards"],
	["down", "downwards"],
	["front", "forwards", "forward"],
	["back", "backwards"],
	["to", "towards", "toward"],
	["beside", "besides"],
	["out", "outwards"],
	["in", "inwards"],
	["points", "pointed"]
]

special_directions = [
	"to",
	"towards",
	"toward",
	"beside",
	"besides",
	"behind",
	"facing",
	"points",
	"pointed",
	"at",
	"on",
	"near"
]

colors = [
	"red",
	"white",
	"blue",
	"grey",
	"gray",
	"black",
	"silver",
	"green",
	"yellow",
	"tan",
	"brown",
	"orange",
	"maroon",
	"gold",
	"purple",
	"pink"
]

movements = [
	"turn",
	"move",
	"rotate",
	"bring",
	"bend",
	'lift',
	"raise",
	"lower",
	"twist",
	"keep",
	"shift",
	"adjust",
	"pull",
	"push",
	"stretch",
	"angle",
	"straighten",
	"drop",
	"lean",
	"place"
]

implict_directions = [
	"raise",
	"lower",
	"lift",
	"pull",
	"bend",
	"straighten",
	"drop",
	"bring",
	"place"
]

implict_directions_that_need_modifier = [
	"pull",
	"bring"
]

def make_pairs(caption):
	words = caption.replace('.', '').split()
	pairs = {}

	for i in range(0, len(words)):
		pair_p1 = ""
		pair_p2 = ""

		if words[i] in body_parts:
			if ((words[i-1] == "your" or words[i-1] == "the") and words[i-2] in directions) and words[i-3] not in implict_directions:
				continue
			if (i+1 < len(words)):
				if (words[i+1] == "your" and words[i] == "face"):
					continue

			if (words[i-1] == "right" or words[i-1] == "left"):
				pair_p1 += words[i-1] + " "
			pair_p1 += words[i]

			for j in range(0, len(words)):
				if j > i:
					if words[j] in directions or (words[j] in special_directions):
						if (words[j] in special_directions):
							if j+1 < len(words):
								if words[j+1] == "the" or words[j+1] == "your":
									if j+3 < len(words):
										if words[j+2] in colors:
											pair_p2 += words[j+3]
											break
										elif j+2 < len(words):
											pair_p2 += words[j+2]

											if words[j+2] in implict_directions:
												if words[j+3] in body_parts:
													pair_p2 = words[j+3]
													break
												elif words[j+3] == "right" or words[j+3] == "left":
													if j+4 < len(words):
														if words[j+4] in body_parts:
															pair_p2 = words[j+3] + " " + words[j+4]
															break

									elif j+2 < len(words):
										pair_p2 += words[j+2]
										break
								elif words[j+1] in body_parts:
									if (words[j+1] == "right" or words[j+1] == "left") and j+2 < len(words):
										pair_p2 += words[j+1] + ' ' + words[j+2]
										break
									else:
										pair_p2 += words[j+1]
										break
						elif words[j] in directions and (j + 3 < len(words) and not words[j+1] in special_directions):
							pair_p2 += words[j]
							break
						elif (words[j] == "right" or words[j] == "left") and words[j-1] == "the":
							if j+1 < len(words):
								if not (words[j+1] in body_parts):
									pair_p2 += words[j]
									break
							else:
								pair_p2 += words[j]
								break
						elif not (words[j] == "right" or words[j] == "left"):
							if j+1 < len(words):
								if (words[j+1] in special_directions):
									if j+2 < len(words):
										if words[j+2] == "the" or words[j+2] == "your":
											if j+4 < len(words):
												if words[j+3] in colors:
													pair_p2 += words[j+4]
													break
												elif j+3 < len(words):
													pair_p2 += words[j+3]
													
													if words[j+3] == "right" or words[j+3] == "left":
														if words[j+4] in body_parts:
															pair_p2 += ' ' + words[j+4]
													break
											elif j+3 < len(words):
												pair_p2 += words[j+3]
												break
								else:
									pair_p2 += words[j]
									break
							else:
								pair_p2 += words[j]
								break

						if (len(pair_p2) > 0):
							break
					elif words[j] in movements:
						pair_p1 = ""
						pair_p2 = ""
						break
			if len(pair_p2) == 0:
				if i-1 >= 0:
					if (len(pair_p1) == 0):
						if words[i-1] == "right" or words[i-1] == "left":
							pair_p1 += words[i-1] + " "
						pair_p1 += words[i]
				
					if words[i-1] == "right" or words[i-1] == "left":
						i -= 1

					if words[i-1] == "your" or words[i-1] == "the":
						down_step = 2
						while True:
							if i-down_step < 0 or down_step > 4:
								break

							if i-down_step >= 0 and i-down_step-1 >= 0 and words[i-down_step] in directions and words[i-down_step-1] in implict_directions:
								pair_p2 += words[i-down_step]
								break
							elif i-down_step >= 0:
								if words[i-down_step] in implict_directions and words[i-down_step] not in implict_directions_that_need_modifier:
									pair_p2 += words[i-down_step]
									break
							down_step += 1
			elif pair_p2 == "level":
				if words[j-2] in body_parts:
					pair_p2 = words[j-2]
				else:
					move_up = 1
					while True:
						if j+move_up >= len(words) or words[j+move_up] in movements:
							break
						
						if j+move_up+1 < len(words):
							if words[j+move_up] == "your" and words[j+move_up+1] in body_parts:
								pair_p2 = words[j+move_up+1]
								break
						move_up += 1


		if len(pair_p1) > 0 and len(pair_p2) > 0:
			pairs.update({pair_p1 : pair_p2})
	return pairs

files = sys.argv[1:]

for f in files:
	print()
	image_with_caption = {}
	image_with_gt = {}

	gen_txt = f

	with open(gen_txt, 'r') as f:
		gen = json.load(f)

		for k,v in gen.items():
			image_with_caption.update({k : v[0]})
			image_with_gt.update({k : v[1]})

	cnt_global = 0.0

	for image,c in image_with_caption.items():		
		pairs_gen = make_pairs(c)

		pairs_gt1 = make_pairs(image_with_gt[image][0])
		pairs_gt2 = make_pairs(image_with_gt[image][1])
		pairs_gt3 = make_pairs(image_with_gt[image][2])

		cnt_local = 0.0

		for bp,d in pairs_gen.items():
			if bp in pairs_gt1.keys():
				if pairs_gt1[bp] == d:
					cnt_local += 1
				elif d in directions:
					for ds in direction_syn:
						if d in ds and pairs_gt1[bp] in ds:
							cnt_local += 1
				elif d in body_parts:
					for bps in body_parts_syn:
						if d in bps and pairs_gt1[bp] in bps:
							cnt_local += 1
				
			if bp in pairs_gt2.keys():
				if pairs_gt2[bp] == d:
					cnt_local += 1
				elif d in directions:
					for ds in direction_syn:
						if d in ds and pairs_gt2[bp] in ds:
							cnt_local += 1
				elif d in body_parts:
					for bps in body_parts_syn:
						if d in bps and pairs_gt2[bp] in bps:
							cnt_local += 1
				
			if bp in pairs_gt3.keys():
				if pairs_gt3[bp] == d:
					cnt_local += 1
				elif d in directions:
					for ds in direction_syn:
						if d in ds and pairs_gt3[bp] in ds:
							cnt_local += 1
				elif d in body_parts:
					for bps in body_parts_syn:
						if d in bps and pairs_gt3[bp] in bps:
							cnt_local += 1
		
		cnt_global += (cnt_local / 3)

	print(gen_txt + ": ", cnt_global / len(image_with_caption))