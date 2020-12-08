# coding=utf-8
import json
import numpy as np
import glob
from collections import Counter
import csv
import nltk
from nltk.corpus import wordnet
import sys

wordfile = open('objectlisthindi.txt', encoding="utf-8")
objects = wordfile.readlines()
wordfile.close()

letter_changers = [
	"ा",
	"ी",
	"े",
	"ं",
	"ो",
	"ों"
]

object_list = []
for word in objects:
	word = word.strip().lower()

	if len(word) > 0:
		object_list.append(word)

		if word[-1] in letter_changers:
			object_list.append((word[:-1] + "a").strip())

object_list = list(set(object_list))


def add_original_in_list(list_to_update):
	for word in list_to_update:
		if word[-1] in letter_changers:
			list_to_update.append((word[:-1] + "a").strip())

body_parts = {
	"कमर", "कूल्हे", "कूल्हों",
	"बाएं पैर", "पैर", "पैर", "जांघ", "बाईं जांघ", "जांघ", "हाथों",
	"बाएं घुटने", "घुटने", "घुटने", "घुटनों",
	"बाएं पैर", "पैर", "पैर", "एड़ी", "एड़ी",
	"बाएं पैर की अंगुली", "पैर की अंगुली", "पैर की अंगुली",
	"दाहिने पैर", "पैर", "जांघ", "दाहिनी जांघ", "जांघ",
	"दाहिने घुटने", "घुटने", "घुटनों",
	"दाहिने पैर", "पैर", "ऊँची एड़ी के जूते", "एड़ी","भुजाओं",
	"राइट टो", "टो", "कंधों",
	"लेफ्ट आर्म", "आर्म", "आर्म्स", "फोरआर्म्स", "फोरआर्म", "भुजा", "लेफ्ट भुजा", "बांह",
	"लेफ्ट शोल्डर", "शोल्डर", "शोल्डर",
	"बाएं हाथ", "हाथ", "हाथ", "बाईं कलाई", "कलाई", "कलाई", "बाएं हथेली", "हथेली", "हथेलियां",
	"बाईं उंगली", "उंगली", "उंगलियां","उंगलियों",
	"बाईं कोहनी", "कोहनी", "कोहनी",
	"राइट आर्म", "आर्म", "फोरआर्म्स", "फोरआर्म", "भुजा", "राइट भुजा",
	"राइट शोल्डर", "शोल्डर",
	"राइट हैंड", "हैंड", "राइट रिस्ट", "रिस्ट", "राइट पाम", "पाम",
	"राइट फिंगर", "फिंगर",
	"दाईं कोहनी", "कोहनी",
	"सिर", "चेहरा", "आँखें", "माथे",
	"धड़", "नाभि", "छाती", "शरीर", "पेट",
	"गर्दन", "गला", "कंधे"
}
body_parts = list(body_parts)


rights = [
	"दाहिने",
	"दाहिनी",
	"दाहिना",
	"राइट",
	"दाएं",
	"दाईं"
]

lefts = [
	"बाएं",
	"बाईं",
	"लेफ्ट",
	"बांया"
]

body_parts_syn = [
	["कमर", "कूल्हे", "कूल्हे"],
	["बाएं पैर", "पैर", "पैर", "जांघ", "बाएं जांघ", "जांघ"],
	["बाएं घुटने", "घुटने", "घुटने", "घुटनों"],
	["बाएं पैर", "पैर", "पैर", "ऊँची एड़ी के जूते", "एड़ी"],
	["बाएं पैर की अंगुली", "पैर की अंगुली", "पैर की अंगुली"],
	["दाहिना पैर", "पैर", "पैर", "जांघ", "दाहिनी जांघ", "जांघ"],
	["दाहिने घुटने", "घुटने", "घुटने", "घुटनों"],
	["दाहिने पैर", "पैर", "पैर", "ऊँची एड़ी के जूते", "ए]ड़ी"],
	["दाहिने पैर की अंगुली", "पैर की अंगुली", "पैर की अंगुली"],
	["लेफ्ट आर्म", "आर्म", "आर्म्स", "फोरआर्म्स", "फोरआर्म", "भुजाओं", "भुजा", "बांह", "बाँह"],
	["बाएं कंधे", "कंधे", "कंधे", "कंधों"],
	["बाएं हाथ", "हाथ", "हाथ", "बाईं कलाई", "कलाई", "कलाई", "बाईं हथेली", "हथेली", "हथेलियां", "हाथों"],
	["बाएं उंगली", "उंगली", "उंगलियां", "उंगलियों"],
	["बाएं कोहनी", "कोहनी", "कोहनी"],
	["राइट आर्म", "आर्म", "आर्म्स", "फोरआर्म्स", "फोरआर्म", "भुजाओं", "भुजा", "बांह", "बाँह"],
	["दाहिने कंधे", "कंधे", "कंधों"],
	["दाहिने हाथ", "हाथ", "हाथ", "दाहिनी कलाई", "कलाई", "कलाई", "दाहिनी हथेली", "हथेली", "हथेलियाँ", "हाथों"],
	["दाईं उंगली", "उंगली", "उंगलियां", "उंगलियों"],
	["दाहिनी कोहनी", "कोहनी", "कोहनी"],
	["सिर", "चेहरा", "आँखें", "माथे"],
	["धड़", "नाभि", "छाती", "शरीर", "पेट"],
	["गर्दन", "गला"]
]

directions = [
	"दाहिने",
	"दाहिनी",
	"दाहिना",
	"बांया",
	"बाएं",
	"बाईं",
	"लेफ्ट",
	"राइट",
	"दाएं",
	"दाईं",
	"सही",
	"बाएं",
	"यूपी",
	"ऊपर",
	"अधोमुखी",
	"नीचे",
	"गिराओ",
	"आगे",
	"उठाएं",
	"सामने",
	"वापस",
	"बाहर",
	"पर",
	"पीछे",
	"अंक",
	"बताया",
	"मोड़ते",
	"सीधा",
	"मोड़ें",
	"मोड़ने",
	"अंदर",
	"मोड़ना",
	"मोड़ो"
]



direction_syn = [
	["दाहिने", "दाहिनी", "दाहिना","राइट",
	"दाएं",
	"दाईं",
	"सही"],
	["बाएं",
	"बाईं",
	"लेफ्ट","बाएं","बांया"],
	["यूपी", "ऊपर", "उठाएं"],
	["अधोमुखी",
	"नीचे","गिराओ"],
	["मोड़ते", "मोड़ें","मोड़ो","मोड़ना"],
	["वापस","पीछे"]
]



special_directions = [
	"को",
	"की ओर",
	"की ओर",
	"के बगल में",
	"पीछे",
	"का सामना करना पड़",
	"पोइंट्स ",
	"बताया",
	"पर",
	"अत",
	"पास में"
]

skip_words = [
	"से"
]
colors = [
	"लाल",
	"सफेद",
	"नीला",
	"ग्रे",
	"ग्रे",
	"काली",
	"चांदी",
	"हरा",
	"पीला",
	"तन",
	"ब्राउन",
	"संतरा",
	"लाल",
	"सोना",
	"बैंगनी",
	"गुलाबी"
]

movements = [
	"मोड़",
	"ले",
	"घूर्णन",
	"लाओ",
	"झुकना",
	'लिफ्ट',
	"बढ़ाने",
	"कम",
	"मोड़",
	"रख",
	"जाना",
	"समायोजि",
	"खींचें",
	"धक्का",
	"खिंचाव",
	"कोण",
	"सीधा",
	"ड्रॉप",
	"दुबला",
	"जगह",
	"गिराओ",
	"लाएं",
	"मोड़ें"
]

yours = [
	"आपकी",
	"अपने",
	"अपनी",
	"अपना"
]


implict_movement = [
	"आगे",
	"नीचे",
	"बाहर",
	"सामने"
]


modifier_words = [
	"थोड़ा-सा",
	"थोड़ा",
	"लगभग",
	"थोड़ा",
	"भी",
	"अधिक"
]

add_original_in_list(body_parts)
add_original_in_list(rights)
add_original_in_list(lefts)
for syn in body_parts_syn:
	add_original_in_list(syn)
add_original_in_list(directions)
for syn in direction_syn:
	add_original_in_list(syn)
add_original_in_list(special_directions)
add_original_in_list(colors)
add_original_in_list(movements)
add_original_in_list(yours)
add_original_in_list(implict_movement)
add_original_in_list(modifier_words)
add_original_in_list(skip_words)

break_words = [
	"की",
	"को"
]

def make_pairs(caption):
	words = caption.replace("।", "").split()
	pairs = {}

	for i in range(0, len(words)):
		if words[i][-1] in letter_changers and not words[i] in break_words:
			words[i] = words[i][:-1] + "a"

	for i in range(0, len(words)):
		pair_p1 = ""
		pair_p2 = ""

		if words[i] in body_parts:
			if (words[i-1] in rights or words[i-1] in lefts):
		 		pair_p1 += words[i-1] + " "
			pair_p1 += words[i]

		for j in range(0, len(words)):
			if j > i:
				if j+1 < len(words):
					if words[j] in break_words:
						if words[j+1] in modifier_words:
							if j+2 < len(words) and words[j+2] in directions:
								pair_p2 += words[j+2]
								break
							elif j+2 < len(words) and words[j+2].isnumeric():
								if j+3 < len(words) and words[j+3] in directions:
									pair_p2 += words[j+3]
								elif j+4 < len(words) and words[j+4] in directions:
									pair_p2 += words[j+4]
						
						if words[j+1] in directions:
							if (words[j+1] in rights or words[j+1] in lefts) and words[j+2] in body_parts:
								pair_p2 += words[j+1] + " " + words[j+2]
							else:
								pair_p2 += words[j+1]
						elif words[j+1] in yours:
							if j+3 < len(words):
								if (words[j+2] in rights or words[j+2] in lefts) and words[j+3] in body_parts:
									pair_p2 += words[j+2] + " " + words[j+3]
								elif words[j+2] in directions:
									pair_p2 += words[j+2]
								else:
									pair_p2 += words[j+2]
							elif j+2 < len(words):
								pair_p2 += words[j+2]
						elif (words[j+1] in colors):
							pair_p2 += words[j+2]
						elif words[j+1] in object_list or words[j+1] in body_parts:
							pair_p2 += words[j+1]
							
					elif words[i+1] in implict_movement or words[i+1] in object_list:
						pair_p2 += words[i+1]
					elif words[i+1] in yours:
						if i+3 < len(words):
							if (words[i+2] in rights or words[i+2] in lefts) and words[i+3] in body_parts:
								pair_p2 += words[i+2] + " " + words[i+3]
							elif words[i+2] in directions:
								pair_p2 += words[i+2]
							else:
								pair_p2 += words[i+2]
					elif words[j] in movements or j > i+4:
						pair_p1 = ""
						pair_p2 = ""
						break

				if (len(pair_p2) > 0):
					break

		if len(pair_p1) > 0 and len(pair_p2) > 0:
			a = pair_p1.split(' ')
			b = pair_p2.split(' ')

			if len(a) > 1:
				if a[0] in rights:
					pair_p1 = "दाहिन " + a[1]
				elif a[0] in lefts:
					pair_p1 = "बाए " + a[1]

			if len(b) > 1:
				if b[0] in rights:
					pair_p2 = "दाहिन " + b[1]
				elif b[0] in lefts:
					pair_p2 = "बाए " + b[1]

			pairs.update({pair_p1.strip() : pair_p2.strip()})
	return pairs


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