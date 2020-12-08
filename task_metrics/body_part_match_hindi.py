import json
import numpy as np
import glob
from collections import Counter
import csv
import sys

def distance_finder(x1,y1,z1,x2,y2,z2):
	dist = (((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))**(1/2)
	return dist


index_join = [
	[0, 1, 6, 11],
	[1, 2, 3],
	[2],
	[3, 4],
	[4],
	[6, 7, 8],
	[7],
	[8, 9],
	[9],
	[15, 16, 17],
	[15],
	[17, 22],
	[22],
	[16],
	[42, 43, 44],
	[42],
	[44, 49],
	[49],
	[43],
	[38, 40],
	[38, 11],
	[38]
]

words_join = [
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

valid_i = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 15, 16, 17, 22, 38, 40, 42, 43, 44, 49]

files = sys.argv[2:]

for f in files:
	print()
	image_with_caption = {}
	image_with_gt = {}

	gen_txt = f

	with open(gen_txt, 'r', encoding='utf-8') as f:
		gen = json.load(f)

		for k,v in gen.items():
			image_with_caption.update({k : v[0]})
			image_with_gt.update({k : v[1]})

	cnt_global = 0.0

	for image,caption in image_with_caption.items():
		current_file = sys.argv[1] + "current_" + image + ".txt"
		target_file = sys.argv[1] + "target_" + image + ".txt"

		targetLines = []
		currentLines = []

		with open(target_file) as aFileTarget:
			for line in aFileTarget:
				strip = line.strip()
				targetLines.append(strip)

		with open(current_file) as aFile:
			for line in aFile:
				strip = line.strip()
				currentLines.append(strip)

		moved_joints = {}

		for i in range(0, len(currentLines)):
			name = currentLines[i].strip().split(":")[1]
			dist = 0

			if i in valid_i:
				if i == 40:
					currentCoords = currentLines[i].strip().split(":")[1].replace("(", "").replace(")", "").split(", ")
					targetCoords = targetLines[i].strip().split(":")[1].replace("(", "").replace(")", "").split(", ")

					currentCoords2 = currentLines[39].strip().split(":")[1].replace("(", "").replace(")", "").split(", ")
					targetCoords2 = targetLines[39].strip().split(":")[1].replace("(", "").replace(")", "").split(", ")

					dist = distance_finder((float(currentCoords[0]) + float(currentCoords2[0]) / 2), (float(currentCoords[1]) + float(currentCoords2[1]) / 2),
											(float(currentCoords[2]) + float(currentCoords2[2]) / 2), (float(targetCoords[0]) + float(targetCoords2[0]) / 2), 
											(float(targetCoords[1]) + float(targetCoords2[1]) / 2), (float(targetCoords[2]) + float(targetCoords2[2]) / 2))
					
					
				else:				
					currentCoords = currentLines[i].strip().split(":")[1].replace("(", "").replace(")", "").split(", ")
					targetCoords = targetLines[i].strip().split(":")[1].replace("(", "").replace(")", "").split(", ")

					dist = distance_finder(float(currentCoords[0]), float(currentCoords[1]), float(currentCoords[2]), float(targetCoords[0]), float(targetCoords[1]), float(targetCoords[2]))
			
				moved_joints.update({i : dist})

		moved_joints_words_indexes = []

		for i in range(0, len(index_join)):
			index_list = index_join[i]
			
			avg_dist = 0

			for index in index_list:
				avg_dist += moved_joints[index]
			
			avg_dist /= len(index_list)

			if avg_dist > 0.10:
				moved_joints_words_indexes.append(i)

		words_to_look_for = []
		
		for i in moved_joints_words_indexes:
			for w in words_join[i]:
				if w not in words_to_look_for:
					words_to_look_for.append(w)
		
		cnt_local = 0.0
		words_done = []
		
		for word in words_to_look_for:
			if word in caption and word not in words_done:
				if ' ' in word:
					words_done.append(word.split()[1])
				words_done.append(word)

				if word in image_with_gt[image][0]:
					cnt_local += 1
				if word in image_with_gt[image][1]:
					cnt_local += 1
				if word in image_with_gt[image][2]:
					cnt_local += 1

		cnt_global += (cnt_local / 3)


	print(gen_txt + ": ", cnt_global / len(image_with_caption))