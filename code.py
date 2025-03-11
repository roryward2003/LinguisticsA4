import json
from collections import Counter
import numpy as np
import pandas as pd
import re
import nltk
from nltk.data import find
import gensim
import sklearn
import sklearn.discriminant_analysis
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
from sympy.parsing.sympy_parser import parse_expr

np.random.seed(0)
nltk.download('word2vec_sample')

# type: ignore (suppress VSCode warnings about filename)

########-------------- PART 1: LANGUAGE MODELING --------------########

class NgramLM:
	def __init__(self):
		"""
		N-gram Language Model
		"""
		# Dictionary to store next-word possibilities for bigrams. Maintains a list for each bigram.
		self.bigram_prefix_to_trigram = {}
		
		# Dictionary to store counts of corresponding next-word possibilities for bigrams. Maintains a list for each bigram.
		self.bigram_prefix_to_trigram_weights = {}

	def load_trigrams(self):
		"""
		Loads the trigrams from the data file and fills the dictionaries defined above.

		Parameters
		----------

		Returns
		-------
		"""
		with open("data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt", encoding="utf-8") as f:
			lines = f.readlines()
			for line in lines:
				word1, word2, word3, count = line.strip().split()
				if (word1, word2) not in self.bigram_prefix_to_trigram:
					self.bigram_prefix_to_trigram[(word1, word2)] = []
					self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
				self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
				self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))

		# [print(word) for word in self.bigram_prefix_to_trigram[("mandate", "masks")]]

	def top_next_word(self, word1, word2, n=10):
		"""
		Retrieve top n next words and their probabilities given a bigram prefix.

		Parameters
		----------
		word1: str
			The first word in the bigram.
		word2: str
			The second word in the bigram.
		n: int
			Number of words to return.
			
		Returns
		-------
		next_words: list
			The retrieved top n next words.
		probs: list
			The probabilities corresponding to the retrieved words.
		"""
		next_words = []
		probs = []

		# If no next words available, return two empty lists
		if (word1, word2) not in self.bigram_prefix_to_trigram:
			return next_words, probs
		
		all_words = self.bigram_prefix_to_trigram[(word1, word2)]
		all_probs = self.bigram_prefix_to_trigram_weights[(word1, word2)]
		sumOfWeights = sum(self.bigram_prefix_to_trigram_weights[(word1, word2)])
		
		# Sort the pairs by value, then select the top n
		all_pairs = list(zip(all_words, all_probs))
		all_pairs.sort(key=lambda item: -item[1])
		top_pairs = all_pairs[:n]
		for word, prob in top_pairs:
			next_words.append(word)
			probs.append(prob/sumOfWeights)

		return next_words, probs
	
	def sample_next_word(self, word1, word2, n=10):
		"""
		Sample n next words and their probabilities given a bigram prefix using the probability distribution defined by frequency counts.

		Parameters
		----------
		word1: str
			The first word in the bigram.
		word2: str
			The second word in the bigram.
		n: int
			Number of words to return.
			
		Returns
		-------
		next_words: list
			The sampled n next words.
		probs: list
			The probabilities corresponding to the retrieved words.
		"""
		# If no next words available, return a list of "<EOS>"
		if (word1, word2) not in self.bigram_prefix_to_trigram_weights:
			return ["<EOS>"]*n, [0.0]*n
		
		# Ensure n is safe for use (avoid attempted oversampling)
		if len(self.bigram_prefix_to_trigram_weights[(word1, word2)])<n:
			safe_n = len(self.bigram_prefix_to_trigram_weights[(word1, word2)])
		else:
			safe_n = n
		
		# Prep some data so we don't need to repeatedly calucalte this value later on
		sumOfWeights = sum(self.bigram_prefix_to_trigram_weights[(word1, word2)])
		allProbs = [val/sumOfWeights for val in self.bigram_prefix_to_trigram_weights[(word1, word2)]]

		# Choose a sample of
		next_words = list(np.random.choice(self.bigram_prefix_to_trigram[(word1, word2)], size=safe_n, replace=False, p=allProbs))
		indices = [self.bigram_prefix_to_trigram[(word1, word2)].index(word) for word in next_words]		
		probs = [self.bigram_prefix_to_trigram_weights[(word1, word2)][i]/sumOfWeights for i in indices]

		while len(probs)<n:
			probs.append(0.0)
			next_words.append("<EOS>")
		
		return next_words, probs
	
	def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
		"""
		Generate sentences using beam search.

		Parameters
		----------
		prefix: str
			String containing two (or more) words separated by spaces.
		beam: int
			The beam size.
		sampler: Callable
			The function used to sample next word.
		max_len: int
			Maximum length of sentence (as measure by number of words) to generate (excluding "<EOS>").
			
		Returns
		-------
		sentences: list
			The top generated sentences
		probs: list
			The probabilities corresponding to the generated sentences
		"""

		prefix_words = prefix.split()
		word1 = prefix_words[len(prefix_words)-2]
		word2 = prefix_words[len(prefix_words)-1]

		next_words, next_probs = sampler(word1, word2, beam)

		best_sentences = [prefix_words.copy() for _ in range(beam)]
		[best_sentences[i].append(next_words[i]) for i in range(beam)]

		best_probs = next_probs.copy()
		next_sentences = []
		next_sentence_probs = []

		# Iterate until max_len-1 reached
		for i in range(len(prefix_words), max_len-1):
			# For each of the best sentences
			for j in range(beam):
				# If its ended, keep it
				if best_sentences[j][len(best_sentences[j])-1] == "<EOS>":
					next_sentences.append(best_sentences[j].copy())
					next_sentence_probs.append(best_probs[j])
				# If not, expand it into (beam) new sentences
				else:
					next_words, next_probs = sampler(best_sentences[j][len(best_sentences[j])-2],
						best_sentences[j][len(best_sentences[j])-1], beam)
					for k, next_word in enumerate(next_words):
						temp = best_sentences[j].copy()
						temp.append(next_word)
						next_sentences.append(temp)
						next_sentence_probs.append(best_probs[j]*next_probs[k])
			# Cull the generated sentences back to the top (beam)
			best_indices = np.argpartition(next_sentence_probs, -beam)[-beam:]
			best_sentences = [next_sentences[index].copy() for index in best_indices]
			best_probs = [next_sentence_probs[index] for index in best_indices]
			next_sentence_probs = []
			next_sentences = []

		# Append <EOS> to any sentences that don't already end in <EOS>
		for i in range(len(best_sentences)):
			if best_sentences[i][len(best_sentences[i])-1] != "<EOS>":
				best_sentences[i].append("<EOS>")

		# Create a dict using the best pairs and sort it by value, descending
		best_dict = dict([(" ".join(best_sentences[i]), best_probs[i]) for i in range(len(best_sentences))])
		sorted_dict = dict(sorted(best_dict.items(), key=lambda item: (-item[1], item[0])))

		return list(sorted_dict.keys()), list(sorted_dict.values())







# #####------------- CODE TO TEST YOUR FUNCTIONS FOR LANGUAGE MODELING

# print("======================================================================")
# print("Checking Language Model")
# print("======================================================================")

# # Define your language model object
# language_model = NgramLM()
# # Load trigram data
# language_model.load_trigrams()

# print("------------- Evaluating top next word prediction -------------")
# next_words, probs = language_model.top_next_word("middle", "of", 10)
# for word, prob in zip(next_words, probs):
# 	print(word, prob)
# # Your first 5 lines of output should be exactly:
# # a 0.807981220657277
# # the 0.06948356807511737
# # pandemic 0.023943661971830985
# # this 0.016901408450704224
# # an 0.0107981220657277

# print("------------- Evaluating sample next word prediction -------------")
# next_words, probs = language_model.sample_next_word("middle", "of", 10)
# for word, prob in zip(next_words, probs):
# 	print(word, prob)
# # My first 5 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# # a 0.807981220657277
# # pandemic 0.023943661971830985
# # august 0.0018779342723004694
# # stage 0.0018779342723004694
# # an 0.0107981220657277

# print("------------- Evaluating beam search -------------")
# sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word)
# for sent, prob in zip(sentences, probs):
# 	print(sent, prob)
# print("#########################\n")
# # Your first 3 lines of output should be exactly:
# # <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# # <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# # <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

# sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word)
# for sent, prob in zip(sentences, probs):
# 	print(sent, prob)
# print("#########################\n")
# # Your first 3 lines of output should be exactly:
# # <BOS1> <BOS2> biden calls for a 30 bonus URL #cashgem #cashappfriday #stayathome <EOS> 0.0002495268686322749
# # <BOS1> <BOS2> biden says all u.s. governors should mandate masks <EOS> 1.6894510541025754e-05
# # <BOS1> <BOS2> biden says all u.s. governors question cost of a pandemic <EOS> 8.777606198953028e-07

# sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word)
# for sent, prob in zip(sentences, probs):
# 	print(sent, prob)
# print("#########################\n")
# # My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# # <BOS1> <BOS2> trump eyes new unproven coronavirus treatment URL <EOS> 0.00021893147502903603
# # <BOS1> <BOS2> trump eyes new unproven coronavirus cure URL <EOS> 0.0001719607222046247
# # <BOS1> <BOS2> trump eyes new unproven virus cure promoted by mypillow ceo over unproven therapeutic URL <EOS> 9.773272077557522e-05

# sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word)
# for sent, prob in zip(sentences, probs):
# 	print(sent, prob)
# # My first 3 lines of output look like this: (YOUR OUTPUT CAN BE DIFFERENT!)
# # <BOS1> <BOS2> biden is elected <EOS> 0.001236227651321991
# # <BOS1> <BOS2> biden dropping ten points given trump a confidence trickster URL <EOS> 5.1049579351466146e-05
# # <BOS1> <BOS2> biden dropping ten points given trump four years <EOS> 4.367575122292103e-05







########-------------- PART 2: Semantic Parsing --------------########

class Text2SQLParser:
	def __init__(self):
		"""
		Basic Text2SQL Parser. This module just attempts to classify the user queries into different "categories" of SQL queries.
		"""
		self.parser_files = "data/semantic-parser"
		self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
		self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

		self.train_file = "sql_train.tsv"
		self.test_file = "sql_val.tsv"

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
		self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")

		self.ls_labels = list(self.train_df["Label"].unique())

	def predict_label_using_keywords(self, question):
		"""
		Predicts the label for the question using custom-defined keywords.

		Parameters
		----------
		question: str
			The question whose label is to be predicted.
			
		Returns
		-------
		label: str
			The predicted label.
		"""
		
		multi_table_keywords = ["that have", "that never", "not have", "who have", "who never", "have never"
			"who took", "that took", "who are", "that are", "part of", "relate", "for all", "for the",
			"of all the", "correspond", "the different", ", and"]

		ordering_keywords = ["alphabetical", "ascending", "decreasing", "descending", "in order", "order by", "sort",
			"ordered by", "ordered in", "lexicographic"]
		
		grouping_keywords = ["many", "number of", "more", "less", "most", "least" "sum", "average", "count",
			"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "min", "max", "highest", "lowest"]

		if any(keyword in question for keyword in ordering_keywords):
			return 'ordering'
		if any(keyword in question for keyword in multi_table_keywords):
			return 'multi_table'
		if any(keyword in question for keyword in grouping_keywords):
			return 'grouping'
		return 'comparison'

	def evaluate_accuracy(self, prediction_function_name):
		"""
		Gives label wise accuracy of your model.

		Parameters
		----------
		prediction_function_name: Callable
			The function used for predicting labels.
			
		Returns
		-------
		accs: dict
			The accuracies of predicting each label.
		main_acc: float
			The overall average accuracy
		"""
		correct = Counter()
		total = Counter()
		main_acc = 0
		main_cnt = 0
		for i in range(len(self.test_df)):
			q = self.test_df.loc[i]["Question"].split(":")[1].split("|")[0].strip()
			gold_label = self.test_df.loc[i]['Label']
			if prediction_function_name(q) == gold_label:
				correct[gold_label] += 1
				main_acc += 1
			total[gold_label] += 1
			main_cnt += 1
		accs = {}
		for label in self.ls_labels:
			accs[label] = (correct[label]/total[label])*100
		return accs, 100*main_acc/main_cnt

	def get_sentence_representation(self, sentence):
		"""
		Gives the average word2vec representation of a sentence.

		Parameters
		----------
		sentence: str
			The sentence whose representation is to be returned.
			
		Returns
		-------
		sentence_vector: np.ndarray
			The representation of the sentence.
		"""
		sentence_vector = np.zeros(300)
		split = sentence.split()

		word_vecs = []
		
		for word in split:
			if word in self.word2vec_model:
				word_vecs.append(self.word2vec_model[word])

		sentence_vector = np.average(np.array(word_vecs), axis=0)
			
		return sentence_vector
	
	def init_ml_classifier(self):
		"""
		Initializes the ML classifier.

		Parameters
		----------
			
		Returns
		-------
		"""

		# self.classifier = sklearn.svm.SVC(gamma=2, C=1)                                # 76.2
		# self.classifier = sklearn.svm.SVC(gamma=1, C=1)                                # 68.3
		# self.classifier = sklearn.svm.SVC(gamma=10, C=1)                               # 81.0
		# self.classifier = sklearn.svm.SVC(gamma=2, C=2)                                # 79.4
		# self.classifier = sklearn.svm.SVC(gamma=2, C=5)                                # 81.0
		# self.classifier = sklearn.svm.SVC(gamma=2, C=50)                               # 84.1
		self.classifier = sklearn.svm.SVC(gamma=10, C=5)                               # 82.5
	
	def train_label_ml_classifier(self):
		"""
		Train the classifier.

		Parameters
		----------
			
		Returns
		-------
		"""
		sentences = []
		labels = []
		with open("data/semantic-parser/sql_train.tsv", encoding="utf-8") as f:
			lines = f.readlines()
			lines.pop(0) # Remove header line from data set
			for line in lines:
				labels.append(line.split().pop(-1)) # Last word is label
				words = line.split("|")[0].split()  # Words up until first '|' taken
				words.pop(0)                        # Remove first word, which is a label
				sentences.append(" ".join(words))   # Join the words into a string

		sentence_vectors = [self.get_sentence_representation(sentence) for sentence in sentences]
		self.classifier.fit(sentence_vectors, labels)
	
	def predict_label_using_ml_classifier(self, question):
		"""
		Predicts the label of the question using the classifier.

		Parameters
		----------
		question: str
			The question whose label is to be predicted.
			
		Returns
		-------
		predicted_label: str
			The predicted label.
		"""
		predicted_label = self.classifier.predict([self.get_sentence_representation(question)])[0]

		return predicted_label






class MusicAsstSlotPredictor:
	def __init__(self):
		"""
		Slot Predictor for the Music Assistant.
		"""
		self.parser_files = "data/semantic-parser"
		self.train_data = []
		self.test_questions = []
		self.test_answers = []

		self.slot_names = set()

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		with open(f'{self.parser_files}/music_asst_train.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.train_data.append(json.loads(line))

		with open(f'{self.parser_files}/music_asst_val_ques.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.test_questions.append(json.loads(line))

		with open(f'{self.parser_files}/music_asst_val_ans.txt') as f:
			lines = f.readlines()
			for line in lines:
				self.test_answers.append(json.loads(line))
	
	def get_slots(self):
		"""
		Get all the unique slots.

		Parameters
		----------
			
		Returns
		-------
		"""
		for sample in self.train_data:
			for slot_name in sample['slots']:
				self.slot_names.add(slot_name)
	
	def predict_slot_values(self, question):
		"""
		Predicts the values for the slots.

		Parameters
		----------
		question: str
			The question for which the slots are to be predicted.
			
		Returns
		-------
		slots: dict
			The predicted slots.
		"""
		slots = {}
		for slot_name in self.slot_names:
			slots[slot_name] = None

		# playlist and playlist_owner
		# Case 1
		playlist_match = re.search(" (my|the|a|[^ ]+'s) playlist .*(named|called|name) ", question)
		if playlist_match is None: # Case 2
			playlist_match = re.search(" (my|the|a|[^ ]+'s) playlist ", question)
		if playlist_match is None: # Case 3
			playlist_match = re.search(" (my|the|a|[^ ]+'s) .+ playlist ", question)
		if playlist_match is None: # Case 4
			playlist_match = re.search(" (my|[^ ]+'s) .+", question)
		if playlist_match is not None: # If any match was found
			playlist = playlist_match.group().split()
			playlist_owner = playlist.pop(0)
			if "'" in playlist_owner:
				playlist_owner = playlist_owner[:-2]
			slots['playlist_owner'] = playlist_owner

			if playlist[0] == "playlist":    # Cases 1 and 2
				slots['playlist'] = question[playlist_match.end():]
			elif playlist[len(playlist)-1] == "playlist": # Case 3
				playlist.pop(-1)
				slots['playlist'] = " ".join(playlist)
			else:                            # Case 4
				slots['playlist'] = " ".join(playlist)

		# artist
		artist_match = re.search(" [A-Z][a-z]+ [A-Z][a-z]+ ", question)
		if artist_match is not None:
			slots['artist'] = artist_match.group()[1:-1]

		# music_item
		music_items = ["album", "song", "track", "tune", "artist"]
		for music_item in music_items:
			if music_item in question:
				slots['music_item'] = music_item

		# entity_name
		music_item_match = re.search("([Aa]dd|[Pp]ut) .+ (to|in|into|on|onto) ", question)
		if music_item_match is not None:
			selected_words = music_item_match.group().split()[1:-1]
			if not any(music_item in selected_words for music_item in music_items):
				slots['entity_name'] = " ".join(selected_words)
			if slots['playlist'] is None:
				slots['playlist'] = question[music_item_match.end():]

		return slots
	
	def get_confusion_matrix(self, slot_prediction_function, questions, answers):
		"""
		Find the true positive, true negative, false positive, and false negative examples with respect to the prediction of a slot being active or not (irrespective of value assigned).

		Parameters
		----------
		slot_prediction_function: Callable
			The function used for predicting slot values.
		questions: list
			The test questions
		answers: list
			The ground-truth test answers
			
		Returns
		-------
		tp: dict
			The indices of true positive examples are listed for each slot
		fp: dict
			The indices of false positive examples are listed for each slot
		tn: dict
			The indices of true negative examples are listed for each slot
		fn: dict
			The indices of false negative examples are listed for each slot
		"""
		tp = {}
		fp = {}
		tn = {}
		fn = {}
		for slot_name in self.slot_names:
			tp[slot_name] = []
		for slot_name in self.slot_names:
			fp[slot_name] = []
		for slot_name in self.slot_names:
			tn[slot_name] = []
		for slot_name in self.slot_names:
			fn[slot_name] = []

		for i, question in enumerate(questions):
			prediction = slot_prediction_function(question)
			for slot_name in self.slot_names:
				if prediction[slot_name] is not None:
					if slot_name in answers[i]['slots']:
						tp[slot_name].append(i)
					else:
						fp[slot_name].append(i)
				else:
					if slot_name not in answers[i]['slots']:
						tn[slot_name].append(i)
					else:
						fn[slot_name].append(i)

		return tp, fp, tn, fn
	
	def evaluate_slot_prediction_recall(self, slot_prediction_function):
		"""
		Evaluates the recall for the slot predictor. Note: This also takes into account the exact value predicted for the slot 
		and not just whether the slot is active like in the get_confusion_matrix() method

		Parameters
		----------
		slot_prediction_function: Callable
			The function used for predicting slot values.
			
		Returns
		-------
		accs: dict
			The recall for predicting the value for each slot.
		"""
		correct = Counter()
		total = Counter()
		# predict slots for each question
		for i, question in enumerate(self.test_questions):
			i = self.test_questions.index(question)
			gold_slots = self.test_answers[i]['slots']
			predicted_slots = slot_prediction_function(question)
			for name in self.slot_names:
				if name in gold_slots:
					total[name] += 1.0
					if predicted_slots.get(name, None) != None and predicted_slots.get(name).lower() == gold_slots.get(name).lower():
						correct[name] += 1.0
		accs = {}
		for name in self.slot_names:
			accs[name] = (correct[name] / total[name]) * 100
		return accs









class MathParser:
	def __init__(self):
		"""
		Math Word Problem Solver.
		"""
		self.parser_files = "data/semantic-parser"
		self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
		self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

		self.train_file = "math_train.tsv"
		self.test_file = "math_val.tsv"

	def load_data(self):
		"""
		Load the data from file.

		Parameters
		----------
			
		Returns
		-------
		"""
		self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
		self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")

	def get_sentence_representation(self, sentence):
		"""
		Gives the average word2vec representation of a sentence.

		Parameters
		----------
		sentence: str
			The sentence whose representation is to be returned.
			
		Returns
		-------
		sentence_vector: np.ndarray
			The representation of the sentence.
		"""
		sentence_vector = np.zeros(300)
		split = sentence.split()

		word_vecs = []
		
		for word in split:
			if word in self.word2vec_model:
				word_vecs.append(self.word2vec_model[word])

		sentence_vector = np.average(np.array(word_vecs), axis=0)
			
		return sentence_vector
	
	def init_model(self):
		"""
		Initializes the ML classifier.

		Parameters
		----------
			
		Returns
		-------
		"""

		# self.classifier = sklearn.svm.SVC(gamma=2, C=1)                                # 29.6 |
		# self.opclassifier = sklearn.svm.SVC(gamma=1, C=1)                              # 22.2 | 21.5
		# self.classifier = sklearn.svm.SVC(gamma=10, C=5)                               # 33.3 | 32.7

		# self.opclassifier = sklearn.neighbors.KNeighborsClassifier()                 # 44.4 | 29.9
		# self.invclassifier = sklearn.neighbors.KNeighborsClassifier()                # 44.4 | 29.9
		
		# self.invclassifier = sklearn.naive_bayes.GaussianNB()                        # 37.0 | 25.2
		# self.opclassifier = sklearn.neighbors.KNeighborsClassifier()                 # 37.0 | 25.2

 		# 33.3 | 32.7
		# self.invclassifier = sklearn.linear_model.LogisticRegression(
		# 	solver='liblinear',
		# 	penalty='l2',
		# 	C=15,
		# 	max_iter=1000
		# )
		# 	kernel='rbf',
		# 	gamma=10,
		# 	C=5.0,
		# 	decision_function_shape='ovr'
		# )

		# self.classifier = sklearn.neighbors.KNeighborsClassifier()                     # 48.1 | 25
		# self.classifier = sklearn.neighbors.KNeighborsClassifier(1)                    # 48.1 |
		# self.classifier = sklearn.neighbors.KNeighborsClassifier(20)                   # 40.7 |

		# self.classifier = sklearn.gaussian_process.GaussianProcessClassifier           # 25.9 | 22.4
		# self.classifier = sklearn.linear_model.LogisticRegression()                    # 25.9 |
		# self.classifier = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()# 22.2 | 23.4

		# 48.1 | 
		# self.opclassifier = sklearn.naive_bayes.GaussianNB()                             # 44.4 | 32.7
		self.opclassifier = sklearn.svm.SVC(gamma=10, C=5)                               # 33.3 | 32.7
		self.invclassifier = sklearn.ensemble.RandomForestClassifier(max_depth=5,
			n_estimators=10, max_features=1, random_state=42)                            # 37.0 | 29.9
		
		# self.opclassifier = sklearn.neural_network.MLPClassifier(alpha=1,
		# 	max_iter=1000, random_state=42)                       # 33.3 | 32.7

		self.train_models()

	def train_models(self):
		"""
		Initializes the ML classifier.

		Parameters
		----------
			
		Returns
		-------
		"""
		questions = []
		ops = []
		inv = []
		with open("data/semantic-parser/math_train.tsv", encoding="utf-8") as f:
			lines = f.readlines()
			lines.pop(0) # Remove header line from data set
			for line in lines:
				words = line.split()
				first_val = float(re.search(r"\d+", line).group())
				words.pop(-1)                                        # ignore result
				inv.append(float(first_val) == float(words.pop(-1))) # check if inverted (y-x != x-y)
				op = words.pop(-1)                                   # get operator
				match op:
					case '+':
						ops.append('+')
					case '*':
						ops.append('*')
					case '-':
						ops.append('-')
					case '/':
						ops.append('/')
					case '_':
						ops.append('+')
				words.pop(-1)                  # remove the first operand
				questions.append(" ".join(words))

		question_vectors = [self.get_sentence_representation(question) for question in questions]
		self.opclassifier.fit(question_vectors, ops)
		self.invclassifier.fit(question_vectors, inv)


	def predict_equation_from_question(self, question):
		"""
		Predicts the equation for the question.

		Parameters
		----------
		question: str
			The question whose equation is to be predicted.
			
		Returns
		-------
		equation: str
			The predicted equation.
		"""
		eq = ""
		temp = 0.0
		values = re.findall(r"\d+", question)
		if len(values) == 2:
			x = float(values[0])
			y = float(values[1])
			op = self.opclassifier.predict([self.get_sentence_representation(question)])[0]
			inv = self.invclassifier.predict([self.get_sentence_representation(question)])[0]
			if inv:
				temp = x
				x = y
				y = temp
			match op:
				case '+':
					eq = str(x)+" + "+str(y)
				case '*':
					eq = str(x)+" * "+str(y)
				case '-':
					eq = str(x)+" - "+str(y)
				case '/':
					eq = str(x)+" / "+str(y)
		else:
			eq = "1.0 + 1.0"
		return eq
	
	def ans_evaluator(self, equation):
		"""
		Parses the equation to obtain the final answer.

		Parameters
		----------
		equation: str
			The equation to be parsed.
			
		Returns
		-------
		final_ans: float
			The final answer.
		"""
		try:
			final_ans = parse_expr(equation, evaluate = True)
		except:
			final_ans = -1000.112
		return final_ans
	
	def evaluate_accuracy(self, prediction_function_name):
		"""
		Gives accuracy of your model.

		Parameters
		----------
		prediction_function_name: Callable
			The function used for predicting equations.
			
		Returns
		-------
		main_acc: float
			The overall average accuracy
		"""
		acc = 0
		tot = 0
		for i in range(len(self.test_df)):
			ques = self.test_df.loc[i]["Question"]
			gold_ans = self.test_df.loc[i]["Answer"]
			pred_eq = prediction_function_name(ques)
			pred_ans = self.ans_evaluator(pred_eq)

			if abs(gold_ans - pred_ans) < 0.1:
				acc += 1
			tot += 1
		return 100*acc/tot









# #####------------- CODE TO TEST YOUR FUNCTIONS FOR SEMANTIC PARSING

# print()
# print()

# ### PART 1: Text2SQL Parser

# print("======================================================================")
# print("Checking Text2SQL Parser")
# print("======================================================================")

# # Define your text2sql parser object
# sql_parser = Text2SQLParser()

# # Load the data files
# sql_parser.load_data()

# # Initialize the ML classifier
# sql_parser.init_ml_classifier()

# # Train the classifier
# sql_parser.train_label_ml_classifier()

# # Evaluating the keyword-based label classifier. 
# print("------------- Evaluating keyword-based label classifier -------------")
# accs, _ = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_keywords)
# for label in accs:
# 	print(label + ": " + str(accs[label]))

# # Evaluate the ML classifier
# print("------------- Evaluating ML classifier -------------")
# sql_parser.train_label_ml_classifier()
# _, overall_acc = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_ml_classifier)
# print("Overall accuracy: ", str(overall_acc))

# print()
# print()


# ### PART 2: Music Assistant Slot Predictor

# print("======================================================================")
# print("Checking Music Assistant Slot Predictor")
# print("======================================================================")

# # Define your semantic parser object
# semantic_parser = MusicAsstSlotPredictor()
# # Load semantic parser data
# semantic_parser.load_data()

# # Look at the slots
# print("------------- slots -------------")
# semantic_parser.get_slots()
# print(semantic_parser.slot_names)

# # Evaluate slot predictor
# # Our reference implementation got these numbers on the validation set. You can ask others on Slack what they got.
# # playlist_owner: 100.0
# # music_item: 100.0
# # entity_name: 16.666666666666664
# # artist: 14.285714285714285
# # playlist: 52.94117647058824
# print("------------- Evaluating slot predictor -------------")
# accs = semantic_parser.evaluate_slot_prediction_recall(semantic_parser.predict_slot_values)
# for slot in accs:
# 	print(slot + ": " + str(accs[slot]))

# # Evaluate Confusion matrix examples
# print("------------- Confusion matrix examples -------------")
# tp, fp, tn, fn = semantic_parser.get_confusion_matrix(semantic_parser.predict_slot_values, semantic_parser.test_questions, semantic_parser.test_answers)
# print(tp)
# print(fp)
# print(tn)
# print(fn)

# print()
# print()


### PART 3. Math Equation Predictor

print("======================================================================")
print("Checking Math Parser")
print("======================================================================")

# Define your math parser object
math_parser = MathParser()

# Load the data files
math_parser.load_data()

# Initialize and train the model
math_parser.init_model()

# Get accuracy
print("------------- Accuracy of Equation Prediction -------------")
acc = math_parser.evaluate_accuracy(math_parser.predict_equation_from_question)
print("Accuracy: ", acc)