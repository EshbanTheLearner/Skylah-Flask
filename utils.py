import pickle

labels = []
verdict = None

def depression_detect(results, labels=labels):
	act_result = []
	labels = labels
	cumm_score = 0

	for result in results:
		for r in result:
			cumm_score = cumm_score + r 
			act_result.append(round(r, 2))
			labels.append(lookup(r))
	cumm_score = cumm_score/len(act_result)
	return act_result, labels, cumm_score

def lookup(value, verdict=verdict):
    verdict = verdict
    
    data = {0:"Highly Depressed", 1:"Highly Depressed", 2:"Depressed", 3:"Depressed", 4:"Mildly Depressed", 5:"Not Depressed", 
    6:"Barely Happy", 7:"Happy", 8:"Quite Happy", 9:"Very Happy"}
    
    '''

    data = {0:"Happy", 1:"Happy", 2:"Happy", 3:"Quite Happy", 4:"Not Depressed", 5:"Not Depressed", 
    6:"Mildly Depressed", 7:"Mildly Depressed", 8:"Depressed", 9:"Highly Depressed"}
    '''
    
    value = round(value,1)*10
    #print("VALUE->", value)
    if value in data:
    	verdict = data[value]

    return verdict

def load_clf():
    f = open("models/classifier/classifier.pickle", 'rb')
    clf = pickle.load(f)
    f.close()

    f = open("models/classifier/clf_tokenizer.pickle", 'rb')
    tokenizer_obj = pickle.load(f)
    f.close()

    return clf, tokenizer_obj    