import json
import os, sys
import glob
import subprocess

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install("evaluate")

import evaluate 
bleu_score = evaluate.load("bleu")

def load_labels(filename):
	with open(filename, encoding="utf-8") as f:
		labels = f.readlines()
	labels = [x.strip() for x in labels]
	return labels

def split_sequence(sequence, chunk_size):
    chunks=[]
    for i in range(0, len(sequence), chunk_size):
        chunks.append(sequence[i: i + chunk_size])
    return chunks
		
def print_usage():
    print (
    '''Usage:
    python3 NADI2024-ST3-Scorer.py NADI2024_subtask3_dev_GOLD.txt UBC_subtask3_dev_1.txt
        ''')


if __name__ == '__main__':
    
    truth_file = sys.argv[1]
    prediction_file = sys.argv[2]

    truth = load_labels(truth_file)
    truth = [[t.strip()] for t in truth]

    prediction = load_labels(prediction_file)
    prediction = [p.strip() for p in prediction]
    
    if (len(truth) != len(prediction)):
        print ("both files must have same number of instances")
        exit()

    chunk_size=100 
    truth_chunks= split_sequence(truth, chunk_size)

    truth_Egyptain=truth_chunks[0]
    truth_Emirati=truth_chunks[1]
    truth_Jordanian=truth_chunks[2]
    truth_Palestinian=truth_chunks[3]

    prediction_chunks= split_sequence(prediction, chunk_size)

    prediction_Egyptain=prediction_chunks[0]
    prediction_Emirati=prediction_chunks[1]
    prediction_Jordanian=prediction_chunks[2]
    prediction_Palestinian=prediction_chunks[3]

    ### get scores
    results_Egyptain = bleu_score.compute(predictions=prediction_Egyptain, references=truth_Egyptain)
    results_Emirati = bleu_score.compute(predictions=prediction_Emirati, references=truth_Emirati)
    results_Jordanian = bleu_score.compute(predictions=prediction_Jordanian, references=truth_Jordanian)
    results_Palestinian = bleu_score.compute(predictions=prediction_Palestinian, references=truth_Palestinian)
    overall_results = bleu_score.compute(predictions=prediction, references=truth)

    #write to a text file
    print('Scores:')
    scores = {
              'Overall': overall_results['bleu']*100,
              'Egyptain': results_Egyptain['bleu']*100,
              'Emirati': results_Emirati['bleu']*100,
              'Jordanian': results_Jordanian['bleu']*100,
              'Palestinian': results_Palestinian['bleu']*100, 
             }
    print(scores)

    with open(prediction_file.split("/")[-1].split(".")[0]+"_result.txt", 'w') as score_file:
        score_file.write("Overall: %0.12f\n" % scores["Overall"])
        score_file.write("Egyptain: %0.12f\n" % scores["Egyptain"])
        score_file.write("Emirati: %0.12f\n" % scores["Emirati"])
        score_file.write("Jordanian: %0.12f\n" % scores["Jordanian"])
        score_file.write("Palestinian: %0.12f\n" % scores["Palestinian"])

    
   
