import process_sequence_fasta as pro_seq_fasta
from keras.models import model_from_json
from keras.preprocessing import sequence
import pandas as pd
import subprocess
import sys
import numpy as np
import pickle

"""
Author: Carlos Garcia-Perez
Date: 26.06.2019 this script make predictions on new sequence data
                 first version of the script

This script has these functions:

    * process_data
    * nuc2aa
    * fix_fasta_line

"""


def label_decoder(probs):
    lb_encoder = pickle.load(open('/training_code/lbl_encoder.pkl', 'rb'))
    probs *= 100.0
    results = []
    show_results = []
    for i, c in enumerate(probs):
        r = []
        for idx, p in enumerate(c):
            p = round(p, 2)
            if p > 0.001:
                r.append((i + 1, idx, '%0.02f' % p))
        results.append(r)

    for s in results:
        if len(s) > 1:
            labels_class = lb_encoder.inverse_transform([i[1] for i in s])
            seq = 'sequence ' + str(s[0][0]) + ':' + '\t'
            for idx, p in enumerate(s):
                seq = seq + labels_class[idx] + '\t' + str(p[2]) + ', '
            show_results.append(seq[:-2])
        else:
            labels_class = lb_encoder.inverse_transform([s[0][1], 0])
            seq = 'sequence ' + str(s[0][0]) + ':' + '\t' + labels_class[0] + '\t' + s[0][2]
            show_results.append(seq)
    
    nf = open("/data/prediction_results.txt", "w")
    for line in show_results:
        nf.write(line + '\n')
    nf.close()


def fix_fasta_line(seq):
    if '*' in seq:
        seq = seq.replace('*', '')
    if 'X' in seq:
        seq = seq.replace('X', '')

    return seq


def nuc2aa(file):
    subprocess.run('transeq -sequence ' + file + ' -outseq /training_code/to_predict.fasta -frame=1 -sformat pearson', shell=True)


def process_data(file):
    seqs = []
    with open(file) as f:
        seq = f.readline().strip()
        seq = seq.replace('\t','|')
        seq = seq + '\t'
        for line in f:
            line = line.strip()
            if line[0] == ">":
                s1 = seq.split('\t')
                seq_id = s1[0].split("|")[0]
                s1[1] = fix_fasta_line(s1[1])
                sequence = pro_seq_fasta.to_word_index(s1[1])
                seqs.append([seq_id, sequence])
                line = line.replace('\t','|')
                seq = line + '\t'
            else:
                seq = seq + line

        s1 = seq.split('\t')
        seq_id = s1[0].split("|")[0]
        s1[1] = fix_fasta_line(s1[1])
        sequence = pro_seq_fasta.to_word_index(s1[1])
        seqs.append([seq_id, sequence])

        df = pd.DataFrame(seqs, columns=['seq_id', 'sequence'])

    return df


file = '/data/dataset_test_silva.fasta' #sys.argv[1]
file2 = '/training_code/to_predict.fasta'
# process nuc file

nuc2aa(file)
df = process_data(file2)

X = np.array(df['sequence'])
X = sequence.pad_sequences(X, maxlen=763)

json_file = open('/training_code/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("/training_code/model.h5")

predictions_classes = model.predict_classes(X)
predictions_probs = model.predict(X)

label_decoder(predictions_probs)
