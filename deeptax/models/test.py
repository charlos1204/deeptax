import numpy as np
import pickle
import process_sequence_fasta as pro_seq_fasta
import sys

#fname = '/data/RDP_sequences_filtered.fasta'

#sequence_df = pro_seq_fasta.process_fasta(fname)

#X = np.array(sequence_df['sequence'])

#max_len = max([len(s) for s in X])
#classes = 27

#info = [classes, max_len]


info = pickle.load(open("/data/info.pkl", 'rb'))

print(info)

intro = int(sys.argv[1])

if intro == 1:
    print(intro)
else:
    print('Es otra madre, no jalo....')

#pickle.dump(info, open("/data/info.pkl", 'wb'), protocol=4)
