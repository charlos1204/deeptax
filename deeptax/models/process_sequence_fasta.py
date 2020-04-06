import pandas as pd
from sklearn.utils import shuffle

"""
Author: Carlos Garcia-Perez
Date: 25.06.2016 complete description to function fix_fasta_file
      12.06.2019 first version of the script

fasta file processor

This script process a fasta forma like.

This script has these functions:
    * process_fasta(f)
    * fix_fasta_file(f)
    * to_word_index(sequence)
"""


def process_fasta(file, type):
    """
    This function process a fasta file.
    must have a header followed by the sequence. The sequence must be free of * and gaps (-) if this is not the case,
    run the fix_fasta_file function.

    example of a fasta format sequence:
            >1_sample|database|bacteria_name
            STCSSEMRRDVEEYRWRRQPPGITLTFMPESVGSKQDIPWTPTMSISCWAALLSSEANANRPPGEYGRKIKTQRNRGPAQAVDDVDFDATRRTLPGLDMYG

    :param file: name of the fasta file
    :return: return a data frame with the cols bacteria, seq_id, sequence
    """
    seqs = []
    with open(file) as f:
        seq = f.readline().strip()
        seq = seq + '\t'
        for line in f:
            line = line.strip()
            if line[0] == ">":
                s1 = seq.split('\t')
                bac = s1[0].split("|")[2]
                sequence = to_word_index(s1[1], type)
                seqs.append([bac, sequence])
                seq = line + '\t'
            else:
                seq = seq + line

        s1 = seq.split('\t')
        bac = s1[0].split("|")[2]
        sequence = to_word_index(s1[1], type)
        seqs.append([bac, sequence])

        df = pd.DataFrame(seqs, columns=['bacteria', 'sequence'])
        df = shuffle(df)

    return df


def fix_fasta_file(file):
    """
    This function remove "*" and "-" and "X" fom sequences and save into a new file
    must have a header followed by the sequence.

    example of a fasta format sequence:
            >1_sample|database|bacteria_name
            STCSSEMRRDVEEYRWRRQPPGITLTFMPESVGSKQDIPWTPTMSISCWAALLSSEANANRPPGEYGRKIKTQRNRGPAQAVDDVDFDATRRTLPGLDMYG

    :param file: name of the fasta file
    """
    fix_seq = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if '*' in line:
                line = line.replace('*', '')
            if 'X' in line:
                line = line.replace('X', '')
            fix_seq.append(line)

    nf = open("./new_sequences.fasta", "w")
    for line in fix_seq:
        nf.write(line + '\n')
    nf.close()


def to_word_index(sequence, type):
    if type == 'aa':
        amino_alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                          'Y']

        amino_to_index = dict((c, i) for i, c in enumerate(amino_alphabet))
        amino_index_encoded = [amino_to_index[char] for char in sequence]

        return amino_index_encoded
    else:
        nuc_alphabet = ['A', 'G', 'T', 'C', 'U','N']

        #nuc_to_index = dict((c, i) for i, c in enumerate(nuc_alphabet))
        nuc_to_index = dict((c, (i + 1)) for i, c in enumerate(nuc_alphabet))
        nuc_index_encoded = [nuc_to_index[char] for char in sequence if char in nuc_alphabet]

        return nuc_index_encoded
