import sentencepiece as spm
import argparse

# Hàm detokenize sử dụng sentencepiece
def load_spm(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def detokenize_line(sp, token_line):
    pieces = token_line.strip().split()
    return sp.decode(pieces)

def detokenize_file(model_path, input_file, output_file):
    sp = load_spm(model_path)

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            detok = detokenize_line(sp, line)
            fout.write(detok + '\n')

# Hàm main với argparse
def main():

    model = 'processed_data/spm_model.model'
    inputs = ['processed_data/test.lo']
    outputs = ['detokenized/test.lo']
    for input, output in zip(inputs, outputs):
        detokenize_file(model, input, output)

if __name__ == '__main__':
    main()
