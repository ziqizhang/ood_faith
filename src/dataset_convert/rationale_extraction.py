'''
This takes the rationale output files computed by ood_faith, and produces summaries

example rationale files: /home/zz/Work/ood_faith/output/output_place/extracted_rationales/AmazDigiMu/data/topk
'''
import datetime

'''
given a list of json rationale extracted files, merge them into a dataframe and output
the list of unique class labels
'''

import json, pandas,csv,sys, re
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def load_into_df(*rationale_files):
    unique_labels=set()
    df = pandas.DataFrame(columns=['exp_split', 'label_id', 'text', 'full text doc'])
    for f in rationale_files:
        with open(f) as input_file:
            file_contents = input_file.read()

            parsed_json = json.loads(file_contents)

            i=0
            for entry in parsed_json:
                df.loc[i]=[entry['exp_split'],entry['label_id'],entry['text'],entry['full text doc']]
                unique_labels.add(entry['label_id'])
                i+=1
                if i%5000==0:
                    print(str(datetime.datetime.now())+", "+str(i))

                # if i>10000:
                #     break
    return unique_labels, df

'''
Produce analysis stats for the rationales
'''
def quant_analyse(dataframe, labels, outfolder):
    labels=sorted(labels)
    for l in labels:
        print(l)
        selected=dataframe.loc[dataframe['label_id'] == l]
        freq=analyse_word_freq(selected, 'text')
        with open(outfolder+"/{}.csv".format(l.replace("\\s+","_")), 'w', newline='\n', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for k, v in freq.items():
                writer.writerow([k, v, v/len(selected)])

def analyse_word_freq(dataframe, col_name):
    freq = {}
    for index, row in dataframe.iterrows():
        attention = row[col_name]
        unique=set(re.split(r"\s+", attention))
        for w in unique:
            if w in STOPWORDS:
                continue
            if w in freq.keys():
                freq[w]+=1
            else:
                freq[w]=1
    freq=dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
    return freq

if __name__ == "__main__":
    labels, df=load_into_df(sys.argv[1], sys.argv[2])
    quant_analyse(df, labels, sys.argv[3])
