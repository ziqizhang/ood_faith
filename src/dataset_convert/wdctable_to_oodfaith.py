'''
For converting the wdctable instance classification dataset (created by omaima fallatah)
to nfolds used for rationale extraction

example input data file: /home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML
'''

import pandas as pd
import sys, json, re
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import nltk
from bs4 import BeautifulSoup

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z.]')
STOPWORDS = set(nltk.stopwords.words('english'))
def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    text = ' '.join([i for i in text.split() if not i.isdigit()])
    new = text.replace("_", " ")
    return new
def convert_wdctable_corpus(in_file, nfold,
                            col_name, col_desc, col_label, outfolder):
    skf5=StratifiedKFold(n_splits=nfold, random_state=42, shuffle=True)
    skf2=StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

    filename=in_file[in_file.rindex("/")+1:]
    subfolder=outfolder+"/"+filename[:filename.index(".")]

    #read csv
    X=[]
    y=[]
    labels = set()
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    for index, row in df.iterrows():
        X.append(row)
        y.append(row[col_label])
        labels.add(row[col_label])

    labels=list(labels)
    #create 8:2 splits

    for i, (train_index, test_index) in enumerate(skf5.split(X, y)):
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        train=[]
        for train_i in train_index:
            train.append(X[train_i])
        test=[]
        for test_i in test_index:
            test.append(X[test_i])
        data_train = make_train_or_test(train,"train", labels, col_label, col_name)
        data_test=make_train_or_test(test, "test", labels, col_label, col_name)
        data_dev = make_train_or_test(test, "dev", labels, col_label, col_name)
        outfolder=subfolder+"/{}".format(i)
        Path(outfolder).mkdir(parents=True, exist_ok=True)

        with open(outfolder + "/train.json".format(i, filename), 'w') as the_file:
            the_file.write(json.dumps(data_train, indent=4))
        with open(outfolder + "/test.json".format(i, filename), 'w') as the_file:
            the_file.write(json.dumps(data_test, indent=4))
        with open(outfolder + "/dev.json".format(i, filename), 'w') as the_file:
            the_file.write(json.dumps(data_dev, indent=4))


def make_json_object(count, label_id, text, train_or_test, label, json_list):
    item = {"annotation_id": train_or_test + "_" + str(count),
            "exp_split": train_or_test,
            "text": text,
            "label": label_id,
            "label_id": label}
    json_list.append(item)


def make_train_or_test(dataset, train_or_test, label_list, label_col_index, text_col_index):
    data = []
    count = 0
    for row in dataset:
        label = row[label_col_index]
        string= str(row[text_col_index])
        string= re.sub('[^0-9a-zA-Z.]+', ' ', string).replace("\\s+"," ").strip()

        make_json_object(count, label_list.index(label), string,
                         train_or_test, label, data)
        count += 1
    return data


'''
/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv
5
5
2
3
/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/ood_faith_folds


/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv
5
5
2
3
/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/ood_faith_folds


/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/CreativeWork.csv
5
5
2
3
/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/ood_faith_folds

'''
if __name__ == "__main__":
    convert_wdctable_corpus(sys.argv[1],int(sys.argv[2]),
                            int(sys.argv[3]),int(sys.argv[4]),
                            int(sys.argv[5]),
                            sys.argv[6])



