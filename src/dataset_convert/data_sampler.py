'''
given a dataset creates random subset of size x% and output as csv
'''
import pandas as pd
def sample(outfolder, *in_files):
    dataframes=[]
    for in_file in in_files:
        dataframes.append(pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8"))

    df=pd.concat(dataframes)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop('description', axis=1, inplace=True)
    df.drop('name_tpage_domain', axis=1, inplace=True)
    df.drop('label', axis=1, inplace=True)
    print(df.shape)

    s_shared=df.groupby('schemaorg_class', group_keys=False).apply(lambda x: x.sample(frac=0.0005))
    s=df.groupby('schemaorg_class', group_keys=False).apply(lambda x: x.sample(frac=0.02))
    s=s.sample(frac=1)
    s1 = s.iloc[0:10000, :]
    s2 = s.iloc[10000:, :]

    s_shared=s_shared.sort_values(['schemaorg_class', 'page_domain'],
                         ascending=[True, True])
    s1 = s1.sort_values(['schemaorg_class', 'page_domain'],
                         ascending=[True, True])
    s2 = s2.sort_values(['schemaorg_class', 'page_domain'],
                        ascending=[True, True])

    s_shared.to_csv(outfolder+"/sample_shared.csv", sep=',', encoding='utf-8')
    s1.to_csv(outfolder + "/sample_1.csv", sep=',', encoding='utf-8')
    s2.to_csv(outfolder + "/sample_2.csv", sep=',', encoding='utf-8')
    print("ok")

if __name__ == "__main__":
    sample("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/",
           "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv",
           "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv",
           "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/CreativeWork.csv"
           )

