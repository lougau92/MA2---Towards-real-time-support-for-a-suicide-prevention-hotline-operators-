import sys
import argparse
import os
import pandas as pd

# from googletrans import Translator
# translator = Translator()
translator = 0

def translate_file(fpath):
    file_to_translate = open(fpath, 'r')
    lines = file_to_translate.readlines()
    

    translate_file = os.path.splitext(fpath)[0] + "_eng" + os.path.splitext(fpath)[1]
    tfile = open(translate_file, 'a')

    for line in lines:
        translated_text = translator.translate(line, src='nl')
        tfile.write(translated_text.text + "\n")
    

    file_to_translate.close()
    tfile.close()


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid file path")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=file_path)


    args = parser.parse_args()


    if args.f is not None:
        translate_file(args.f)
      
# louis code  
image_folder_path = "C:/Users/louis/github_vs/113_data/"
en_categories_df = pd.read_csv(image_folder_path + "CDS_translation_EN_NE.tsv", sep="\t")
en_categories_df["CDS (Nederlands)"] = en_categories_df["CDS (Nederlands)"].apply(lambda x: x.split('", "'))
s = en_categories_df.apply(lambda x: pd.Series(x["CDS (Nederlands)"]),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'CDS (NED)'
en_categories_df = en_categories_df.drop("CDS (Nederlands)", axis=1).join(s).reset_index(drop=True)
en_categories_df["CDS (NED)"] = en_categories_df["CDS (NED)"].apply( lambda x : x.replace('[',""))
en_categories_df["CDS (NED)"] = en_categories_df["CDS (NED)"].apply( lambda x : x.replace(']',""))
en_categories_df["CDS (NED)"] = en_categories_df["CDS (NED)"].apply( lambda x : x.replace('"',""))
en_categories_df["CDS (NED)"] = en_categories_df["CDS (NED)"].str.strip()

def translate_cds(cds,disp =False):
    en_cds = []
    ne_cds = []
    for c in cds:
        trans = en_categories_df[en_categories_df['CDS (NED)']==c]
        if len(trans) ==0: 
            if disp: print(c)
            continue
        trans_str = str(trans['CDS (English)'].values).replace('[', " ")
        trans_str = trans_str.replace(']', " ")
        trans_str = trans_str.replace("' '", "/")
        trans_str = trans_str.replace("'", "").strip()       
        en_cds.append(trans_str)
        ne_cds.append(c)
    print("Number of translated columns: ",len(en_cds)," out of ",len(cds))
    return en_cds,ne_cds

def categorise(df_all,disp =False):
    for category in en_categories_df['categories'].unique():
        category_cols = en_categories_df[en_categories_df['categories']== category]['CDS (NED)'].str.strip().to_list()
        common_cols = list(set(category_cols).intersection(set(df_all.columns)))
        if disp: print("category =",category,"common_cols =",len(common_cols))
        df_all[category] = df_all[common_cols].sum(axis=1)
    return df_all
  
def translate(df_all):      
    cols = df_all.columns.tolist()
    cols_EN,cols_NE = translate_cds(cols)
    df_all.rename(columns = dict(zip(cols_NE,cols_EN)) , inplace=True)
    return df_all
