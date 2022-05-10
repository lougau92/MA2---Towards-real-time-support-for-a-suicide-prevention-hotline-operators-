import sys
import argparse
import os

from googletrans import Translator
translator = Translator()

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