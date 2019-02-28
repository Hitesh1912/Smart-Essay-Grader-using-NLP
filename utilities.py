import os
import glob

def write_to_file(filename, data):
    target_dir = r"Results/"
    filename = target_dir + filename + ".txt"
    with open(filename, encoding='utf-8-sig', mode='w') as fp:
        for character, count in data.items():
            fp.write('{},{}\n'.format(character, count))


def write_dict_to_file(filename, data):
    target_dir = r"Results/"
    filename = target_dir + filename + ".txt"
    with open(filename, encoding='utf-8-sig', mode='w') as fp:
        for character, count in data:
            fp.write('{},{}\n'.format(character, count))


def normal_write_to_file(filename, data):
    target_dir = r"Results/"
    filename = target_dir + filename + ".txt"
    with open(filename, encoding='utf-8', mode='w') as fp:
        for item in data:
            fp.write('{}\n'.format(item))



def fileToCollection(filename):
    contents = open(filename, 'r', encoding='utf-8')
    collection = contents.readlines()
    collection_dict ={}
    for line in collection:
        if line.strip() == '': continue
        line = line.strip().split()
        # print (line)
        gram = line[0]
        # print (qId)
        freq= line[2]
        if gram !='Character':
            if gram not in collection_dict.keys():
                collection_dict[gram] = [freq]
            else:
                collection_dict[gram].append(freq)
    # print (collection_dict)
    return collection_dict



def combine_files(filename):
    result_filename = filename
    # read_files = glob.glob("gutenberg/*.txt")
    read_files = glob.glob("brown/*.txt")
    with open(result_filename, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())


def rename_files(dir):
    filelist = os.listdir(os.path.abspath(dir))
    ext = '.txt'
    print(filelist)
    for fname in filelist:
        os.rename(dir+fname, dir+fname + ext)







