import os

def read_data(chunk_size=200, overlap=50):
    filepath = "data\\"

    file_list = os.listdir(filepath)
    word_docs = {}

    for file in file_list:

        ## Reading in file.
        file_str = ''
        with open(filepath + file, mode="r", encoding='utf-8') as infile:
            for line in infile:
                stripped = line.strip()
                file_str += stripped
        
        ## breaking up big string into chunk_size w overlap
        prev_idx = 0
        count = 1
        while True:
            if prev_idx+chunk_size > len(file_str): 
                chunk = file_str[prev_idx:]
                word_docs[file+f' chunk{count}'] = chunk
                break
            else:  
                chunk = file_str[prev_idx:prev_idx+chunk_size]
            prev_idx = (prev_idx+chunk_size)-overlap
            word_docs[file+f'chunk{count}'] = chunk
            count+=1

    return word_docs


print(read_data())