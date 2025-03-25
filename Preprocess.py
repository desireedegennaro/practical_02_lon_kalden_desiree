

import os



def read_data(chunk_sizes=[200, 500, 1000], overlaps=[0, 50, 100]):
    filepath = "data\\"
    word_docs = {}

    file_list = os.listdir(filepath)
    for file in file_list:

        ## Reading in file.
        file_str = ''
        with open(filepath + file, mode="r", encoding='utf-8') as infile:
            for line in infile:
                stripped = line.strip()
                file_str += stripped
        chunking = {}
        ## chunking w overlap
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                chunks = []
                prev_idx = 0
                count = 1
        while prev_idx < len(file_str):
            if prev_idx+chunk_size >= len(file_str): 
                chunk = file_str[prev_idx:]
            else:  
                chunk = file_str[prev_idx:prev_idx+chunk_size]
            chunks.append(chunk)
            prev_idx = (prev_idx+chunk_size)-overlap
        chunks = chunking[f"chunk{chunk_size}_overlap{overlap}"] 
    word_docs[file] = chunking

    return word_docs

read_data()