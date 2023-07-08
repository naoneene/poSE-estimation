import os
import subprocess
from multiprocessing import Process

NUM_PROCESS = 12 # define number of processes to speed up

VIDEOS_DIR = 'h36m' # path to Human3.6M videos

def find_files():
    SUBJECTS = [7] #1, 5, 6, 7, 8, 9, 11

    vids = []
    ff = {}

    for S in SUBJECTS:
        fs = os.listdir(os.path.join(VIDEOS_DIR, f'S{S}'))

        removed = [x for x in fs if x.startswith('_ALL')]
        fs = [x for x in fs if x not in removed]

        for i in range(len(fs)):
            fs[i] = os.path.join(VIDEOS_DIR, f'S{S}', str(fs[i]))

        vids += fs

    for vid in vids:
        folder = vid.replace('Videos', 'Images')
        folder = folder.replace(' ', '_')
        folder = folder[:-4]
        ff[vid] = folder

    with open('all.txt', 'w') as f:
        for k,v in ff.items():
            f.write(f"{k} - {v}\n")

    return split_dict_equally(ff, chunks=NUM_PROCESS)

def split_dict_equally(input_dict, chunks=2):
    return_list = [dict() for idx in range(chunks)]
    idx = 0
    for k,v in input_dict.items():
        return_list[idx][k] = v
        if idx < chunks-1:
            idx += 1
        else:
            idx = 0
    return return_list

def run_process(task):
    for vid, folder in task.items():

        with open('processed.txt', 'w') as f:
            f.write(f"{vid} - {folder}\n")

        os.system(f'mkdir -p {folder}')
        print(f'\n ======================== Converting ...{vid}... ========================')
        subprocess.run(['ffmpeg', '-i', vid, '-q:v', '3', folder + '/%6d.jpg'])

if __name__ == '__main__':

    tasks = find_files()

    for i in range(NUM_PROCESS):
        p = Process(target=run_process, args=(tasks[i],))
        p.start()
        p.join()
