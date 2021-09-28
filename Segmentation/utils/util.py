import os

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def write_log(log_file, string):
    print(string)
    log_file.write(string+'\n')
    log_file.flush()