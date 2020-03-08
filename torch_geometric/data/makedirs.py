import os
import os.path as osp
import errno
import shutil


def cleardir(path):
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

def makedirs(path):
    try:
        path = osp.expanduser(osp.normpath(path))
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
