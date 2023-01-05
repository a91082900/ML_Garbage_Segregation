from extract import *
import os
import cv2
import multiprocessing

os.makedirs('Preprocessed', exist_ok=True)

def proc(f, dirpath):
    img = cv2.imread(os.path.join(dirpath, f))
    # output = garbage_extract(img)
    output = garbage_extract_no_preprocess(img)
    path = os.path.join('Preprocessed', *dirpath.split(os.sep)[1:])
    name = os.path.splitext(f)[0] + '_preprocessed.jpg'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, name), output)
    print(os.path.join(dirpath, f))

if __name__ == '__main__':
    pool = multiprocessing.Pool(8)

    for dirpath, dirnames, filenames in os.walk('Camera'):
        for f in filenames:
            if f.endswith('.jpg'):
                pool.apply_async(proc, args=(f, dirpath))

    pool.close()            
    pool.join()
