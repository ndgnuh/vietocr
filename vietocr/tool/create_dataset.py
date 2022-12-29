import sys
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import re
from multiprocessing import Process, Queue


def checkImageIsValid(imageBin):
    isvalid = True
    imgH = None
    imgW = None

    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)

        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            isvalid = False
    except Exception as e:
        isvalid = False

    return isvalid, imgH, imgW


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset_(output_path: str,
                   root_dir: str,
                   annotation_path: str,
                   chunk_size: int = 10000):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    context = Queue()

    annotation_path = os.path.join(root_dir, annotation_path)

    def split_annotation(line):
        splits = re.split(r"\s+", line.strip())
        return splits[0], " ".join(splits[1:])

    def write(ctx: Queue, output_path: str, annotations):
        offset = ctx.get()
        target = min(offset + chunk_size, total)
        env = lmdb.open(output_path, map_size=2**40)
        with env.begin(write=True) as tnx:
            iterable = tqdm(
                range(offset, target),
                leave=True,
                initial=offset,
                total=total
            )
            for cnt in iterable:
                imageFile, label = annotations[cnt]
                imagePath = os.path.join(root_dir, imageFile)

                # if not os.path.exists(imagePath):
                #     error += 1
                #     continue

                with open(imagePath, 'rb') as f:
                    imageBin = f.read()
                isvalid, imgH, imgW = checkImageIsValid(imageBin)

                # Skip over this
                # if not isvalid:
                #     error += 1
                #     continue

                imageKey = 'image-%09d' % cnt
                labelKey = 'label-%09d' % cnt
                pathKey = 'path-%09d' % cnt
                dimKey = 'dim-%09d' % cnt

                tnx.put(imageKey.encode(), imageBin)
                tnx.put(labelKey.encode(), label.encode())
                tnx.put(pathKey.encode(), imageFile.encode())
                tnx.put(dimKey.encode(), np.array(
                    [imgH, imgW], dtype=np.int32).tobytes()
                )

            # Update max size
            tnx.put('num-samples'.encode(), str(total).encode())

        ctx.put(target)

    # Actual works
    # read annotation
    with open(annotation_path, 'r') as ann_file:
        lines = ann_file.readlines()
        annotations = [split_annotation(line) for line in lines]

    # multiprocess writing
    count = 0
    total = len(annotations)
    ctx = Queue()
    ctx.put(0)  # Initial offset
    while count < total:
        p = Process(target=write, args=(ctx, output_path, annotations))
        p.start()
        p.join()

        # Update loop state and return the offset for next write
        count = ctx.get()
        ctx.put(count)

    print('Created dataset with %d samples' % total)


def createDataset(outputPath, root_dir, annotation_path):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    annotation_path = os.path.join(root_dir, annotation_path)

    def split_annotation(line):
        splits = re.split(r"\s+", line.strip())
        return splits[0], " ".join(splits[1:])

    with open(annotation_path, 'r') as ann_file:
        lines = ann_file.readlines()
        annotations = [split_annotation(line) for line in lines]

    nSamples = len(annotations)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    error = 0

    pbar = tqdm(range(nSamples), ncols=100,
                desc='Create {}'.format(outputPath))
    for i in pbar:
        imageFile, label = annotations[i]
        imagePath = os.path.join(root_dir, imageFile)

        if not os.path.exists(imagePath):
            error += 1
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        isvalid, imgH, imgW = checkImageIsValid(imageBin)

        if not isvalid:
            error += 1
            continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        pathKey = 'path-%09d' % cnt
        dimKey = 'dim-%09d' % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[pathKey] = imageFile.encode()
        cache[dimKey] = np.array([imgH, imgW], dtype=np.int32).tobytes()

        cnt += 1

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}

    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)

    if error > 0:
        print('Remove {} invalid images'.format(error))
    print('Created dataset with %d samples' % nSamples)
    sys.stdout.flush()
