import cv2
import numpy as np


def add_bbox(num, img, bbox, rank, lock):
    bbox = bbox.astype(int)
    cat_name = 'person'
    cat_size = cv2.getTextSize(cat_name + '#0000', cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, thickness=1)[0]

    if lock is 1:
        color = (0, 0, 255) if (num is 0) else (255, 160, 160)
    if lock is 2:
        color = (0, 0, 255) if (num == 0 or num == 1) else (255, 160, 160)
    if lock is 3:
        color = (0, 0, 255) if (num == 0 or num == 1 or num == 2) else (255, 160, 160)

    txt = '{} #{:d}'.format(cat_name, rank)

    if bbox[1] - cat_size[1] - 5 < 0:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] + 1),
                      (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 5),
                      color, thickness=-1)
        cv2.putText(img, txt,
                    (bbox[0], bbox[1] + cat_size[1] + 1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] - cat_size[1] - 5),
                      (bbox[0] + cat_size[0], bbox[1] - 1),
                      color, thickness=-1)
        cv2.putText(img, txt,
                    (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.rectangle(img,
                  (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]),
                  color, thickness=3)

    return img

def add_bbox_lock_on(num, img, bbox, id, lock):
    bbox = bbox.astype(int)
    cat_name = 'person'
    cat_size = cv2.getTextSize(cat_name + '#0000', cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, thickness=1)[0]

    if lock is 1:
        color = (0, 0, 255) if (num is 0) else (255, 160, 160)
    if lock is 2:
        color = (0, 0, 255) if (num == 0 or num == 1) else (255, 160, 160)
    if lock is 3:
        color = (0, 0, 255) if (num == 0 or num == 1 or num == 2) else (255, 160, 160)

    txt = '{} #{:d}'.format(cat_name, id)

    if id > 0:
        if bbox[1] - cat_size[1] - 5 < 0:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] + 1),
                          (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 5),
                          color, thickness=-1)
            cv2.putText(img, txt,
                        (bbox[0], bbox[1] + cat_size[1] + 1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] - cat_size[1] - 5),
                          (bbox[0] + cat_size[0], bbox[1] - 1),
                          color, thickness=-1)
            cv2.putText(img, txt,
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(img,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      color, thickness=3)

    return img

