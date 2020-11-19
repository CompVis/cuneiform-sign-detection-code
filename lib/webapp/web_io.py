import numpy as np
import os


def make_folder(res_path):
    # create folder, if it does not exist
    if not os.path.exists(res_path):
        os.makedirs(res_path)


def export_detections_to_web(destination, all_boxes, lbl2lbl):
    # open file stream
    with open(destination, 'w') as examples_file:
        # write header
        examples_file.write("label, x1, y1, x2, y2, score \n")
        # iterate over all classes
        for class_idx, class_detections in enumerate(all_boxes):
            # avoid empty detections or background class
            if len(class_detections[0]) > 0 and class_idx > 0:
                # over all detections per class
                for detection in class_detections[0]:
                    # write detection to file
                    #print("{} {} {} {} {} {}".format(int(lbl2lbl[class_idx]), *detection))
                    examples_file.write("{}, {}, {}, {}, {}, {} \n".format(int(lbl2lbl[class_idx]), *detection))


def convert_to_all_boxes(seg_gen_annos, relative_bboxes, scale, num_labels):
    all_boxes = [[] for _ in range(num_labels)]

    for anno_idx, anno_rec in seg_gen_annos.iterrows():
        # [x1, y1, x2, y2, score]
        box = np.zeros((1, 5))
        box[0, :4] = np.array(relative_bboxes[anno_idx]) * scale
        box[0, 4] = anno_rec.det_score
        # assign to class
        all_boxes[anno_rec.newLabel].append(box)

    # for each class stack list of bounding boxes together
    all_boxes = [np.stack(el).squeeze(axis=1) if len(el) > 0 else el for el in all_boxes]

    return all_boxes


def convert_alignments_for_eval(detections, total_labels=240):
    # convert from RANSAC format (Nx9) to all_boxes
    all_boxes = [[] for _ in range(total_labels)]

    for temp in detections:
        # temp: [ID, cx, cy, score, x1, y1, x2, y2, idx]

        # copy data to _new_ all_boxes
        box = np.zeros((1, 5))
        box[0, :4] = temp[4:8]
        box[0, 4] = temp[3]
        all_boxes[np.int(temp[0])].append(box)

    # for each class stack list of bounding boxes together
    all_boxes = [np.stack(el).squeeze(axis=1) if len(el) > 0 else el for el in all_boxes]

    return all_boxes

