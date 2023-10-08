import cv2
import json
import glob
import os
from pprint import pprint

if __name__ == "__main__":
    root = "data/football"
    video_paths = list(glob.iglob("{}/*/*.mp4".format(root)))
    anno_paths = list(glob.iglob("{}/*/*.json".format(root)))
    output_path = "football_yolo"
    if os.path.isdir(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "images"))
        os.mkdir(os.path.join(output_path, "labels"))

    for video_id, (video_path, anno_path) in enumerate(zip(video_paths, anno_paths)):
        counter = 1
        video = cv2.VideoCapture(video_path)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        #print(num_frames)
        with open(anno_path, "r") as json_file:
            json_data = json.load(json_file)
        image_width  = json_data["images"][0]["width"]
        image_height = json_data["images"][0]["height"]
        #pprint(type(json_data)) # Print the information of json file as a dictionary
        while video.isOpened():
            flag, frame = video.read() # frame is the numpy arr
            if not flag:
                break
            cv2.imwrite(os.path.join(output_path,"images","{}_{}.jpg".format(video_id+1, counter)), frame)
            objects = [item for item in json_data["annotations"] if item["image_id"] == counter and item["category_id"] > 2]
            # # print(objects)
            # # print(len(objects))
            with open(os.path.join(output_path, "labels", "{}_{}.txt".format(video_id+1,counter)), "w") as txt_file:
             for obj in objects:
                 bbox = obj["bbox"]
                 xmin, ymin, w, h = bbox
                 xmin /= image_width
                 ymin /= image_height
                 w /= image_width
                 h  /= image_height
                 xcent = xmin + w/2
                 ycent = ymin + h/2

                if obj["category_id"] == 3:
                    cls = 0
                else:
                    cls = 1
                txt_file.write("{} {:6f} {:6f} {:6f} {:6f}\n".format(cls, xcent, ycent, w, h))

             #Test only
             xmax = xmin + w
             ymax = ymin + h
             xmin *= image_width
             ymin *= image_height
             xmax *= image_width
             ymax *= image_height
             cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(xmax)), color=(0, 0, 255), thickness=1)

            #     #xmin, ymin, xmax, ynax = bbox
            #     # cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax),int(xmax)), color = (0,0,255), thickness = 5)
            #     # x_cent, y_cent, w, h = bbox
            #     # xmin = x_cent - width/2
            #     # ymin = y_cent - height/2
            #     # xmax = x_cent + width/2
            #     # ymax = y_cent + height/2
            #
            #     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(xmax)), color=(0, 0, 255), thickness=5)
            # cv2.imwrite("test.jpg", frame)
            counter += 1