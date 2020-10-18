# Converts Labelme json files to COCO Dataset File
# Thank you to https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/

import os
import argparse
import json

from labelme import utils
import numpy as np
import glob
import PIL

class labelMeToCoco(object):
    def __init__(self, labelMeJson=[], saveJsonPath="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """

        self.labelMeJson = labelMeJson
        self.saveJsonPath = saveJsonPath
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annId = 1
        self.height = 0
        self.width = 0

        self.SaveJson()
    
    def DataTransfer(self):
        for num, jsonFile in enumerate(self.labelMeJson):
            with open(jsonFile, "r") as fp:
                data = json.load(fp)
                self.images.append(self.Image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    self.annotations.append(self.Annotation(points, label, num))
                    self.annId += 1
        
        # Sort labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.Category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.GetCatId(annotation["category_id"])

    def Image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image
    
    def Category(self, label):
        category = {}
        category["supercategory"] = label[0]
        category["id"] = len(self.categories)
        category["name"] = label[0]
        
        return category
    
    def Annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num
        annotation["bbox"] = list(map(float, self.GetBbox(points)))
        annotation["category_id"] = label[0] # self.getcatid(label)
        annotation["id"] = self.annId

        return annotation
    
    def GetCatId(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1
    
    def GetBbox(self, points):
        polygons = points
        mask = self.PolygonsToMask([self.height, self.width], polygons)
        return self.MaskToBox(mask)
    
    def MaskToBox(self, mask):
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        columns = index[:, 1]

        leftTopRow = np.min(rows) # y
        leftTopColumn = np.min(columns) # x

        rightBottomRow = np.max(rows)
        rightBottomColumn = np.max(columns)

        return [
            leftTopColumn,
            leftTopRow,
            rightBottomColumn - leftTopColumn,
            rightBottomRow - leftTopRow,
        ]

    def PolygonsToMask(self, imageShape, polygons):
        mask = np.zeros(imageShape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        
        return mask

    def DataToCoco(self):
        dataCoco = {}
        dataCoco["images"] = self.images
        dataCoco["categories"] = self.categories
        dataCoco["annotations"] = self.annotations
        
        return dataCoco

    def SaveJson(self):
        print("Saving Coco Json File")
        self.DataTransfer()
        self.dataCoco = self.DataToCoco()

        print(self.saveJsonPath)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.saveJsonPath)), exist_ok=True
        )
        json.dump(self.dataCoco, open(self.saveJsonPath, "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Labelme annotation to coco data json file."
    )

    parser.add_argument(
        "LabelMeImages",
        help="Directory to labelme images and annotation json files.",
        type=str
    )

    parser.add_argument(
        "--output",
        help="Output json file path.",
        default="trainval.json"
    )

    args = parser.parse_args()
    labelMeJson = glob.glob(os.path.join(args.LabelMeImages, "*.json"))
    labelMeToCoco(labelMeJson, args.output)