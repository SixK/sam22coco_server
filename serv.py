import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
from pycocotools import coco
import cv2
from PIL import Image
import numpy as np
import json

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

checkpoint = "./model/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

def masksToAnnotation(masks):
    annotations=[]
    id=0
    for mask in masks:
        contours, _ = cv2.findContours(mask["segmentation"].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert the contour to the format required for segmentation in COCO format
        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
            segmentation.append([int(coord) for pair in contour_pairs for coord in pair])

        # Define annotation in COCO format
        annotation = {
            "id": id,
            "image_id": 0,
            "category_id": 1,
            "segmentation": segmentation,
            "width": 800,
            "height": 600,
            "area": int(cv2.contourArea(contours[0])),
            "bbox": [int(x) for x in cv2.boundingRect(contours[0])],
            "metadata": {},
            "color": "#f4311f",
            "iscrowd": 0
        }
        annotations.append(annotation)
        id+=1

    return annotations

# here category name must match 1 category from coco_annotator
# create a sam category or change category name below
def createCoco(annotations):
    coco='''
{
    "coco": {
        "categories": [
            {
                "id": 1,
                "name": "sam",
                "supercategory": null,
                "metadata": {},
                "color": "#1c6bfb"
            }
        ],
        "images": [
            {
                "id": 0,
                "width": 800,
                "height": 600,
                "file_name": "",
                "path": "",
                "license": null,
                "fickr_url": null,
                "coco_url": null,
                "date_captured": null,
                "metadata": {}
            }
        ],
        "annotations": '''+json.dumps(annotations)+'''
        }
    }'''
    return coco

app = Flask(__name__)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route('/', methods=['POST'])
def process_image():
    print(dir(request.files))
    # img_file = request.files['file']
    img_file = request.files['image']

    print(img_file)

    img = Image.open(img_file.stream).convert('RGB')
    print('img:', img.width, img.height)
    img = np.asarray(img)
    print(img)

    masks = mask_generator.generate(img)
    annot=masksToAnnotation(masks)

    print(annot)
    coco=createCoco(annot)

    print("coco:", coco)
    return json.loads(coco)


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
