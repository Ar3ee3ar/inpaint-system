from ultralytics import YOLO
import numpy as np
# import matplotlib.pyplot as plt

from .segment_anything import SamPredictor


def obj_detect(img:np.ndarray,model:object) -> np.ndarray:
    """
    Performs object detection on an input image using the provided model.
    Args:
        img (np.ndarray): The input image as a NumPy array.
        model (object): The object detection model with a 'predict' method.
    Returns:
        np.ndarray: An array of predicted bounding boxes in xyxy format.
    """

    # model = YOLO("yolov8n.pt")
    results = model.predict(source=img, conf=0.25)
    predicted_boxes = results[0].boxes.xyxy
    return predicted_boxes
    

def bbox(img_type:str, img:np.ndarray, xy_box:np.ndarray,model:object) ->np.ndarray:
    """
    Generates a segmentation mask for a given bounding box or set of bounding boxes using a SAM (Segment Anything Model) predictor.
    Args:
        img_type (str): The type of input, either "per" for a single bounding box or "per_yolo" for YOLO-format bounding boxes.
        img (np.ndarray): The input image as a NumPy array.
        xy_box (np.ndarray): The bounding box coordinates. For "per", this should be a single box; for "per_yolo", this should be an array of boxes in YOLO format.
        model (object): The pre-loaded SAM model object.
    Returns:
        np.ndarray: The final segmentation mask as a NumPy array, scaled to 255 for visualization.
    Notes:
        - For "per", the function uses a single bounding box to predict the mask.
        - For "per_yolo", the function transforms YOLO-format boxes and predicts masks for all boxes.
        - All predicted masks are combined into a single mask for visualization.
    """

    sam = model
    mask_predictor = SamPredictor(sam)
    
    # point prompt --------------------------------------------------
    # input_point = np.array([[256, 100],[256,200], [256, 300]])
    # input_label = np.array([0,1, 0])
    # masks, scores, logits = mask_predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True,
    # )
    # h, w = masks.shape[-2:]
    # # color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # masks = masks.reshape(h, w, 1)

    if (img_type == "per"):
        mask_predictor.set_image(img)
        input_box = np.array(xy_box)
        masks, _, _ = mask_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
    elif (img_type == "per_yolo"):
        # yolo format ---------------------------------------------------------------
        # run SAM model on all the boxes
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(xy_box, img.shape[:2])
        masks, scores, logits = mask_predictor.predict_torch(
            boxes = transformed_boxes,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        ) # this function have upscale image from output model (1024x1024) to original size

    # combine all masks into one for easy visualization
    final_mask = None
    if(len(masks)>1):
        for i in range(len(masks)-1):
            h, w = masks[i].shape[-2:]
            if final_mask is None:
                final_mask = np.bitwise_or(masks[i].reshape(h, w, 1), masks[i+1].reshape(h, w, 1))
            else:
                final_mask = np.bitwise_or(final_mask, masks[i+1].reshape(h, w, 1))
    elif(len(masks)==1):
        h, w = masks[0].shape[-2:]
        final_mask = np.full((img.shape[0],img.shape[1]), False, np.bool_) ## White background
        final_mask = masks[0].reshape(h, w, 1)
    final_mask = 255 * final_mask # final_mask.numpy()
    return final_mask

