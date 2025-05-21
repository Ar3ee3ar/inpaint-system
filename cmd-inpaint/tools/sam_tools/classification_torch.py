from ultralytics import YOLO
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from .segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import time

print(os.getcwd())

sam = sam_model_registry["vit_l"](checkpoint="weight_sam/sam_vit_l_0b3195.pth").to(device=torch.device('cpu'))

def obj_detect(img):
    model = YOLO("yolov8n.pt")
    start_box = time.time()
    results = model.predict(source=img, conf=0.25)
    predicted_boxes = results[0].boxes.xyxy
#     # use cv2 to visualize the bounding boxes on the image
#     for box in predicted_boxes:
#         cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#         # image_bgr = cv2.resize(image_bgr, (512, 256))
#         # cv2.imshow("YOLOv8 predictions", image_bgr)
#         # cv2.waitKey(0)
#     end_box = time.time()
#     print('box time: ',(end_box - start_box))
#     cv2.imwrite("D:/inpaint_gan/test_img/segment/"+str(count_num)+"_box"+side+".jpg", image_bgr)
# #     return predicted_boxes, image_bgr
    print(predicted_boxes)
    return predicted_boxes
    

def bbox(img_type, img, xy_box):
    main_dir = "D:/inpaint_gan/"
    count_num = 12
    # model = YOLO("yolov8n.pt")
    # load the SAM model

    mask_predictor = SamPredictor(sam)

    if(img_type == 'per'):
        side = 'top'
        # img_path = "D:/new_car_ds/pano_8k/IMAGE_OUTPUT_per/"+str(count_num)+"_256_source.jpg"
        # img_path = "D:/inpaint_gan/car_ds/pic/Input/LB_0_000001.jpg"
        # img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # # read in the image for visualization
        # image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # run the model on the image
    #     start_box = time.time()
    #     results = model.predict(source=img_path, conf=0.25)
    #     predicted_boxes = results[0].boxes.xyxy

    #     # use cv2 to visualize the bounding boxes on the image
    #     for box in predicted_boxes:
    #         cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    #         # image_bgr = cv2.resize(image_bgr, (512, 256))
    #         # cv2.imshow("YOLOv8 predictions", image_bgr)
    #         # cv2.waitKey(0)
    #     end_box = time.time()
    #     print('box time: ',(end_box - start_box))
    #     cv2.imwrite("D:/inpaint_gan/test_img/segment/"+str(count_num)+"_box"+side+".jpg", image_bgr)
    # #     return predicted_boxes, image_bgr
    #     print(predicted_boxes)

    # def segment(predicted_boxes,image_bgr):

        # transform the YOLOv8 predicted boxes to match input format expected by SAM model
        # image_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        # print(len(predicted_boxes))
        if(True):
            print('start segment')
            start_seg = time.time()
            # transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])


            # run SAM model on all the boxes
            # image_bgr = img_ori
            mask_predictor.set_image(img)
            # yolo format
            # masks, scores, logits = mask_predictor.predict_torch(
            #     boxes = transformed_boxes,
            #     multimask_output=False,
            #     point_coords=None,
            #     point_labels=None
            # ) # this function have upscale image from output model (1024x1024) to original size
            
            # point prompt
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

            input_box = np.array(xy_box)
            masks, _, _ = mask_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )


            # print(masks)
            # print(scores)
            # print(logits)

            # combine all masks into one for easy visualization
            final_mask = None
            if(len(masks)>1):
                for i in range(len(masks)-1):
                    # print(masks[i].shape)
                    # h, w = masks.shape[-2:]
                    # color = np.array([30/255, 144/255, 255/255, 0.6])
                    # mask_image = masks[i].reshape(h, w, 1) * color.reshape(1, 1, -1)
                    # plt.imshow(mask_image)
                    # plt.show()
                    h, w = masks[i].shape[-2:]
                    print(masks[i][0].shape,' ',h,' ',w)
                    # color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                    # masks = masks.reshape(h, w, 1)
                    if final_mask is None:
                        final_mask = np.bitwise_or(masks[i].reshape(h, w, 1), masks[i+1].reshape(h, w, 1))
                    else:
                        final_mask = np.bitwise_or(final_mask, masks[i+1].reshape(h, w, 1))
                    # final_mask = 255 * final_mask.numpy()
                    # # print(final_mask.max())
                    # # print(final_mask.min())        
                    # # print(type(final_mask))
                    # # print(final_mask)
            elif(len(masks)==1):
                h, w = masks[0].shape[-2:]
                final_mask = np.full((img.shape[0],img.shape[1]), False, np.bool_) ## White background
                final_mask = masks[0].reshape(h, w, 1)
            end_seg = time.time()
            print('seg time: ',(end_seg - start_seg))
            print('save seg')
            final_mask = 255 * final_mask # final_mask.numpy()
        else:
            final_mask = np.full((image_bgr.shape[0],image_bgr.shape[1]), 0.0, np.float32) ## White background

        # visualize the predicted masks
        print(final_mask)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_rgb)
        # plt.imshow(final_mask,cmap='gray',alpha=0.7)
        # plt.show()
        # image_save = cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite("D:/inpaint_gan/test_img/segment/"+str(count_num)+"_seg_"+side+".jpg", final_mask):
            raise Exception("Could not write image")
        return final_mask
    # elif(img_type == 'cube'):

    #     sides = ['_R']

    #     for side in sides:
    #         img_path = main_dir + "car_ds/pic/Output_cube/LB_0_"+str(count_num).rjust(6, '0')+side+".jpg"
    #         # img_path = "D:/inpaint_gan/car_ds/pic/Input/LB_0_000001.jpg"
    #         img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #         # read in the image for visualization
    #         image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #         # run the model on the image
    #         start_box = time.time()
    #         results = model.predict(source=img_path, conf=0.25)
    #         predicted_boxes = results[0].boxes.xyxy

    #         # use cv2 to visualize the bounding boxes on the image
    #         for box in predicted_boxes:
    #             cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    #             # image_bgr = cv2.resize(image_bgr, (512, 256))
    #             # cv2.imshow("YOLOv8 predictions", image_bgr)
    #             # cv2.waitKey(0)
    #         end_box = time.time()
    #         print('box time: ',(end_box - start_box))
    #         cv2.imwrite("D:/inpaint_gan/test_img/segment/"+str(count_num)+"_box"+side+".jpg", image_bgr)
    #     #     return predicted_boxes, image_bgr

    #     # def segment(predicted_boxes,image_bgr):

    #         # transform the YOLOv8 predicted boxes to match input format expected by SAM model
    #         image_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    #         print(len(predicted_boxes))
    #         if(len(predicted_boxes) > 0):
    #             print('start segment')
    #             start_seg = time.time()
    #             transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])


    #             # run SAM model on all the boxes
    #             # image_bgr = img_ori
    #             mask_predictor.set_image(img_ori)
    #             masks, scores, logits = mask_predictor.predict_torch(
    #                 boxes = transformed_boxes,
    #                 multimask_output=False,
    #                 point_coords=None,
    #                 point_labels=None
    #             ) # this function have upscale image from output model (1024x1024) to original size

    #             # print(masks)
    #             # print(scores)
    #             # print(logits)

    #             # combine all masks into one for easy visualization
    #             final_mask = None
    #             if(len(masks)>1):
    #                 for i in range(len(masks) - 1):
    #                     if final_mask is None:
    #                         final_mask = np.bitwise_or(masks[i][0], masks[i+1][0])
    #                     else:
    #                         final_mask = np.bitwise_or(final_mask, masks[i+1][0])
    #                     # final_mask = 255 * final_mask.numpy()
    #                     # print(final_mask.max())
    #                     # print(final_mask.min())        
    #                     # print(type(final_mask))
    #                     # print(final_mask)
    #             elif(len(masks)==1):
    #                 final_mask = np.full((image_bgr.shape[0],image_bgr.shape[1]), False, np.bool_) ## White background
    #                 final_mask = np.bitwise_or(final_mask,masks[0][0])
    #             end_seg = time.time()
    #             print('seg time: ',(end_seg - start_seg))
    #             print('save seg')
    #             final_mask = 255 * final_mask.numpy()
    #         else:
    #             final_mask = np.full((image_bgr.shape[0],image_bgr.shape[1]), 0.0, np.float32) ## White background

    #         # visualize the predicted masks
    #         print(final_mask)
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(image_rgb)
    #         plt.imshow(final_mask,cmap='gray',alpha=0.7)
    #         plt.show()
    #         # image_save = cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR)
    #         if not cv2.imwrite("D:/inpaint_gan/test_img/segment/"+str(count_num)+"_seg"+side+".jpg", final_mask):
    #             raise Exception("Could not write image")
        # return final_mask

# bbox('per')
# img_seg = segment(box,img_bgr)
# img_path = "D:/new_car_ds/pano_8k/IMAGE_OUTPUT_per/"+str(12)+"_256_source.jpg"
# xy_box = [189,65,323,351]
# img_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
# image_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
# # read in the image for visualization
# # image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
# predict_box = obj_detect(img_ori)
# final_mask = bbox("per", img_ori, xy_box)
# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb)
# plt.imshow(final_mask,cmap='gray',alpha=0.7)
# plt.show()

