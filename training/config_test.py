from box import Box

config = {
    "dir": "D:/inpaint_gan/",
    "mask" : "D:/inpaint_gan/car_ds/pic/mask/binary_mask_test.jpg", # binary_mask_test.jpg | 8k_focus_simple2.jpg
    "weightG" : "D:/inpaint_gan/weight/nakorn/adapt_5/bestG_weight.h5",
    "model_name": "p2p",
    "ds_name": "i9-10",
    "random_box": False,
    "random_mask": False,
    "save": True,
    "save_fol": "nakorn/adapt_5_maskN",
    "img": [20],
    "pic_4x" : ""
    }

cfg_test = Box(config)