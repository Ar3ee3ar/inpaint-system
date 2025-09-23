# proprocess pipeline
    > deepfill preprocess pipeline (recommend)
        1. python data_aug_save.py -> gen dataset image
        2. python single_create_list.py --mode "per" --config_name new-hii_aug_ds --fol_img new-hii_aug_ds --mask new-hii_aug_ds -> สร้าง config dataset
        (optional) combine 
        python mix_create_list.py --mode combine --want_config test_c1 test_c2 --new_conf_name mix_c1c2_test  
        3.  python group_folder.py --config "config name" --mode "df" -> จัดกลุ่มไฟล์ตาม format deepfill --isCombine <True if dataset from multiple dataset>(train test val)

    > pix2pix preprocess pipeline
        1. python data_aug_save.py -> gen dataset image
        2. python single_create_list.py --mode "per" --config_name new-hii_aug_ds --fol_img new-hii_aug_ds --mask new-hii_aug_ds -> สร้าง config dataset
        (optional) combine 
        python mix_create_list.py --mode combine --want_config test_c1 test_c2 --new_conf_name mix_c1c2_test  

    ! reverse process
    python group_folder.py --config "config name" --mode "p2p" -> จัดกลุ่มไฟล์ตาม format p2p (one folder)

# training pipeline
    > deepfill (recommend)
        - first time setting
            1. cd deepfill & python -m venv .venv
            2. .venv/scripts/activate
            3. pip install -r requirements.txt
        - come back to train model
            1. cd deepfill & python -m venv venv
            2. .venv/scripts/activate
            3. edit inpaint.yml in deepfill in category:
            - dataset: '../car_ds/pic/<PicFolderName>'
            - val: true if you want to view validation results in tensorboard
            - ckpt_dir: 'logs/<checkpoint_folder>/models'
            - max_iters: number of training step (1) 
            - val_psteps: amount of picture to validate per step 
            - img_shapes: [width, height, 3]
            - batch_size: amount of image to train per step
            4. python train.py
            5. Model for inpaint system can find in G.h5 in 'logs/<checkpoint_folder>/models'
    - pix2pix
        1. edit config.py in category:
            - dir: absolute path of folder
            - config_file: config file name from preprocess pipeline
            - wandb
                : True (if you have wandb account)
                    add .env file and setting data same as .env-sample
                : False (logs will save to tensorboard)
            - ckpt_path: '<root_folder>/logs/<logs_folder>/training_checkpoints'
            - save: save model every 'save' steps
            - step: amount of epoch t train model
            - img_size: 512 (this model only use square size image)
            - batch_size: amount of image to train per sub-steps
        2. python training.py

