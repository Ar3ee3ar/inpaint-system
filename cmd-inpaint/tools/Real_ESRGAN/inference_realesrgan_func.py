import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main_upscale(img):
    # determine models according to model names
    model_name = 'RealESRGAN_x4plus'
    denoise_strength = 0.5
    tile = 0
    tile_pad = 10
    pre_pad = 0 
    fp32 = False
    gpu_id=None
    face_enhance = True
    outscale = 4

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    # elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    # elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    # elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    #     netscale = 2
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    # elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    # elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    #     ]

    # determine model paths
    # if args.model_path is not None:
    #     model_path = args.model_path
    # else:
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    # os.makedirs(args.output, exist_ok=True)

    # if os.path.isfile(args.input):
    #     paths = [args.input]
    # else:
    #     paths = sorted(glob.glob(os.path.join(args.input, '*')))

    # for idx, path in enumerate(paths):
    #     imgname, extension = os.path.splitext(os.path.basename(path))
    #     print('Testing', idx, imgname)

    # path = 'upload/new_0_output.jpg'
    # imgname, extension = os.path.splitext(os.path.basename(path))
    # print(imgname,' ',extension)
    output = 'results'
    suffix = 'out'
    ext = 'auto'
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        if face_enhance:
            _, _, output_img = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output_img, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    return output_img