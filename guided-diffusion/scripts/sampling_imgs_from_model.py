"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os
import sys

o_path = os.getcwd()
sys.path.append(o_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch as th
import torchvision as tv
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(log_suffix="save_generating_imgs_with_grid")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    save_folder = "{}".format("/".join((args.model_path).split('/')[0:-1]))
    checkpoint_name = (args.model_path).split('/')[-1]
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger.log("****sampling {} images from checkpoint={}, save in {}****".format(args.num_samples, checkpoint_name, save_folder))
    
    num_sampling_imgs = 0

    while num_sampling_imgs < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        num_sampling_imgs += int(sample.shape[0])

        tv.utils.save_image(sample/255., "{}/generated_{}_{}.png".format(save_folder, checkpoint_name, num_sampling_imgs),nrow=3)
        print("**********save {}/generated_{}_{}.png************".format(save_folder, checkpoint_name, num_sampling_imgs))
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=64,
        batch_size=64,
        use_ddim=False,
        model_path="",
        data_dir="",
        dataset_name="default"
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
