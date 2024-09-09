"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os
import sys
import numpy as np
import torch.distributed as dist
import joblib
o_path = os.getcwd()
sys.path.append(o_path)
print("system path = {}".format(o_path))
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    args_to_log = vars(args)
    print("argumentation information = {}".format(args_to_log))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        diffusion_steps=args.diffusion_steps
    )

    logger.log("start to record loss ...")
    run_bpd_evaluation(logger, model, diffusion, data, args.num_samples, args.clip_denoised, args.analysis_data_save_path, args.log_interval)


def run_bpd_evaluation(logger, model, diffusion, data, num_samples, clip_denoised, save_file_path="tmp", log_interval=10):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    data_metrics_lists = []
    tensors_to_imgs_dict = {}
    fw = open("{}.joblib".format(save_file_path), 'ab+')

    logger.log("save data file path {}.joblib".format(save_file_path))

    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        #batch, cond, t = next(self.data) # need to check the shape of t
        #step_loss = self.one_loss_step(batch, cond, t) # cond is a dictionary

        for b_id in range(batch.shape[0]):

            img_tensor = np.array2string(batch[b_id].cpu().numpy(), precision=4, separator=',', suppress_small=True)

            one_sample_metrics_all_steps = []
            steps_t = []
            batch_imgs_list = []

            for step_id in range(minibatch_metrics['vb'].shape[1]):
                
                # add img 
                if img_tensor in tensors_to_imgs_dict:
                    batch_imgs_list.append(tensors_to_imgs_dict[img_tensor])
                else:
                    img_idx = len(tensors_to_imgs_dict)
                    batch_imgs_list.append(img_idx)
                    tensors_to_imgs_dict[img_tensor] = img_idx
                
                step_vb_score = float(minibatch_metrics['vb'][b_id][step_id])
                step_xstart_mse_loss = float(minibatch_metrics['xstart_mse'][b_id][step_id])
                step_mse_loss = float(minibatch_metrics["mse"][b_id][step_id])

                one_sample_metric = {"vb": step_vb_score, "xstart_mse": step_xstart_mse_loss, "mse": step_mse_loss}

                one_sample_metrics_all_steps.append(one_sample_metric)
                steps_t.append(step_id)
            step_total_bpd = float(minibatch_metrics['total_bpd'][b_id])
            step_prior_bpd = float(minibatch_metrics["prior_bpd"][b_id])
            data_metrics_lists.append((batch_imgs_list, one_sample_metrics_all_steps, steps_t, step_total_bpd, step_prior_bpd))
        
        num_complete += dist.get_world_size() * batch.shape[0]

        if num_complete % log_interval == 0:
            #logger.dumpkvs()
            joblib.dump(data_metrics_lists, fw)
            logger.log("dump {}({}) batch records into {}".format(len(data_metrics_lists), num_complete, save_file_path))
            data_metrics_lists = []


    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True, 
        num_samples=1000, 
        batch_size=1, 
        model_path="",
        analysis_data_save_path="tmp",
        dataset_name="default",
        log_interval=10
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
