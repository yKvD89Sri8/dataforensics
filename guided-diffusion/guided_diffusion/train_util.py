import copy
import functools
import gc
import os
import joblib 

import numpy as np
import blobfile as bf
import torch as th
import torchvision as tv
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion.script_util import NUM_CLASSES

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        class_cond=None, 
        num_temp_samples=None, 
        use_ddim=None, 
        image_size=None,
        clip_denoised=None
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.class_cond = class_cond
        self.num_temp_samples = num_temp_samples
        self.use_ddim = use_ddim
        self.image_size = image_size
        self.clip_denoised = clip_denoised

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        try:
            while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                batch, cond = next(self.data)
                self.run_step(batch, cond) # cond is a dictionary
                
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Log the intermediate image generation
                    self.sampling_images_in_training(self.class_cond, self.num_temp_samples, self.use_ddim, self.image_size, self.clip_denoised)
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            # Save the last checkpoint if it wasn't already saved.
            if (self.step - 1) % self.save_interval != 0:
                self.save()
        except ValueError:
            logger.log("**************finish**************")
    
    def sample_test_images(self, num_imgs=9):
        self.sampling_images_in_training(self.class_cond, num_imgs, self.use_ddim, self.image_size, self.clip_denoised)
        print("finish sample_test_images")

    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    """
    compute the loss 
    """
    def loss_analyze(self, save_file_path):
        data_loss_lists = []
        tensors_to_imgs_dict = {}
        fw = open("{}.joblib".format(save_file_path), 'ab+')
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            batch, cond, t = next(self.data) # need to check the shape of t
            step_loss = self.one_loss_step(batch, cond, t) # cond is a dictionary
            batch_imgs_list = []
            for b_id in range(batch.shape[0]):

                img_tensor = np.array2string(batch[b_id].numpy(), precision=4, separator=',', suppress_small=True)
                
                if img_tensor in tensors_to_imgs_dict:
                    batch_imgs_list.append(tensors_to_imgs_dict[img_tensor])
                else:
                    img_idx = len(tensors_to_imgs_dict)
                    batch_imgs_list.append(img_idx)
                    tensors_to_imgs_dict[img_tensor] = img_idx                     
                     
            data_loss_lists.append((batch_imgs_list, step_loss, t.detach().cpu())) # step_loss = {"step_losses": ,"step_images_log_prob":}
            
            if self.step % self.log_interval == 0:
                #logger.dumpkvs()
                joblib.dump(data_loss_lists, fw)
                logger.log("dump {} batch records into {}".format(len(data_loss_lists), save_file_path))
                data_loss_lists = []
            if self.step % self.save_interval == 0:
                # disable the model save
                # self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        with open("{}_img2ids.joblib".format(save_file_path), 'ab') as fw_img2ids:
            joblib.dump(tensors_to_imgs_dict, fw_img2ids)
            logger.log("finishing the loss analysis extraction, save tensors_to_imgs_dict in {}".format("{}_img2ids.joblib".format(save_file_path)))
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            print("**********save model(step={})**********".format(self.step))

    """
    compute the loss for each step
    return a steps_loss dict: {"step_losses": ,"step_images_log_prob":}
    """
    def one_loss_step(self, batch, cond, t):
        steps_loss = self.forward_diffusion_steps_losses(batch, cond, t)
        #self.log_step()
        return steps_loss

    """
    compute a forward loss for each time steps
    """
    def forward_diffusion_steps_losses(self, batch, cond, t):
        
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            t = t.to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            #_, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial( # setup a function to compute the corresponding loss value
                self.diffusion.training_losses,
                self.ddp_model, # model
                micro,  # x_start
                t,   # timestep
                model_kwargs=micro_cond, 
            )
            # here, losses is a dictionary
            if last_batch or not self.use_ddp: # what does it mean that self.use_ddp
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            micro = micro.cpu()
            
            step_losses = losses['loss'].detach().cpu().numpy()
            # calculate the likelihood score  for one step
            step_images_log_prob = losses['images_log_prob'].detach().cpu().numpy()
            del losses
            th.cuda.empty_cache()
            #loss = (losses["loss"] * weights).mean()
            return {"step_losses": step_losses, "step_images_log_prob": step_images_log_prob}
            #self.mp_trainer.backward(loss)      

    """
    random sample a few examples to check the model performance
    num_samples: this should be very small, since we only need to monitor the sampling quality
    """
    def sampling_images_in_training(self, class_cond, num_samples, use_ddim, image_size, clip_denoised):
        # we implement image sampling and save here
        self.model.eval()
        logger.log("sampling {} images in model training...".format(num_samples))
        
        model_kwargs = {}
        if class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(self.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            self.model,
            (num_samples, 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # This is a trick to process the images.
        intermediate_imgs_save_path = "{}/intermediate_imgs_steps={}.png".format(get_blob_logdir(), self.step)
        tv.utils.save_image(sample/255., intermediate_imgs_save_path, nrow=3, padding=2)

        self.model.train()
        logger.log("sampling in model training complete")
        

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial( # setup a function to compute the corresponding loss value
                self.diffusion.training_losses,
                self.ddp_model, # model
                micro,  # x_start
                t,   # timestep
                model_kwargs=micro_cond,
            )
            # here, losses is a dictionary
            if last_batch or not self.use_ddp: # what does it mean that self.use_ddp
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            # exclude our added scores
            log_loss_dict( 
                self.diffusion, t, {k: v * weights for k, v in losses.items() if k!="images_log_prob"} 
            )
            self.mp_trainer.backward(loss)


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

# this function is to output the log_loss, which can be regarded as the likelihood scores?
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)