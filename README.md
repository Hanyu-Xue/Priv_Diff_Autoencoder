



# New update (2023/06/05)
Add new file for fixing bugs: run_anonymization.ipynb

<del> # New update (2023/06/04) </del>
<del> Add new file for multi-gpu v0: run_anonymization-multi_GPU.ipynb <del>

# New update (2023/05/29)
Add new file: Evaluation.ipynb

# New update (2023/05/25)
Add new file: FaRL_features_for_Pair.ipynb

# New update (2023/04/25)

This upate includes the new paper content:
1.  Environment set up + new model FaRL [https://github.com/FacePerceiver/FaRL].
2.  Experiments note book:
> 2.1 Sample.ipynb
> 2.2 Test_quality.ipynb

`Note: Back up environment first`


# New update (2023/03/19)

This update includes the resolution of the issue with backward gradient propagation, as well as a modification to the structure of the feature extractor network, which now utilizes an ID+Semantic encoder mode.

To update, please follow the steps:

0. Check the base.py. Make sure to comment out the line "with th.no_grad():":

```
    def ddim_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = th.tensor([i] * len(img), device=device)
#             with th.no_grad():
            out = self.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=_kwargs,
                eta=eta,
            )
            out['t'] = t
            yield out
            img = out["sample"]

```

1. Download the New model LatentMapperNew1.py from '/disentanglement/Models/'.

2. Download the Pretrained parameter ("ID_VEC_ffhq70000_mlp.pt") from Google Drive: https://drive.google.com/file/d/10Ayh--HX_27UvOyUZDM_W-O9p0eqG6eL/view?usp=share_link

3. Download the new script Train_ID.ipynb. (Train Loop 1 is already done and saved in 'ID_VEC_ffhq70000_mlp.pt'. Start with Train Loop 2.)

# New update (2023/03/16)

Please downlaod the LatentMapperNew from /disentanglement/Models/ and replace it with the old one in your loacl folder.
Please download the New_train.ipynb. And change the model path based on your enivornment.

# New update (2023/03/15)

Please download the file id_loss.py from '/disentanglement/Losses/' and replace it with the old one in your loacl folder.

# New update (2023/03/14)

Please download the new model params from google drive and put them into the folder checkpoints:
1. mlp model (https://drive.google.com/drive/folders/1MugUVHGn45eGklW7vJYvVjWm9eHLPRrr?usp=share_link)
2. attr model (https://drive.google.com/drive/folders/1fH4W9zNcxdB33N7pM3ORy2z4nDIIRwE2?usp=share_link)

Please download the model LatentMapperNew.py from disentanglement/Models/.

please download the New_train.ipynb.

Run the code in New_train.ipynb. (If you are running the code locally, pleae ignore the colab opertations, for example, Mount drive and install all packages.)
