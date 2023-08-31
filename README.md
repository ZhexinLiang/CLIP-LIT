<div align="center">

<h1>Iterative Prompt Learning for Unsupervised Backlit Image Enhancement</h1>

<div>
    <a href='https://zhexinliang.github.io/' target='_blank'>Zhexin Liang</a>&emsp;
    <a href='https://li-chongyi.github.io/' target='_blank'>Chongyi Li</a>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Shangchen Zhou</a>&emsp;
    <a href='https://jnjaby.github.io/' target='_blank'>Ruicheng Feng</a>&emsp;
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy</a>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp; 
</div>

<div>
    :star: <strong>Accepted to ICCV 2023, Oral</strong>
</div>
<div>
    <h4 align="center">
        <a href="https://zhexinliang.github.io/CLIP_LIT_page/" target='_blank'>[Project Page]</a> •
        <a href="https://arxiv.org/abs/2303.17569" target='_blank'>[arXiv]</a> •
        <a href="https://youtu.be/0qbkxNmkNWU" target='_blank'>[Demo Video]</a>
    </h4>
</div>

https://github.com/ZhexinLiang/CLIP-LIT/assets/122451585/a34f6808-e39e-4428-bb16-f607f80a8b9f

https://github.com/ZhexinLiang/CLIP-LIT/assets/122451585/91b9fefd-2822-43a9-85d9-cd542875362a

<!-- <div>
    <img src="assets/Frankfurt.gif" width="100%"/>
</div> -->

<!-- <div class="img">
    <video muted autoplay="autoplay" loop="loop" width="100%">
        <source src="assets/Dolomites_800.mp4" type="video/mp4">
    </video>
</div> -->
<strong>CLIP-LIT trained using only hundreds of unpaired images yields favorable results on unseen backlit images captured in various scenarios.</strong>
<!-- 
<table>
<tr>
    <td><img src="assets/0032_rgb.gif" width="100%"/></td>
    <td><img src="assets/0032_geo.gif" width="100%"/></td>
    <td><img src="assets/0067_rgb.gif" width="100%"/></td>
    <td><img src="assets/0067_geo.gif" width="100%"/></td>
    <td><img src="assets/0021_rgb_dancing.gif" width="98%"/></td>
    <td><img src="assets/0001_rgb_interpolation.gif" width="88%"/></td>
</tr>
<tr>
    <td align='center' width='14%'>Sample 1 RGB</td>
    <td align='center' width='14%'>Sample 1 Geo</td>
    <td align='center' width='14%'>Sample 2 RGB</td>
    <td align='center' width='14%'>Sample 2 Geo</td>
    <td align='center' width='19%'>Novel Pose Generation</td>
    <td align='center' width='19%'>Latent Space Interpolation</td>
</tr>
</table> -->

:open_book: For more visual results, go checkout our <a href="https://zhexinliang.github.io/CLIP_LIT_page/" target="_blank">project page</a>.

---


</div>

## :mega: Updates
- **2023.08.01**: Training codes, and initial checkpoints are publicly available now.
- **2023.07.27**: Test codes, dataset and enhancement model checkpoint are publicly available now.


## :desktop_computer: Requirements

- Pytorch >= 1.13.1
- CUDA >= 11.3
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/ZhexinLiang/CLIP-LIT.git
cd CLIP-LIT

# create new anaconda env
conda create -n CLIP_LIT python=3.7 -y
conda activate CLIP_LIT

# install python dependencies
pip install -r requirements.txt
```

## :running_woman: Inference

### Prepare Testing Data:
You can put the testing images in the `input` folder. If you want to test the backlit images, you can download the BAID test dataset and the Backlit300 dataset from [[Google Drive](https://drive.google.com/drive/folders/1tnZdCxmWeOXMbzXKf-V4HYI4rBRl90Qk?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1bdGTpVeaHNLWN4uvYLRXXA)].

### Testing:

```
python test.py
```
The path of input images and output images and checkpoints can be changed. 

Example usage:
```
python test.py -i ./Backlit300 -o ./inference_results/Backlit300 -c ./pretrained_models/enhancement_model.pth
```

## :train: Training

### Prepare Training Data and the initial weights:
You should download the backlit and reference image dataset and put it under the repo. In our experiment, we randomly select 380 backlit images from BAID training dataset and 384 well-lit images from DIV2K dataset as the unpaired training data. We provide the training data we use at [[Google Drive](https://drive.google.com/drive/folders/1X1tawqmUsn69T24VmHSl_qmEFxGLzMf0?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1a0_mUpoFJszjH1eHfBbJPw)] for your reference.

You should also download the initial prompt pair checkpoint (`init_prompt_pair.pth`) from [[Release](https://github.com/ZhexinLiang/CLIP-LIT/releases/tag/v1.0.0) | [Google Drive](https://drive.google.com/drive/folders/1mImPIUaYbXfZ_CHPvdNK-xKrt94abQO5?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1H4lOrLaYlS0PYTF4pgfSDw)] and put it into `pretrained_models/init_pretrained_models` folder.

After the data and the initial model weights are prepared, you can use the command to change the training data path, fine-tune the prompt and train the model.

If you don't want to download the initial prompt pair, you can train without the initial checkpoints using the command below. But in this way, the number of the total iterations should be at least $50K$ based on our experiments.
 
### Commands
Example usage:
```
python train.py -b ./train_data/BAID_380/resize_input/ -r ./train_data/DIV2K_384/
```
There are other arguments you may want to change. You can change the hyperparameters using the cmd line.

For example, you can use the following command to **train from scratch**.
```
python train.py \
 -b ./train_data/BAID_380/resize_input/ \
 -r ./train_data/DIV2K_384/             \
 --train_lr 0.00002                     \
 --prompt_lr 0.000005                   \
 --eta_min 5e-6                         \
 --weight_decay 0.001                   \
 --num_epochs 3000                      \
 --num_reconstruction_iters 1000        \
 --num_clip_pretrained_iters 8000       \
 --train_batch_size 8                   \
 --prompt_batch_size 16                 \
 --display_iter 20                      \
 --snapshot_iter 20                     \
 --prompt_display_iter 20               \
 --prompt_snapshot_iter 100             \
 --load_pretrain False                  \
 --load_pretrain_prompt False
```

## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:
```
@article{liang2023iterative,
  title={Iterative Prompt Learning for Unsupervised Backlit Image Enhancement},
  author={Liang, Zhexin and Li, Chongyi and Zhou, Shangchen and Feng, Ruicheng and Loy, Chen Change},
  journal={arXiv preprint arXiv:2303.17569},
  year={2023}
}
```

### Contact
If you have any questions, please feel free to reach out at `zhexinliang@gmail.com`. 

<!-- ## :newspaper_roll: License

Distributed under the S-Lab License. See `LICENSE` for more information.

## :raised_hands: Acknowledgements

This study is supported by NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

This project is built on source codes shared by [Style -->
