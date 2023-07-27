## Iterative Prompt Learning for Unsupervised Backlit Image Enhancement (ICCV 2023)

[Paper](https://arxiv.org/abs/2303.17569) | [Project Page](https://zhexinliang.github.io/CLIP_LIT_page/) | [Video](https://youtu.be/CHgLtcB9XUA)

[Zhexin Liang](https://zhexinliang.github.io/), [Chongyi Li](https://li-chongyi.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Ruicheng Feng](https://jnjaby.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

S-Lab, Nanyang Technological University

:star: If CLIP-LIT is helpful to your images or projects, please help star this repo. Thanks! :hugs: 

### Update
- **2023.07.27**: Test codes, dataset and enhancement model checkpoint are public available now.

### TODO
- [x] Add training code and config files.
- [x] Add prompt pair checkpoints.

### Dependencies and Installation

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

### Quick Inference
<!-- #### Download Pre-trained Model:
Download the pretrained enhancement model from [[Releases](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) | [Google Drive](https://drive.google.com/drive/folders/1b_3qwrzY_kTQh0-SnBoGBgOrJ_PLZSKm?usp=sharing) | [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/s200094_e_ntu_edu_sg/EvDxR7FcAbZMp_MA9ouq7aQB8XTppMb3-T0uGZ_2anI2mg?e=DXsJFo)] to the `pretrained_models` folder.  -->


#### Prepare Testing Data:
You can put the testing images in the `input` folder. If you want to test the backlit images, you can download the BAID test dataset and the Backlit300 dataset from [[Google Drive](https://drive.google.com/drive/folders/1tnZdCxmWeOXMbzXKf-V4HYI4rBRl90Qk?usp=sharing) | [BaiduPan (key:1234)](https://pan.baidu.com/s/1bdGTpVeaHNLWN4uvYLRXXA)].

#### Testing:

```
python test.py
```
The path of input images and output images and checkpoints can be changed. 

Example usage:
```
python test.py -i ./Backlit300 -o ./inference_results/Backlit300 -c /data/CLIP-LIT/pretrained_models/enhancement_model.pth
```

<!-- ### Training: -->

### Citation
If our work is useful for your research, please consider citing:

    @inproceedings{liang2023cliplit,
        author = {Liang, Zhexin and Li, Chongyi and Zhou, Shangchen and Feng, Ruicheng and Loy, Chen Change},
        title = {Iterative Prompt Learning for Unsupervised Backlit Image Enhancement},
        booktitle = {ICCV},
        year = {2023}
    }

### Contact
If you have any questions, please feel free to reach me out at `zhexinliang@gmail.com`. 