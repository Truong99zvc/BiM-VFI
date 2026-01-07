<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS420.Q12 - SELECTED TOPICS IN COMPUTER VISION</b></h1>

## TABLE OF CONTENTS
* [Course Introduction](#course-introduction)
* [Instructor](#instructor)
* [Students](#students)
* [Project](#project)
* [Environment Setting](#environment-setting)
* [Dataset](#dataset)
* [Pretrained Model](#pretrained-model)
* [Evaluation](#evaluation)
* [Training](#training)
* [Demo](#demo)
  - [Command Line Demo](#command-line-demo)
  - [Web Demo](#web-demo)
* [License](#license)

## COURSE INTRODUCTION
<a name="course-introduction"></a>
* **Course Name**: Selected Topics in Computer Vision
* **Course Code**: CS420
* **Class Code**: CS420.Q12
* **Academic Year**: 2025 - 2026
* **Start Date**: September 8, 2025
* **End Date**: December 27, 2025

## INSTRUCTOR
<a name="instructor"></a>
* **TS. Mai Tiến Dũng** - *dungmt@uit.edu.vn*

## STUDENTS
<a name="students"></a>
| Student ID | Name                | Github                                               | Email                   |
|:----------:|:-------------------:|:----------------------------------------------------:|:-----------------------:|
| 22521587   | Trương Phúc Trường  | [Truong99zvc](https://github.com/Truong99zvc/)      | 22521587@gm.uit.edu.vn  |
| 22521571   | Võ Đình Trung       | [votrung654](https://github.com/votrung654/)         | 22521571@gm.uit.edu.vn  |

## PROJECT
<a name="project"></a>
**Project Name**: BiM-VFI: BIDIRECTIONAL MOTION FIELD-GUIDED FRAME INTERPOLATION FOR VIDEO

This repository contains the implementation of BiM-VFI, a bidirectional motion field-guided frame interpolation method for video with non-uniform motions. The project is based on the CVPR 2025 paper by Wonyong Seo, Jihyong Oh, and Munchurl Kim.

## ENVIRONMENT SETTING
<a name="environment-setting"></a>
To run this project, you need to set up your environment as follows:

```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch==2.4.1 torchvision==0.19.1
conda install cupy -c conda-forge
```

### Additional Libraries for Web Demo
To run the web demo, you need to install the following additional libraries:

```bash
pip install flask werkzeug pillow scikit-image
```

**Note**: `opencv-python` and `torch` are already included in the main environment setup above.

## DATASET
<a name="dataset"></a>
### Download
You can download the Vimeo90K dataset used for training and testing from the following link:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)

### Preparation
After downloading the dataset, organize it according to the project structure. The dataset should be placed in the `data` directory.

## PRETRAINED MODEL
<a name="pretrained-model"></a>
Pre-trained model can be downloaded from [here](https://drive.google.com/file/d/18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC/view?usp=sharing).

Place the downloaded model file (`bim_vfi.pth`) in the `pretrained` directory.

## EVALUATION
<a name="evaluation"></a>
Desired evaluation can be done by replacing `benchmark_dataset` section in `cfgs/bim_vfi_benchmark.yaml`.
* `name`: Name of benchmark datasets. The datasets that can be benchmarked are [_vimeo_, _vimeo\_septuplet_, _snu\_film_, _snu\_film\_arb_, _xtest_].
* `args`:
  * `root_path`: Path of each dataset.
  * `split`: Desired splits to evaluate. [_test_, _val_] for _vimeo_ and _vimeo\_septuplet_, [(_easy_), _medium_, _hard_, _extreme_] for _snu\_film_ and _snu\_film\_arb_, and [_single_, _multiple_] for _xtest_.
  * `pyr_lvl`: 3 for vimeo, 5 for snu_film, and 7 for xtest.
* `save_imgs`: `True` if you want to save interpolation results, else `False`. It takes much more time to save images.

Then, run below:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## TRAINING
<a name="training"></a>
For single GPU training,
```bash
python main.py --cfg cfgs/bim_vfi.yaml
```

For multiple GPU training with GPU number 0, 1, 2, 3,
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_vfi.yaml
```

To run with wandb, fill in wandb.yaml and run with
```bash
python main.py --cfg cfgs/bim_vfi.yaml -w
```

## DEMO
<a name="demo"></a>
### Command Line Demo
<a name="command-line-demo"></a>
Custom videos in multiple images or video format can be interpolated as follows.

First, set demo root directory as follows:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...
  - ...

Then, replace `root_path` in `cfgs/bim_vfi_demo.yaml` to desired data root, and run:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

### Web Demo
<a name="web-demo"></a>
The project includes a web-based demo interface for easy video frame interpolation. To run the web demo:

1. **Navigate to the web_demo directory**:
   ```bash
   cd web_demo
   ```

2. **Run the Flask application**:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open your web browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`)

The web demo provides the following features:
- **Image Pair Interpolation**: Upload two images and generate interpolated frames between them
- **Video Interpolation**: Upload a video file and interpolate frames between consecutive frames
- **Frame Sequence Interpolation**: Upload multiple frames and interpolate between them
- **Model Selection**: Choose between different pre-trained models (pretrained, trained_330)
- **Customizable Parameters**: Adjust the number of interpolated frames and output FPS

**Note**: Make sure you have installed all the required libraries mentioned in the [Environment Setting](#environment-setting) section, including the additional libraries for web demo (Flask, werkzeug, Pillow, scikit-image).

## LICENSE
<a name="license"></a>
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr).

