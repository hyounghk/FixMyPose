# FixMyPose / फिक्समाइपोज़
Code and dataset for AAAI 2021 paper ["FixMyPose: Pose Correctional Describing and Retrieval"](https://arxiv.org/abs/2104.01703) Hyounghun Kim\*, Abhay Zala\*, Graham Burri, Mohit Bansal.

<p align="center">
  <a href="https://fixmypose-unc.github.io">fixmypose-unc.github.io</a>
</p>
<p align="center">
  <img src="https://github.com/hyounghk/FixMyPose/blob/main/fixmypose_gif.gif">
</p>

## Prerequisites

- Python 3.6
- [PyTorch 1.4](http://pytorch.org/) or Up
- For others packages, please run this command.
```
pip install -r requirements.txt
```

## Dataset
Please download resized images from [here](https://drive.google.com/file/d/1mPEgNW72tRgipW9nGlMNVYlhpOM_R0_5/view?usp=sharing) and unzip in [dataset/fixmypose](./dataset/fixmypose) folder.<br>
Also, you can download full-sized images from [here](https://drive.google.com/file/d/169RkpcjPoOWFc_iGQfnhD8DilK33Q4Aj/view?usp=sharing).

## Usage

To train the models:
```
# pose correction describing model (English)
bash run.sh

# pose correction describing model (Hindi)
bash run_hindi.sh

# target pose retrieval model (English)
bash run_retrieval.sh

# target pose retrieval model (Hindi)
bash run_retrieval_hindi.sh
```
## Task Specific Metric Usage
```
python body_part_match.py ./path/to/transform_files/ [FILES...]
python direction_match.py [FILES...]
python object_match.py [FILES...]
```
[FILES...] - list of files you wish to run the metric on (e.g. output_val_seen.json output_val_unseen.json output_test_unseen.json ...).
The files will be created in the root directory when you finish evaluations.


## Hindi METEOR Calculation
```
cd langeval/cococaption/pycocoevalcap/meteor
python meteor_hindi.py hindi_meteor.json
```
If you uncomment these two [lines](https://github.com/hyounghk/FixMyPose/blob/95edd63a11c2571d72e3fe314fe4679a8028890a/langeval/cococaption/pycocoevalcap/meteor/meteor.py#L55-L56) when evaluating, you can obtain the hindi_meteor.json file in the root folder.

### Evaluation on Test split
Please contact fixmypose.unc@gmail.com for the test split.

## Acknowledgments
Base code is from ["Expressing Visual Relationships via Language" paper's code repository](https://github.com/airsplay/VisualRelationships).<br>
Caption evaluation code is from [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption).
