# 3D Object Tracking on the Waymo Open Dataset

An approach to 3D Object Tracking which is evaluated on segments on the Waymo Open Dataset.
The approach is heavily influenced by [Weng's Baseline for 3D Multi-Object Tracking](https://github.com/xinshuoweng/AB3DMOT).

### Setup

Create a virtual environment and install dependencies

```
python3 -m venv venv
source venv/bin/active
pip install -r requirements.txt
```

### Usage

Download a dataset segment (.tfrecord file) and run the tracker evaluation with

```
python main.py \
  --tracker predictive \
  --segment path/to/segment.tfrecord \
  --dropout 0.2
```


