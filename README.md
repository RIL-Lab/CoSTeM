# CoSTeM — Time-Lapse Video-Based Embryo Grading via Complementary Spatial-Temporal Pattern Mining

Official training code of the MICCAI 2025 paper:

> **Time-Lapse Video-Based Embryo Grading via Complementary Spatial-Temporal Pattern Mining**
> Yong Sun, Yipeng Wang, Junyu Shi, Zhiyuan Zhang, Yanmei Xiao, Lei Zhu, Manxi Jiang, Qiang Nie
> MICCAI 2025 · [arXiv:2506.04950](https://arxiv.org/abs/2506.04950)

**CoSTeM** grades an embryo directly from its full-length time-lapse microscopy (TLM) video by
mining two complementary patterns: the static **spatial** pattern (morphology) and the dynamic
**temporal** pattern (morphokinetics).

---

## 1. Method

**Input.** One TLM video per embryo. Frames are sampled from 16 h to 144 h after insemination
with a stride of 2 h, giving **64 frames** per video, resized / cropped to `224 × 224`.

**Pipeline** (`CoSTeM` in [`embryo/models/costem.py`](embryo/models/costem.py)):

| Stage | Module | Output |
| --- | --- | --- |
| Frame encoding | frozen `openai/clip-vit-base-patch16` image encoder, hidden states of layers `L = {3, 6, 9, 12}` | `B·T × (N+1) × 768` |
| **Bypass Adaptation Network** | `BypassAdaptationNetwork` — one MLP adaptor + channel reduction per layer | patch tokens `P` (morphological), class tokens `S` (morphokinetic) |
| **MCAE** — Mixture of Cross-Attentive Experts | `MixtureOfCrossAttentiveExperts` — `M = 8` spatial experts attend to the patch tokens of each frame, an MLP router predicts the belonging weight of every expert | `P̂` (`T × 768`) |
| **TSB** — Temporal Selection Block | `TemporalSelectionBlock` — `K = 8` learnable temporal experts and `L = 2` Temporal Selection Layers (cross-attention + FFN) select the decisive frames | spatial pattern (`768`-d) |
| **Temporal Transformer** | `TemporalTransformer` — ViViT-style self-attention over the class-token sequence (`32` tokens, `384`-d, `4` layers) | temporal pattern (`384`-d) |
| Classifier | `Linear(1152 → 256) → GELU → Linear(256 → num_classes)` | logits |

The two branches are the paper's **morphological branch** (patch tokens, spatial pattern) and
**morphokinetic branch** (class tokens, temporal pattern); their outputs are concatenated along
the channel dimension and fed to the MLP classifier.

**Training objective**

```
L_total = L_cls + lambda * L_div
L_cls   = CrossEntropyLoss(weight=[1.0, 2.25, 3.24], label_smoothing=0.1)
L_div   = mean |cosine similarity| between the K temporal experts
```

`lambda` is controlled by `--div_loss_weight` (default `0.1`). `L_div` encourages the temporal
experts to select distinct frames.

**Optimisation.** `AdamW(lr=2.5e-5, betas=(0.9, 0.98), weight_decay=0.0)` with a cosine schedule
and 5 warm-up epochs; 50 epochs, batch size 8, single-process `torchrun` by default.

**Tasks.** `--task Grading` (default, 3 classes `poor / fair / good`, label column `grading`) or
`--task Evaluation` (2 classes, label column `quality`).

---

## 2. Repository structure

```
.
├── README.md
└── embryo
    ├── data
    │   ├── embryo_dataset.py       # EmbryoDataset: reads the xlsx annotations, builds video paths
    │   └── pipeline.py             # mmcv-style pipeline: frame sampling, decoding, augmentation, normalization
    ├── models
    │   ├── costem.py               # CoSTeM: model, dataloader, optimizer/scheduler, train & validation loops
    │   ├── base_model.py           # BaseModel interface
    │   ├── clip_image_encoder.py   # CLIP ViT-B/16 image encoder (build_image_encoder)
    │   └── modules.py              # BypassAdaptationNetwork, MixtureOfCrossAttentiveExperts,
    │                               # TemporalSelectionBlock, TemporalTransformer
    ├── utils
    │   ├── logger.py               # logging helper
    │   └── tools.py                # meters, ClassificationReporter, checkpoint saving
    ├── scripts
    │   ├── train.sh                # ★ launcher of a single training run
    │   └── train.py                # training entry point (arguments + DDP + main loop)
    ├── environment.yml             # full conda environment (exact versions)
    └── requirements.txt            # pip requirements snapshot
```

---

## 3. Installation

Tested with **Python 3.10 + PyTorch 2.4.0 (cu124) + CUDA 12.4**.

```bash
# Option 1: recreate the conda environment (recommended)
conda env create -f embryo/environment.yml
conda activate NewEmbryo

# Option 2: pip
pip install -r embryo/requirements.txt
```

Key dependencies: `torch==2.4.0+cu124`, `torchvision==0.19.0+cu124`, `mmcv-full==1.7.0`,
`decord==0.6.0` (reads the `.avi` videos), `timm==1.0.12`, `transformers==4.47.0`,
`einops`, `imgaug`, `openpyxl`, `pandas`, `scikit-learn`, `tensorboard`, `termcolor`.

> The first run needs internet access: the CLIP ViT-B/16 configuration and weights are
> downloaded from the Hugging Face hub (see [Section 5](#5-pretrained-image-encoder)).

---

## 4. Dataset

The embryo TLM dataset (videos + `train.xlsx` / `val.xlsx` annotations) is released together
with this repository. Unpack it into `embryo/data/embryo_videos/`, or keep it anywhere and pass
`ROOT_PATH` explicitly when launching a run.

### 4.1 Directory layout

```
<ROOT_PATH>/
├── train.xlsx                 # training annotations
├── val.xlsx                   # validation annotations
├── 2022/
│   └── F12345/
│       ├── embryo_1.avi
│       └── embryo_2.avi
└── 2023/
    └── F15140/
        └── embryo_5.avi
```

The path of a video is built from its annotation row as
`<ROOT_PATH>/<year>/<patient>/embryo_<idx>.avi`, where `<patient>_<idx>` is the `ID` column
(e.g. `F12345_1`).

### 4.2 Annotation columns

Both `train.xlsx` and `val.xlsx` must contain:

| Column | Type | Description |
| --- | --- | --- |
| `ID` | str | `patient_embryoIndex`, e.g. `F12345_1` |
| `year` | int | year of the cycle, used to build the video path |
| `female_age` | int | maternal age (not used for training) |
| `quality` | int | binary label, used with `--task Evaluation` |
| `grading` | int | 3-class grade, used with `--task Grading` (default) |
| `effective_duration` | str | usable interval, e.g. `(44.4, 136.3)` |
| `video_duration` | str | full video interval, e.g. `(0.0, 136.3)` |

Extra columns (`FSH`, `AMH`, `clinical_result`, ...) are ignored. Durations are strings of the
form `(start, end)`, in the same time unit as the sampling arguments (hours).

### 4.3 Two rules that affect the loaded samples

* **Frame sampling** — `SampleFrames(sample_start=16, sample_end=144, interval=2)`, i.e.
  `np.arange(16, 144, 2)` → **64** time slots, linearly mapped to frame indices through
  `video_duration`. Time slots outside `effective_duration` are masked (`mask = 0`) and the
  corresponding frames are replaced by zeros.
  If you change the range or the stride you **must** also change `--frame_seq_len` (default 64),
  otherwise the temporal modules receive a different sequence length than they were built for.
* **Blastocyst-stage filter** — `EmbryoDataset.filter_blastocyst_stage()` keeps only the videos
  whose `effective_duration` ends **after 110**; day-3 videos are dropped.

You can check that your annotations parse correctly with:

```bash
python -m embryo.data.embryo_dataset <ROOT_PATH> train.xlsx
```

---

## 5. Pretrained image encoder

The frozen image encoder is initialised from
`embryo/pretrained_models/clip_vit_base_patch16.ckpt`, a dictionary of the form
`{"state_dict": ...}` whose keys are prefixed with `vision_model.`.

The file is **not** stored in the repository:

* **Automatic (default)** — if the file is missing, the weights of
  `openai/clip-vit-base-patch16` are downloaded from the Hugging Face hub and only the vision
  tower is kept.
* **Manual** — create it once with:

  ```bash
  python - <<'PY'
  import torch
  from transformers import CLIPModel
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
  torch.save({"state_dict": model.state_dict()},
             "embryo/pretrained_models/clip_vit_base_patch16.ckpt")
  PY
  ```

Use `--pretrained_ckpt /path/to/other.ckpt` to point to a different file.

---

## 6. Training

### 6.1 With the launcher script

```bash
cd embryo/scripts

# minimal: only the dataset location is required
ROOT_PATH=/path/to/embryo_videos bash train.sh

# full example
ROOT_PATH=/path/to/embryo_videos \
OUTPUT_DIR=../experiments \
EXP_NAME=CoSTeM.seed_3407 \
SEED=3407 \
EPOCHS=50 \
BATCH_SIZE=8 \
CUDA_VISIBLE_DEVICES=0 \
bash train.sh
```

`train.sh` simply calls

```bash
torchrun --nproc-per-node 1 --master_port 29571 train.py \
    --exp_name "$EXP_NAME" --seed "$SEED" --task Grading --num_classes 3 \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" \
    --root_path "$ROOT_PATH" --output_dir "$OUTPUT_DIR"
```

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ROOT_PATH` | `../data/embryo_videos` | dataset root (videos + `train.xlsx` / `val.xlsx`) |
| `OUTPUT_DIR` | `../experiments` | where logs, configs and checkpoints are written |
| `EXP_NAME` | `CoSTeM.seed_<SEED>` | experiment name, dots become nested folders |
| `SEED` | `3407` | random seed |
| `EPOCHS` | `50` | number of epochs |
| `BATCH_SIZE` | `8` | batch size per GPU |
| `LR` | `2.5e-5` | learning rate |
| `TASK` | `Grading` | `Grading` (3 classes) or `Evaluation` (2 classes) |
| `NUM_CLASSES` | `3` | number of classes |
| `NPROC` | `1` | processes (= GPUs) per node |
| `MASTER_PORT` | `29571` | `torchrun` port |
| `EXTRA_ARGS` | empty | extra flags forwarded to `train.py` |

### 6.2 Direct call

```bash
cd embryo/scripts
torchrun --nproc-per-node 1 --master_port 29571 train.py \
    --exp_name CoSTeM.seed_3407 \
    --seed 3407 \
    --root_path /path/to/embryo_videos \
    --output_dir ../experiments
```

Multi-GPU (single node, 8 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 bash train.sh
```

### 6.3 Main arguments of `train.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--task` | `Grading` | `Grading` (3 classes) or `Evaluation` (2 classes) |
| `--num_classes` | `3` | number of classes |
| `--epochs` / `--warmup_epochs` | `50` / `5` | epochs / warm-up epochs |
| `--batch_size` | `8` | batch size per GPU |
| `--accumulation_steps` | `1` | gradient accumulation |
| `--lr` / `--weight_decay` | `2.5e-5` / `0.0` | learning rate / weight decay |
| `--label_smoothing` | `0.1` | label smoothing |
| `--div_loss_weight` | `0.1` | `lambda` of the diversity loss `L_div` |
| `--sample_start` / `--sample_end` / `--sample_stride` | `16` / `144` / `2` | sampling window (h) and stride |
| `--frame_seq_len` | `64` | number of sampled frames, must match the sampling arguments |
| `--input_size` | `224` | input resolution |
| `--seed` | `3407` | random seed |
| `--pretrained_ckpt` | `embryo/pretrained_models/clip_vit_base_patch16.ckpt` | CLIP image encoder weights |

`train.sh` is always executed from `embryo/scripts`, so its relative defaults resolve to
`embryo/data/embryo_videos` and `embryo/experiments`.

---

## 7. Outputs

Each run writes to `<OUTPUT_DIR>/<EXP_NAME with dots as folders>/<YYYY_MM_DD_HH_MM_SS>/`:

```
config.yaml                 # all arguments of the run
log_rank0.txt               # training / validation log (per-epoch classification report)
best.pth                    # checkpoint with the best validation macro-F1
events.out.tfevents.*       # tensorboard events
```

```bash
tensorboard --logdir <OUTPUT_DIR>
```

Metrics are **macro-averaged** accuracy / precision / recall / F1
(`ClassificationReporter` in [`embryo/utils/tools.py`](embryo/utils/tools.py)).

> `best.pth` is selected epoch-by-epoch on the validation macro-F1 (the *best* checkpoint),
> which carries an optimistic bias. Report this caveat when using the number as a final result,
> but compare different methods consistently under this same *best* protocol.

---

## 8. Notes and FAQ

* **`torchrun` is required.** The script initialises a NCCL process group, so
  `python train.py` will fail. Use `torchrun --nproc-per-node 1 ...` even on a single GPU.
* **GPU memory.** 64 frames × 224² × batch 8 fits on a single 32 GB V100. If you run out of
  memory, lower `--batch_size` and increase `--accumulation_steps`.
* **`num_workers`** is hard-coded to 12 in `CoSTeM.build_dataloader`; lower it if your machine
  has few CPU cores.
* **mmcv** must match your CUDA / PyTorch build; if `mmcv-full==1.7.0` fails to install, use a
  pre-built wheel, e.g.
  `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html`.
* **decord cannot read a video** — make sure the `.avi` codec is supported by decord / ffmpeg.
  A minimal check: `python -m embryo.data.pipeline /path/to/embryo_1.avi`.
* **Reproducibility.** The seed is applied to `torch`, `numpy`, `random` and CUDA.
  `cudnn.benchmark=True`, so runs are most stable on identical hardware.

---

## 9. Citation

```bibtex
@inproceedings{sun2025costem,
  title     = {Time-Lapse Video-Based Embryo Grading via Complementary Spatial-Temporal Pattern Mining},
  author    = {Sun, Yong and Wang, Yipeng and Shi, Junyu and Zhang, Zhiyuan and Xiao, Yanmei and Zhu, Lei and Jiang, Manxi and Nie, Qiang},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025},
  eprint    = {2506.04950},
  archivePrefix = {arXiv}
}
```

## 10. Licence

The code is released for academic use. The dataset is subject to the terms of the contributing
clinical centre; please contact the authors before any commercial use.
