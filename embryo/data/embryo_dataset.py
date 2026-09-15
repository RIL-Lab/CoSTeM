import os
import sys
import mmcv
import torch
import torch.utils
import copy
import pandas as pd
from mmcv.parallel import collate
from collections.abc import Mapping, Sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from abc import ABCMeta, abstractmethod

# make the repository root importable no matter where the module is imported from
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from embryo.data.pipeline import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 root_path,
                 ann_file,
                 pipeline,
                 test_mode=False,
                 num_classes=None, 
                 start_index=1,
                 modality='RGB',
                 ):
        super().__init__()
        self.root_path = root_path
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.num_classes = num_classes
        self.modality = modality
        self.start_index = start_index

    @abstractmethod
    def load_annotations(self):
        """Load the annotation information according to the ann_file into video infos"""

    @abstractmethod
    def load_json_annotations(self):
        """Load the annotation form json annotation files"""


class EmbryoDataset(BaseDataset):
    """Dataset class for embyro video data
    
    Arguments:
        root_path (str): path to dir that saves the raw videos according to patient code
        ann_file (str): path to the annotation file that saves meta information of videos for each patient.
        pipeline (MMCV pipeline object): the data processing pipeline to preprocess the videos
    """
    def __init__(self, root_path, ann_file, pipeline=None, sample_ratio=1.0, **kwargs):
        super().__init__(root_path, ann_file, pipeline, **kwargs)
        self.sample_ratio = sample_ratio
        self.pipeline = Compose(pipeline) if pipeline is not None else pipeline
        self.video_infos = self.load_annotations()
        # only videos that reach the blastocyst stage are used
        self.video_infos = self.filter_blastocyst_stage()

    def load_annotations(self):
        if self.ann_file.endswith(".json"):
            return self.load_json_annotations()
        
        elif self.ann_file.endswith(".xlsx"):
            return self.load_xlsx_annotations()

    def load_json_annotations(self):
        """Load annotation infomation form the json file

        Returns:
            video_infos (list[dict]): a list of videos, with each video info saved in a dict that has keys as follows:
                                        patient (str): indicate patient code,
                                        total_frames (int): num of frames contains in the video,
                                        duration ([start, end]): indicate the starting and ending record time of the video,
                                        total_length (float): the temporal length of the video,
                                        filename (str): the storing path of the video in disk,
                                        label (int): the class of the video, with "0" means high-quality and "1" means low-quality.

        """
        ann_file_abspath = os.path.join(self.root_path, self.ann_file)
        assert os.path.exists(ann_file_abspath), f"The annotaion file {os.path.basename(ann_file_abspath)} does not exist."
        video_infos = mmcv.load(ann_file_abspath)

        return video_infos
    
    def load_xlsx_annotations(self):
        """
        Read the annotation infoamtion form the excel file and reformulate the items into a list of dict.
        """
        ann_file_abspath = os.path.join(self.root_path, self.ann_file)
        video_infos = pd.read_excel(ann_file_abspath)
        video_infos = video_infos.to_dict(orient="records")

        # build mapping dict, may be used during test and analysis
        embryo_id_list = []
        for info in video_infos:
            term = info["ID"]
            embryo_id_list.append(term)
        embryo_id_list = list(set(embryo_id_list))
        self.num2embryo = dict(zip(list(range(len(embryo_id_list))), embryo_id_list))
        self.embryo2num = dict(zip(embryo_id_list, list(range(len(embryo_id_list)))))

        # parse anno infomation
        for info in video_infos:
            term = info["ID"]
            embryo_ID = self.embryo2num[term]
            info["embryo_ID"] = embryo_ID

            year = info["year"]
            patient, video_idx = term.split("_")
            complete_path = os.path.join(self.root_path, str(year), str(patient), f"embryo_{int(video_idx)}.avi")
            info["filename"] = complete_path

            try:
                quality, grading, female_age = int(info["quality"]), int(info["grading"]), int(info["female_age"])
            except:
                print(info["ID"])

            info["quality"] = quality
            info["grading"] = grading
            info["female_age"] = female_age

            effective_duration = info["effective_duration"].replace(", ", ",").replace("(", "").replace(")", "")
            effective_start, effective_end = map(float, effective_duration.split(","))

            video_duration = info["video_duration"].replace(", ", ",").replace("(", "").replace(")", "")    
            video_start, video_end = map(float, video_duration.split(","))

            info["effective_duration"] = (effective_start, effective_end)
            info["video_duration"] = (video_start, video_end)

        return video_infos   
    
    def filter_blastocyst_stage(self):
        """Keep only the videos whose effective duration reaches the blastocyst stage."""
        new_info = []
        for info in self.video_infos:
            if info["effective_duration"][1] > 110.0:
                new_info.append(info)

        return new_info

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        if self.pipeline:
            results = self.pipeline(results)
        
        return results   


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Sample elements randomly form a given list of indices, without replacement.

    Arguments: 
        indices (sequence): a sequence of indices to sample
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)
    
    def set_epoch(self, epoch):
        self.epoch = epoch


def mmcv_collate(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


if __name__ == "__main__":
    # minimal smoke test: python -m embryo.data.embryo_dataset <root_path> <ann_file>
    root_path = sys.argv[1] if len(sys.argv) > 1 else "embryo/data/embryo_videos"
    anno_file = sys.argv[2] if len(sys.argv) > 2 else "train.xlsx"

    ds = EmbryoDataset(root_path=root_path, ann_file=anno_file, pipeline=None)
    ds.load_annotations()
    print(len(ds.video_infos))
    print(ds[0])
