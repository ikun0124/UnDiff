"""
Network Traffic Dataset.

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the traffic 1-dimension dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from torch.utils.data import Dataset,DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class Traffic2DFlowDataset(Dataset):
    """Btech Dataset class.
    """
    def __init__(
        self,
        train_data_path: str | Path,
        valid_data_path: str | Path,
        test_data_path: str | Path,
        train_label_path: str | Path,
        valid_label_path: str | Path,
        test_label_path: str | Path,
        dataset_type: str,
        flow_num: int,
    ) -> None:
        super().__init__()

        self.train_data_path: str | Path = train_data_path
        self.valid_data_path: str | Path = valid_data_path
        self.test_data_path: str | Path = test_data_path
        self.train_label_path: str | Path =  train_label_path
        self.valid_label_path: str | Path = valid_label_path
        self.test_label_path: str | Path = test_label_path
        self.dataset_type: str = dataset_type
        self.flow_num: int = flow_num
        self.data = self._read_data_npy()
        self.label = self._read_label_npy()

    def _read_data_npy(self):
        if self.dataset_type == 'train':
            data = np.load(self.train_data_path)
        elif self.dataset_type == 'valid':
            data = np.load(self.valid_data_path)
        elif self.dataset_type == 'test':
            data = np.load(self.test_data_path)
        else:
            raise RuntimeError('here error.')


        return data[:, :self.flow_num, :, :]

    def _read_label_npy(self):
        if self.dataset_type == 'train':
            label = np.load(self.train_label_path)
        elif self.dataset_type == 'valid':
            label = np.load(self.valid_label_path)
        elif self.dataset_type == 'test':
            label = np.load(self.test_label_path)
        else:
            raise RuntimeError('here error.')

        return label[:, :self.flow_num]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """organize to fit
        """
        item: dict = {}
        
        item['data'] = self.data[index, :, :, :]
        item['label'] = self.label[index, 0]

        return item


class Traffic2DFlow(LightningDataModule):
    """one dimension csv"""
    def __init__(
        self,
        dataset_root,
        train_data_dir: str,
        valid_data_dir: str,
        test_data_dir: str,
        train_label_dir: str,
        valid_label_dir: str,
        test_label_dir: str,
        create_validation_set: str,
        flow_num: int = 4,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 8,
    )->None:
        super().__init__()

        self.root = dataset_root if isinstance(dataset_root,Path) else Path(dataset_root)
        self.train_data_path = self.root / train_data_dir
        self.valid_data_path = self.root / valid_data_dir
        self.test_data_path = self.root / test_data_dir
        self.train_label_path = self.root / train_label_dir
        self.valid_label_path = self.root / valid_label_dir
        self.test_label_path = self.root / test_label_dir
        self.create_validation_set = create_validation_set

        self.flow_num = flow_num
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers



    def setup(self,stage: str | None = None)->None:
        """
            Setup train,validtion and test data.
        
        Args:
            stage: Train/Val/Test  stages
        """
        if stage in (None,"fit"):
            self.train_data=Traffic2DFlowDataset(
                train_data_path=self.train_data_path,
                valid_data_path=self.valid_data_path,
                test_data_path=self.test_data_path,
                train_label_path=self.train_label_path,
                valid_label_path=self.valid_label_path,
                test_label_path=self.test_label_path,
                flow_num=self.flow_num,
                dataset_type="train",
            )
        
        if self.create_validation_set:
            self.valid_data=Traffic2DFlowDataset(
                train_data_path=self.train_data_path,
                valid_data_path=self.valid_data_path,
                test_data_path=self.test_data_path,
                train_label_path=self.train_label_path,
                valid_label_path=self.valid_label_path,
                test_label_path=self.test_label_path,
                flow_num=self.flow_num,
                dataset_type="valid",
            )

        self.test_data=Traffic2DFlowDataset(
                train_data_path=self.train_data_path,
                valid_data_path=self.valid_data_path,
                test_data_path=self.test_data_path,
                train_label_path=self.train_label_path,
                valid_label_path=self.valid_label_path,
                test_label_path=self.test_label_path,
                flow_num=self.flow_num,
                dataset_type="test",
            )

        if stage == "predict":
            
            self.inference = self.test_data
    
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        dataset = self.valid_data if self.create_validation_set else self.test_data
        return DataLoader(dataset=dataset, shuffle=False, batch_size=self.train_batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.inference, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)