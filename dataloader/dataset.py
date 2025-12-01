#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import dataloader.utils as transforms
import torch.utils.data

import pickle as pkl
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class DataSet(torch.utils.data.Dataset):
    """Common dataset for Oxford/Paris with labels."""

    def __init__(self, data_path, dataset, fn, split, scale_list):
        assert os.path.exists(data_path), f"Data path '{data_path}' not found"
        self._data_path = data_path
        self._dataset = dataset
        self._fn = fn
        self._split = split
        self._scale_list = scale_list

        # Build the database
        self._construct_db()

        # Prepare labels
        self._labels = self._prepare_labels()

    def _construct_db(self):
        """Constructs the db."""
        self._db = []
        with open(os.path.join(self._data_path, self._dataset, self._fn), 'rb') as f:
            gnd = pkl.load(f)

        if self._split == "query":
            for i, im_fn in enumerate(gnd["qimlist"]):
                im_path = os.path.join(self._data_path, self._dataset, "jpg", im_fn + ".jpg")
                self._db.append({"im_path": im_path, "bbox": gnd["gnd"][i]["bbx"]})
        elif self._split == "db":
            for im_fn in gnd["imlist"]:
                im_path = os.path.join(self._data_path, self._dataset, "jpg", im_fn + ".jpg")
                self._db.append({"im_path": im_path})
        else:
            raise ValueError(f"Unsupported split {self._split}")

        self._gnd = gnd  # keep for label preparation

    def _prepare_labels(self):
        """Returns a list of integer labels for each image in the dataset."""
        labels = []
        if self._split == "db":
            class_id = 0
            for img_idx in range(len(self._db)):
                found = False
                for qidx, g in enumerate(self._gnd["gnd"]):
                    if img_idx in g.get("ok", []) + g.get("junk", []):
                        labels.append(qidx)  # class = query index
                        found = True
                        break
                if not found:
                    labels.append(class_id)  # fallback class
                    class_id += 1
        elif self._split == "query":
            labels = list(range(len(self._db)))
        else:
            raise ValueError("Invalid split")
        return labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1]).astype(np.float32)
        im /= 255.0
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        im_path = self._db[index]["im_path"]
        try:
            im = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"Image not found: {im_path}")

            # Crop for query
            if self._split == "query":
                bbx = self._db[index]["bbox"]
                y1, y2 = max(0,int(bbx[1])), min(im.shape[0], int(bbx[3]))
                x1, x2 = max(0,int(bbx[0])), min(im.shape[1], int(bbx[2]))
                im = im[y1:y2, x1:x2]

            # Resize to network input
            im = cv2.resize(im, (224, 224))

        except Exception as e:
            print(f"Error loading {im_path}: {e}")
            im = np.zeros((224, 224, 3), dtype=np.uint8)

        im_tensor = torch.from_numpy(self._prepare_im(im))  # shape [3,H,W]
        label_tensor = torch.tensor(self._labels[index], dtype=torch.long)

        return im_tensor, label_tensor

    def __len__(self):
        return len(self._db)
