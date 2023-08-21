# -*- coding: utf-8 -*-
"""track datasets with W&B. https://docs.wandb.ai/tutorials/artifacts"""
import torch 
import wandb

from typing import Dict, Any, Optional

import os
import sys
from LANAM.data.gamdataset import GAMDataset


def load_and_log(datasets, 
                 project_name, 
                 job_type, 
                 artifact_name, 
                 description: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 incremental: bool = (False),
                 use_as: Optional[str] = None):
    """Log a dataset to W&B."""
    # start a run with specified project and a descriptive job tag 
    with wandb.init(project=project_name, job_type=job_type) as run:
        
        names = ["training", "validation", "test"]

        # create Artifact, with user-defined description and meta-data
        if metadata is None:
            metadata = dict()
        metadata['sizes']={names[index]: len(dataset) for index, dataset in enumerate(datasets)}
        raw_data = wandb.Artifact(name=artifact_name, 
                                  type="dataset", 
                                  metadata=metadata, 
                                  incremental=incremental, 
                                  use_as=use_as)

        for name, data in zip(names, datasets):
            # Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                torch.save(data.tensors, file)

        # Save the artifact to W&B.
        run.log_artifact(raw_data)


def preprocess(tensors, use_test, batch_size) -> torch.utils.data.Dataset: 
    """Prepare the data"""
    X, y, fnn = tensors
    return GAMDataset(X, y, fnn, batch_size=batch_size, use_test=use_test)


def read(data_dir, split): 
    """read tensors from file `split.pt` in the directory `data.dir`."""
    filename = split + '.pt'
    X, y, fnn = torch.load(os.path.join(data_dir, filename))
    return X, y, fnn

def preprocess_and_log(project_name, 
                       job_type, 
                       artifact_or_name,
                       batch_size=64,
                       ):
    """fetch and preprocess data of job type `job_type` from W&B project `project_name`.
    Returns: 
    --------
    processed_datasets, GAMDataset: customized dataset for generalized additive models.
    """
    with wandb.init(project=project_name, job_type=job_type) as run:
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact(artifact_or_name)
        # if need be, download the artifact
        raw_dataset = raw_data_artifact.download()
        
        processed_datasets = dict()
        for split in ['training', 'validation', 'test']: 
            raw_split = read(raw_dataset, split)
            use_test = True if split == 'test' else False
            processed_dataset = preprocess(raw_split, use_test, batch_size)
            processed_datasets[split] = processed_dataset

    return processed_datasets

