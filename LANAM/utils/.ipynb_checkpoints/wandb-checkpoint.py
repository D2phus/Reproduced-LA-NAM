"""track datasets with W&B. https://docs.wandb.ai/tutorials/artifacts"""
import torch 
import pandas as pd 

import wandb

from typing import Dict, Any, Optional, Tuple

import os
import sys
import json
from LANAM.config.default import defaults
from LANAM.data.base import LANAMDataset, LANAMSyntheticDataset

cfg = defaults()

def log_LANAMSyntheticDataset(dataset: LANAMSyntheticDataset, 
                              project_name: str, 
                              artifact_name: str, 
                              table_name: str):  
    """Log LANAMSyntheticDataset to W&B table. """
    in_features = dataset.in_features
    sigma = dataset.sigma
    feature_targets = dataset.feature_targets
    
    data = dataset.raw_data
    for idx in range(in_features):
        data[f'feature_target{idx+1}'] = pd.Series(feature_targets[:, idx])
    
    metadata = {
        'sigma': sigma, 
        'in_features': in_features, 
    }
    log_dataframe_to_table(data, project_name, artifact_name, table_name, metadata)

def log_LANAMDataset(dataset: LANAMDataset,
                     project_name: str, 
                    artifact_name: str, 
                    table_name: str):  
    """Log LANAMDataset to W&B table. """
    metadata = {
        'in_features': dataset.in_features
    }
    
    log_dataframe_to_table(dataset.raw_data, project_name, artifact_name, table_name, metadata)
    
def log_dataframe_to_table(data: pd.DataFrame, project_name, artifact_name, table_name, metadata: dict=None): 
    """log pandas.DataFrame data to W&B as a table."""
    table = wandb.Table(dataframe=data)
    with wandb.init(project=project_name) as run:
        if metadata is None: 
            metadata = dict()
        artifact = wandb.Artifact(artifact_name, type='dataset', metadata=metadata)
        artifact.add(table, table_name)
        run.log({table_name: table})
        run.log_artifact(artifact)
        
def load_LANAMDataset(project_name, artifact_or_name, table_name, config=cfg):
    """fetch LANAMDataset from W&B"""
    _, data = load_table_to_dataframe(project_name, artifact_or_name, table_name)
    return LANAMDataset(config,
                        data_path=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1])

def load_LANAMSyntheticDataset(project_name, artifact_or_name, table_name, config=cfg):
    """fetch LANAMSyntheticDataset from W&B"""
    dataset, metadata = load_table_to_dataframe(project_name, artifact_or_name, table_name)
    sigma = metadata['sigma']
    in_features = metadata['in_features']
    data = dataset.loc[:, :'target']
    #print(data)
    feature_targets = torch.tensor(dataset.loc[:, 'feature_target1':].values, dtype=torch.float32) # automatically convert to double; rememmber to specify the data type
    #print(feature_targets.shape)
    return LANAMSyntheticDataset(config,
                        data_path=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1], 
                        feature_targets=feature_targets, 
                        sigma=sigma)
    
    
def load_table_to_dataframe(project_name, artifact_or_name, table_name):
    """load W&B table as pandas.DataFrame data"""
    with wandb.init(project=project_name) as run:
        # ✔️ declare which artifact we'll be using
        artifact = run.use_artifact(artifact_or_name)
        metadata = artifact.metadata
        #print(metadata)
        artifact_dir = artifact.download()
        table_path = f'{artifact_dir}/{table_name}.table.json'
        
        with open(table_path) as file:
            json_dict = json.load(file)
        
        data = pd.DataFrame(json_dict['data'], columns=json_dict['columns'])
    
    return data, metadata

        
def load_and_log(datasets, 
                 project_name, 
                 job_type, 
                 artifact_name, 
                 description: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,):
    """log dataset files of training, validation, and test subsets to W&B."""
    # start a run with specified project and a descriptive job tag 
    with wandb.init(project=project_name, job_type=job_type) as run:
        names = ["training", "validation", "test"]

        # create Artifact, with user-defined description and meta-data
        if metadata is None:
            metadata = dict()
        metadata['sizes']={names[index]: len(dataset) for index, dataset in enumerate(datasets)}
        raw_data = wandb.Artifact(name=artifact_name, 
                                  type="dataset", 
                                  metadata=metadata)

        for name, data in zip(names, datasets):
            # Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                torch.save(data.tensors, file)

        # Save the artifact to W&B.
        run.log_artifact(raw_data)


def preprocess(tensors: Tuple, use_test, batch_size) -> torch.utils.data.Dataset: 
    """convert data to dataset.
    """
    pass

def read(data_dir: str, split: str)->Tuple: 
    """read tensors from file `split.pt` in the directory `data.dir`.
    Args:
    -----
    data_dir: str
        The local folder for downloaded files.
    split: str
        filenames.
    """
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
    processed_datasets: 
        customized dataset.
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

