class WandbConfiguration:
    def __init__(self, 
                experiment_project_name: str='LANAM-test', 
                data_project_name: str='Datasets', 
                log_artifact_name: str='synthetic-4',
                load_artifact_name: str='synthetic-4:v1', 
                table_name: str='synthetic-4'): 
        
        """wandb configuration for dataset and experimental results. 
        Attrs: 
        -----
        experiment_project_name: 
            name of wandb project for this experiment. 
        data_project_name: 
            name of wandb project for dataset.
        log_artifact_name:
            name for artifact to be logged to wandb.
        load_artifact_name:
            name of artifact to be accessed. 
        table_name: 
            name of data table. 
        """
        if table_name is None: 
            raise ValueError('`table_name` is required.')
            
        self.experiment_project_name = experiment_project_name # name of wandb project for this experiment
        self.data_project_name = data_project_name # name of wandb project for data
        self.log_artifact_name = log_artifact_name # artifact name of newly built dataset
        self.load_artifact_name = load_artifact_name # artifact name of the dataset stored in project `data_project_name`
        self.table_name = table_name # unique table name of the loaded/logged data artifact 
