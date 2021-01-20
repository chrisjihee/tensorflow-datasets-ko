pip install tensorflow-datasets-ko==0.2.2
echo 'tensorflow-datasets-ko==0.2.2' > /tmp/beam_requirements.txt

#single worker
python -m tensorflow_datasets.scripts.download_and_prepare --imports tensorflow_datasets_ko.text.c4ko --datasets=c4ko/default --data_dir=gs://$MY_BUCKET/tensorflow_datasets --beam_pipeline_options="project=$MY_PROJECT,job_name=c4,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION"

#500 workers
python -m tensorflow_datasets.scripts.download_and_prepare --imports tensorflow_datasets_ko.text.c4ko --datasets=c4ko/default --data_dir=gs://$MY_BUCKET/tensorflow_datasets --beam_pipeline_options="project=$MY_PROJECT,job_name=c4,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION,max_num_workers=500"

#,autoscaling_algorithm=NONE,num_workers=500
