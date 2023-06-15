# frames-reconstrcution-from-fmri
Project for MIP:lab on frames reconstruction from fmri data


# Obtain data

- Frames are coming from the movie forrest gump blurray. You can use the mk_movie_stimulus file to create the runs
- Hence, you can use the notebook frames extraction to create the different single frames : 32 frames per sample (3599 samples among the 8 movie runs)
- The last data extraction is the fmri data : need to download the functional and the anatomical data of each subject at the following link [https://openneuro.org/datasets/ds000113/versions/1.3.0]
- preprocess the functional data using the anatomical to bring the fmri in the template space (MNI) to further apply a global mask to each volume.
- the python file export_flat_fmri can be use to create the labels once the preprocessing is done (filtered_func_data.nii.gz)

# Data training

- Either encoder or decoder training are implemented in 2 forms : notebook or directly python file.

# Pretrained model 

You can download pretrained model directly here [google drive]

# Acknowledgments

To the whole MIP:lab that received me during this spring semester, was really interesting to be involve in this type of lab.

