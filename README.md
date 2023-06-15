# frames-reconstrcution-from-fmri
Project for MIP:lab on frames reconstruction from fMRI data


# Obtain data

- Frames are coming from the movie forrest gump Blu-ray. You can use the mk_movie_stimulus file to create the runs
- Hence, you can use the notebook frames extraction to create the different single frames: 32 frames per sample (3599 samples among the 8 movie runs)
- The last data extraction is the fMRI data: need to download the functional and the anatomical data of each subject at the following link [https://openneuro.org/datasets/ds000113/versions/1.3.0]
- preprocess the functional data using the anatomical to bring the fMRI in the template space (MNI) to further apply a global mask to each volume.
- the python file export_flat_fmri can be used to create the labels once the preprocessing is done (filtered_func_data.nii.gz)

# Data training

- Either encoder or decoder training is implemented in 2 forms: notebook or direct Python file.

# Pretrained model 

You can download pretrained model directly here [[google drive](https://drive.google.com/drive/folders/1K9bnS2LjOqa7erS0BNR5y0Lc8QKybhv8?usp=sharing)]

# Acknowledgments

To the whole MIP:lab that received me during this spring semester, was really interesting to be involved in the lab. And particular mention to Ekansh Sareen who did his best to help me and guide me through this project.

