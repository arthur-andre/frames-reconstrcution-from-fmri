import os
import numpy as np
import nibabel as nib


subject_list = [1,2]
#but 1 digit subject should be 0X instead only X
for i in range(len(subject_list)):
    if subject_list[i] < 10:
        subject_list[i] = '0' + str(subject_list[i])
    else:
        subject_list[i] = str(subject_list[i])


mask_img = nib.load('/mnt/c/Users/strom/Desktop/fmri_project/mask/smask_occi.nii')
mask_data = mask_img.get_fdata()
        


cumulative_label = 0
term = '.npy'

for subject in subject_list:
    print('subject: ', subject)
    for run in range(8):
        # Get the total volumes
        real_run_name = run + 1
        num_volumes = int(os.popen(f"fslval subjects/sub-{subject}/new_space/{real_run_name}/filtered_func_data.nii.gz dim4").read())

        for i in range(num_volumes):
            # Extract the volume
            os.system(f"fslroi subjects/sub-{subject}/new_space/{real_run_name}/filtered_func_data.nii.gz subjects/sub-{subject}/new_space/{real_run_name}/volume_{i}.nii.gz {i} 1")
            # Register the volume to the standard space
            os.system("flirt -in subjects/sub-{1}/new_space/{2}/volume_{0}.nii.gz -ref /home/arthur_andre/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz -init subjects/sub-{1}/new_space/{2}/reg/example_func2standard.mat -out subjects/sub-{1}/new_space/{2}/registered_volume_{0}.nii.gz".format(i,subject,real_run_name))
            
            # Multiply the registered volume with the mask in standard space
            #os.system(f"fslmaths subjects/sub-{subject}/new_space/{real_run_name}/registered_volume_{i}.nii.gz -mul /mnt/c/Users/strom/Desktop/fmri_project/mask/mask_occi.nii subjects/sub-{subject}/new_space/{real_run_name}/result_{i}.nii.gz")
            
            fmri = nib.load(f'subjects/sub-{subject}/new_space/{real_run_name}/registered_volume_{i}.nii.gz')
            fmri_before_mask = fmri.get_fdata()
            masked_data = np.multiply(fmri_before_mask, mask_data)
            flat_masked_data = masked_data[np.nonzero(mask_data)]
            # Save the masked volume
            np.save(f'/mnt/c/Users/strom/Desktop/fmri_project/subjects/sub-{subject}/flat_fmri/label'+str((i+1+cumulative_label)) + term, flat_masked_data)

            # Clean up intermediate files
            os.system(f"rm subjects/sub-{subject}/new_space/{real_run_name}/volume_{i}.nii.gz subjects/sub-{subject}/new_space/{real_run_name}/registered_volume_{i}.nii.gz")


        cumulative_label += num_volumes

    