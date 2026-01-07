clear; close all;
dataset_path = "./data";

remove_empty_channel = true;

parfor subject_index = 1:1 %多个病人 改这个

    file_path = dataset_path+"/chb"+sprintf("%02d",subject_index)+"/";
    file_prefix = "chb"+sprintf("%02d",subject_index)+"_";
    
    for i = 1:50
        file_name = file_prefix+sprintf("%02d",i)+".edf"
        % file_path
        %%{
        if isfile(file_path+file_name)
            labelEEG(file_path,file_name,remove_empty_channel);
        end
        %}
    end
end

