function [data, write_info] = labelEEG(file_path,file_name,remove_empty_channel)
    if ~isfile(file_path+file_name)
        error(file_path+file_name+" not exist!"+newline ...
              +"Current Folder is " + pwd);
    end

    data = edfread(file_path+file_name);
    info = edfinfo(file_path+file_name);
    write_info = get(info);

    signal_labels = info.SignalLabels;
    fs = info.NumSamples/seconds(info.DataRecordDuration); %采样点/记录时长
    
    EEG = cell2mat(data{:,:})';

    if(remove_empty_channel)
        empty_channel = find(signal_labels=="-");
        EEG(empty_channel,:) = [];

        write_info.NumSignals = size(EEG,1);
        write_info.SignalLabels(empty_channel) = [];
        write_info.TransducerTypes(empty_channel) = [];
        write_info.PhysicalDimensions(empty_channel) = [];
        write_info.PhysicalMin(empty_channel) = [];
        write_info.PhysicalMax(empty_channel) = [];
        write_info.DigitalMin(empty_channel) = [];
        write_info.DigitalMax(empty_channel) = [];
        write_info.Prefilter(empty_channel) = [];
        write_info.NumSamples(empty_channel) = [];
        write_info.SignalReserved(empty_channel) = [];
    end
    
    hdr = edfheader("EDF");
    hdr.DataRecordDuration = write_info.DataRecordDuration;
    hdr.DigitalMax = write_info.DigitalMax;
    hdr.DigitalMin = write_info.DigitalMin;
    hdr.NumDataRecords = write_info.NumDataRecords;
    hdr.NumSignals = write_info.NumSignals;
    hdr.Patient = write_info.Patient;
    hdr.PhysicalDimensions = write_info.PhysicalDimensions;
    hdr.PhysicalMax = write_info.PhysicalMax;
    hdr.PhysicalMin = write_info.PhysicalMin;
    hdr.Prefilter = write_info.Prefilter;
    hdr.Recording = write_info.Recording;
    hdr.Reserved = write_info.Reserved;
    hdr.SignalLabels = write_info.SignalLabels;
    hdr.SignalReserved = write_info.SignalReserved;
    hdr.StartDate = write_info.StartDate;
    hdr.StartTime = write_info.StartTime;
    hdr.TransducerTypes = write_info.TransducerTypes;
    
    % 29
    if any(isnan(EEG(:)))
        warning('检测到 NaN 数据，已自动置零。');
        EEG(isnan(EEG)) = 0; 
    end
    num_channels = numel(hdr.SignalLabels);
    for ch = 1:num_channels
        p_max = double(hdr.PhysicalMax(ch));
        p_min = double(hdr.PhysicalMin(ch));

        EEG(ch, EEG(ch,:) > p_max) = p_max;
        EEG(ch, EEG(ch,:) < p_min) = p_min;
    end

    % 输出
    if ~exist(file_path+"onset_labeled\", 'dir')
        mkdir(file_path+"onset_labeled\")
    end
    if isfile(file_path+file_name+".seizures")
        [beginning_of_seizure, end_of_seizure] = get_seizure_period( file_path+file_name+".seizures" ); 
        Onset = seconds(beginning_of_seizure');
        Annotations = repelem("seizure",size(beginning_of_seizure,2))';
        Duration = seconds(end_of_seizure - beginning_of_seizure)';

        tsal = timetable(Onset,Annotations,Duration);
        edfwrite(file_path+"onset_labeled\"+"onset_labeled_"+file_name,hdr,EEG',tsal);
    else
        edfwrite(file_path+"onset_labeled\"+"onset_labeled_"+file_name,hdr,EEG');
    end
end