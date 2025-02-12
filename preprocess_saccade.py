import os
import sys
import zipfile
import cv2
import numpy as np
import pandas as pd
import datetime
import xml.etree.ElementTree as ET
import argparse
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil


def main(input_path: str, mode: str, output_path: str, clip_length: float):
    """
    Processes either a collection of monthly ZIP folders (mode='monthly_zip') 
    or a single already-unzipped folder (mode='all_unzip'), extracting 
    'Laser Saccade' cases, writing out saccade.avi/saccade.csv, and 
    accumulating metadata in saccade_summary.csv.

    Arguments:
        input_path (str):
            - For 'monthly_zip', this is the root path containing multiple 
              monthly subfolders (like 2023_August, 2023_December, etc.).
            - For 'all_unzip', this is the folder path containing multiple 
              subfolders, each subfolder is already unzipped data of one case.
        mode (str): 'monthly_zip' or 'all_unzip'
        output_path (str): Where to store the processed outputs; the final 
                           saccade_summary.csv is placed next to this path.
    """
    # A list of dicts, each dict is one row of metadata.
    meta_rows = []

    # Ensure output directory exists:
    os.makedirs(output_path, exist_ok=True)

    # --------------------------
    # 1) Dispatch logic by mode
    # --------------------------
    if mode == 'monthly_zip':
        # We expect input_path to have multiple month-named subfolders,
        # each containing many ZIP files. We'll process each ZIP inside each folder.
        month_folders = [
            f for f in os.listdir(input_path)
            # Ignore hidden folders:
            if not f.startswith('.')
            # Only directories:
            and os.path.isdir(os.path.join(input_path, f))
        ]
        # Sort for convenience (optional)
        month_folders.sort()

        for month_folder in month_folders:
            folder_fullpath = os.path.join(input_path, month_folder)
            # Iterate over all ZIP files in that folder
            for item in tqdm(os.listdir(folder_fullpath), desc=f"Processing {month_folder}"):
                if item.startswith('.'):        # ignore hidden
                    continue
                if item.lower().endswith('.zip'):
                    zip_path = os.path.join(folder_fullpath, item)
                    row_data = process_zip_file(zip_path, output_path, clip_length)
                    # process_zip_file() returns a list of metadata rows 
                    # (in case we handle multiple "Laser Saccade" videos).
                    meta_rows.extend(row_data)

    elif mode == 'all_unzip':
        # We expect input_path to have multiple subfolders, each is one 
        # unzipped "case". We just process them as if they were unzipped zips.
        subfolders = [
            f for f in os.listdir(input_path)
            if not f.startswith('.')        # ignore hidden
            and os.path.isdir(os.path.join(input_path, f))
        ]
        subfolders.sort()

        for subf in subfolders:
            case_path = os.path.join(input_path, subf)
            row_data = process_unzipped_folder(case_path, output_path, clip_length)
            meta_rows.extend(row_data)
    else:
        raise ValueError("mode must be either 'monthly_zip' or 'all_unzip'.")

    # -----------------------------
    # 2) Create final summary DataFrame
    # -----------------------------
    if len(meta_rows) == 0:
        print("No valid Laser Saccade cases found. No summary CSV generated.")
        return

    df = pd.DataFrame(meta_rows)

    # The instructions say: 
    # "after getting the dataframe, add a column at front named 'saccade_testuid' 
    # and fill the value with the TestUID in <VW_SaccadeTest>."
    # We'll just rename or copy that column so it's definitely in front.
    if 'TestUID' in df.columns:
        df.insert(0, 'saccade_testguid', df['TestGUID'])

    # Save final CSV side by side with the output_path (i.e. in the same directory).
    if output_path == ".":
        summary_csv_path = os.path.join('saccade_summary.csv')
    else:
        summary_csv_path = os.path.join(output_path + '_saccade_summary.csv')
    df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary to {summary_csv_path}")


def process_zip_file(zip_path: str, save_path: str, clip_length: float):
    """
    Processes a single ZIP file that is expected to contain:
      - One .xml file
      - One .txt file
      - A 'video/' folder with possible .avi files
    If we find an AVI with 'Laser Saccade' in its name, parse the .xml, parse 
    the .txt block, do the downsampling, write out saccade.avi/csv, and return 
    a list of metadata-row dicts.

    Returns:
      A list of dictionaries (each dictionary is one row of metadata). 
      If no "Laser Saccade" .avi found, returns [].
    """
    rows = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # 1) Find any AVI with "Laser Saccade"
            avi_files = [
                f for f in z.namelist()
                if f.lower().endswith('.avi') and ("laser saccade" in f.lower())
            ]
            if not avi_files:
                # No "Laser Saccade" video, skip
                return rows

            # 2) Extract the single .xml file (assume exactly one, or take the first)
            xml_candidates = [
                f for f in z.namelist()
                if not os.path.basename(f).startswith('.')  # <-- ignore hidden
                and f.lower().endswith('.xml')
            ]
            if len(xml_candidates) != 1:
                # No XML, skip
                return rows
            xml_file = xml_candidates[0]
            with z.open(xml_file) as xf:
                xml_content = xf.read()
            # parse XML
            root = ET.fromstring(xml_content)

            # 3) Extract the single .txt file
            txt_candidates = [f for f in z.namelist() if f.lower().endswith('.txt')]
            if len(txt_candidates) != 1:
                # No txt, skip
                return rows
            txt_file = txt_candidates[0]
            with z.open(txt_file) as tf:
                txt_lines = tf.read().decode('utf-8', errors='replace').splitlines()

            # We may have multiple "Laser Saccade" AVIs, so handle each one in turn:
            for avi_path in avi_files:
                # -- gather meta info from XML
                meta_dict = parse_xml_for_metadata(root)

                # If parse_xml_for_metadata() returned None or incomplete, skip
                if meta_dict is None or 'TestGUID' not in meta_dict:
                    continue

                test_uid = meta_dict['TestGUID']
                # -- parse the .txt to extract the block for "Laser Saccade" test_uid
                time_stim_wave = parse_txt_for_saccade_data(txt_lines, test_uid)
                if time_stim_wave is None or len(time_stim_wave) == 0:
                    # Could not extract data from txt, skip
                    continue

                # We have the data in time_stim_wave as a list of (time, stim, wave)
                # Downsample from AvgFrameRate -> FrameRate
                avg_frame_rate = float(meta_dict.get('AvgFrameRate', 0))
                video_frame_rate = float(meta_dict.get('FrameRate', 0))

                downsampled_data = resample_data(
                    time_stim_wave,
                    old_fps=avg_frame_rate,
                    new_fps=video_frame_rate
                )

                # Next, we must match the number of frames to the actual video length
                # 4) Extract the "Laser Saccade" avi from the zip to a temp location 
                #    or directly to the final location. Then check its actual frame count.
                # Create the new folder in save_path named f"{test_uid}"
                test_output_dir = os.path.join(save_path, f"{test_uid}")
                os.makedirs(test_output_dir, exist_ok=True)

                saccade_avi_path = os.path.join(test_output_dir, "saccade.avi")
                # Extract the .avi from zip
                with open(saccade_avi_path, 'wb') as f_out:
                    f_out.write(z.read(avi_path))

                # 5) Count the frames in that extracted video
                cap = cv2.VideoCapture(saccade_avi_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open {saccade_avi_path}")
                    cap_frame_count = 0
                else:
                    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                adjusted_data = match_data_length(downsampled_data, cap_frame_count)

                # Instead of writing one big file, 
                # we now split into subclips (if clip_length > 0):
                split_into_clips(
                    adjusted_data,
                    saccade_avi_path,
                    test_uid,
                    video_frame_rate,
                    clip_length,
                    save_path
                )

                # # 6) Save the adjusted data as saccade.csv
                # csv_save_path = os.path.join(test_output_dir, "saccade.csv")
                # save_data_to_csv(adjusted_data, csv_save_path)

                # Finally, store meta info for summary
                rows.append(meta_dict)

    except Exception as e:
        print(f"Error processing ZIP {zip_path}: {e}")
    return rows


def process_unzipped_folder(folder_path: str, save_path: str, clip_length: float):
    """
    Processes an already-unzipped folder that should contain:
      - exactly one .xml file
      - one .txt file
      - a video/ folder with one or more .avi

    The same logic as process_zip_file, but we do NOT unzip. 
    Returns a list of metadata-row dicts (one for each 'Laser Saccade' found).
    """
    rows = []

    # Find the .xml
    xml_candidates = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.xml')
    ]
    if len(xml_candidates) != 1:
        return rows
    xml_path = os.path.join(folder_path, xml_candidates[0])

    # Parse XML
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return rows

    # Find the .txt
    txt_candidates = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.txt')
    ]
    if len(txt_candidates) != 1:
        return rows
    txt_path = os.path.join(folder_path, txt_candidates[0])
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as tf:
            txt_lines = tf.read().splitlines()
    except:
        return rows

    # The 'video' folder:
    video_folder = os.path.join(folder_path, 'video')
    if not os.path.isdir(video_folder):
        return rows

    # Check for any AVI with "Laser Saccade"
    avi_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith('.avi') and ("laser saccade" in f.lower())
    ]
    if not avi_files:
        # no Laser Saccade
        return rows

    # Possibly multiple Laser Saccade AVIs
    for avi_file in avi_files:
        # parse metadata from XML
        meta_dict = parse_xml_for_metadata(root)
        if meta_dict is None or 'TestGUID' not in meta_dict:
            continue
        test_uid = meta_dict['TestGUID']

        # parse txt block for "Laser Saccade" test_uid
        time_stim_wave = parse_txt_for_saccade_data(txt_lines, test_uid)
        if time_stim_wave is None or len(time_stim_wave) == 0:
            continue

        # resample
        avg_frame_rate = float(meta_dict.get('AvgFrameRate', 0))
        video_frame_rate = float(meta_dict.get('FrameRate', 0))
        downsampled_data = resample_data(
            time_stim_wave,
            old_fps=avg_frame_rate,
            new_fps=video_frame_rate
        )

        # Copy or open the avi from video_folder
        laser_avi_path = os.path.join(video_folder, avi_file)

        # count frames
        cap = cv2.VideoCapture(laser_avi_path)
        if not cap.isOpened():
            cap_frame_count = 0
        else:
            cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # match length
        adjusted_data = match_data_length(downsampled_data, cap_frame_count)

        # 6) create new folder in save_path
        test_output_dir = os.path.join(save_path, f"{test_uid}")
        os.makedirs(test_output_dir, exist_ok=True)

        # copy or re-save the .avi to saccade.avi
        saccade_avi_path = os.path.join(test_output_dir, "saccade.avi")
        # We can just copy it with open/read/write or using shutil
        
        shutil.copy2(laser_avi_path, saccade_avi_path)

        # Now split into subclips
        split_into_clips(
            adjusted_data,
            saccade_avi_path,
            test_uid,
            video_frame_rate,
            clip_length,
            save_path
        )

        # # save the csv
        # csv_save_path = os.path.join(test_output_dir, "saccade.csv")
        # save_data_to_csv(adjusted_data, csv_save_path)

        rows.append(meta_dict)

    return rows


def split_into_clips(adjusted_data,
                     avi_path,
                     base_uid,          # e.g. test_uid
                     video_fps,
                     clip_length,
                     base_output_dir):
    """
    Splits the matched CSV data + the full avi into multiple clips
    of length clip_length (in seconds). The leftover frames are ignored.
    
    Each subclip is stored in a subfolder:
        base_output_dir/base_uid_0/
        base_output_dir/base_uid_1/
        ...
    with files saccade.avi, saccade.csv, waveform.png
    """

    if clip_length <= 0:
        # The .avi is already in base_output_dir, named "saccade.avi".
        # Just write one big CSV and one big waveform.png
        os.makedirs(os.path.join(base_output_dir, base_uid), exist_ok=True)

        full_csv_path = os.path.join(base_output_dir, base_uid, "saccade.csv")
        save_data_to_csv(adjusted_data, full_csv_path)

        # Also plot waveform.png for the entire data
        waveform_png_path = os.path.join(base_output_dir, base_uid, "waveform.png")
        plot_and_save_waveform(adjusted_data, video_fps, waveform_png_path)

        # We are done; no subclips
        return

    # 1) Read all frames from the original AVI
    cap = cv2.VideoCapture(avi_path)
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(frm)
    cap.release()

    total_frames = len(frames)
    frames_per_clip = int(round(clip_length * video_fps))
    if frames_per_clip <= 0:
        return  # nothing to do

    # how many full clips can we get?
    num_clips = total_frames // frames_per_clip
    if num_clips == 0:
        # We have fewer frames than one clip => ignore
        return

    # For each clip, slice out frames and data
    for clip_i in range(num_clips):
        start_frame = clip_i * frames_per_clip
        end_frame   = (clip_i + 1) * frames_per_clip  # exclusive

        sub_frames = frames[start_frame:end_frame]
        sub_data   = adjusted_data[start_frame:end_frame]  # slice the CSV rows

        # create subfolder: e.g. base_output_dir / base_uid_0
        clip_subdir = os.path.join(base_output_dir, f"{base_uid}_{clip_i}")
        os.makedirs(clip_subdir, exist_ok=True)

        # 2) Write out the sub-AVI
        # use the same FPS + resolution
        height, width = sub_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG', 'DIVX', etc.
        out_avi = cv2.VideoWriter(
            os.path.join(clip_subdir, "saccade.avi"),
            fourcc,
            video_fps,
            (width, height)
        )
        for fr in sub_frames:
            out_avi.write(fr)
        out_avi.release()

        # 3) Write out sub CSV
        sub_csv_path = os.path.join(clip_subdir, "saccade.csv")
        save_data_to_csv(sub_data, sub_csv_path)

        # 4) Plot and save a waveform.png
        # we have sub_data as list of (time, stim, wave)
        # timesteps = e.g. 0..N-1 in frames. We'll treat them as frames, 
        # then convert to seconds = index / fps
        plot_and_save_waveform(sub_data, video_fps, os.path.join(clip_subdir, "waveform.png"))

    os.remove(avi_path)
    os.rmdir(os.path.dirname(avi_path)) # delete the entire video, only left the clips

def plot_and_save_waveform(data_tuples, fps, png_path):
    """
    data_tuples is a list of (time_step, stimulus, waveform).
    We'll interpret each row's time_step as a frame index (or “time in frames”). 
    Convert to seconds by dividing by fps. Then plot both curves.
    """
    if not data_tuples:
        return

    # Convert to arrays for easier plotting
    arr = np.array(data_tuples)  # shape [N, 3]
    # arr[:,0] = time_step, arr[:,1] = stimulus, arr[:,2] = waveform
    timesteps = np.arange(len(arr[:, 0]))  # frames
    stimulus  = arr[:, 1]
    waveform  = arr[:, 2]

    # Convert frames -> seconds
    timesteps_sec = timesteps / fps

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps_sec, stimulus, label='Stimulus', linestyle='dashed')
    plt.plot(timesteps_sec, waveform, label='Waveform')
    plt.xlabel(f"Time (seconds)")
    plt.ylabel("Values")
    plt.title("Comparison of Stimulus and Waveform")
    plt.legend()
    plt.savefig(png_path)
    plt.close()


def parse_xml_for_metadata(root):
    """
    Given the root of the XML tree (with namespace="http://tempuri.org/PMRExportDataSet.xsd"),
    extract the required fields:
      <ICSPatient>:
          PatientUID, PatientID, PatientGUID, BirthDate, ...
      <VW_SaccadeTest>:
          TestUID, PatientUID, TestGUID, WorkstationName, AvgFrameRate, IsAbnormal, ...
      <VW_SaccadeVideo>:
          TestUID, VideoUID, PatientUID, VideoGUID, WorkstationName, IsVideoLeftEye, ...
    
    Return a dict of all these fields (strings). Also compute 'age' from BirthDate.
    """

    # Your namespace from the XML's root tag:
    #   <PMRExportDataSet xmlns="http://tempuri.org/PMRExportDataSet.xsd">
    ns = {'ns': 'http://tempuri.org/PMRExportDataSet.xsd'}

    # A helper to safely get .text from a sub-element
    def get_text(parent_elem, xpath):
        """
        parent_elem: an Element
        xpath: something like 'ns:PatientUID' or 'ns:TestUID'
        returns: string or '' if not found
        """
        child = parent_elem.find(xpath, ns)
        if child is not None and child.text is not None:
            return child.text.strip()
        return ''

    # Container for our metadata
    meta = {}

    # 1) Find the <ICSPatient> element
    patient = root.find('ns:ICSPatient', ns)
    if patient is None:
        # If there's no <ICSPatient>, return empty or None
        return None

    # Extract patient info from sub-elements
    meta['PatientUID']  = get_text(patient, 'ns:PatientUID')
    meta['PatientID']   = get_text(patient, 'ns:PatientID')
    meta['PatientGUID'] = get_text(patient, 'ns:PatientGUID')
    birth_date_str      = get_text(patient, 'ns:BirthDate')
    meta['BirthDate']   = birth_date_str
    meta['Gender']      = get_text(patient, 'ns:Gender')
    meta['Country']     = get_text(patient, 'ns:Country')
    meta['GestationAge']= get_text(patient, 'ns:GestationAge')

    # Compute approximate 'age' from BirthDate
    age_years = ''
    if birth_date_str:
        try:
            # The sample BirthDate looks like "1938-09-10T00:00:00"
            # so we can split off the time
            date_only = birth_date_str.split('T')[0]  # "1938-09-10"
            bd = datetime.datetime.strptime(date_only, "%Y-%m-%d")
            today = datetime.datetime.now()
            delta = today - bd
            age_years = int(delta.days // 365)
        except:
            pass
    meta['age'] = age_years

    # 2) Within <ICSPatient>, find <VW_SaccadeTest>
    #    (You might have multiple <VW_SaccadeTest>, but assuming just one for now)
    saccade_test = patient.find('ns:VW_SaccadeTest', ns)
    if saccade_test is not None:
        meta['TestUID']                     = get_text(saccade_test, 'ns:TestUID')
        meta['TestGUID']                    = get_text(saccade_test, 'ns:TestGUID')
        meta['PatientUID_sac']             = get_text(saccade_test, 'ns:PatientUID')
        meta['WorkstationName']            = get_text(saccade_test, 'ns:WorkstationName')
        meta['AvgFrameRate']               = get_text(saccade_test, 'ns:AvgFrameRate')
        meta['IsAbnormal']                 = get_text(saccade_test, 'ns:IsAbnormal')
        meta['AvgPeakVelocityHRRightward'] = get_text(saccade_test, 'ns:AvgPeakVelocityHRRightward')
        meta['AvgPeakVelocityHRLeftward']  = get_text(saccade_test, 'ns:AvgPeakVelocityHRLeftward')
        meta['AvgAccuracyHRRightward']     = get_text(saccade_test, 'ns:AvgAccuracyHRRightward')
        meta['AvgAccuracyHRLeftward']      = get_text(saccade_test, 'ns:AvgAccuracyHRLeftward')
        meta['AvgLatencyHRRightward']      = get_text(saccade_test, 'ns:AvgLatencyHRRightward')
        meta['AvgLatencyHRLeftward']       = get_text(saccade_test, 'ns:AvgLatencyHRLeftward')
        meta['VideoUID']                   = get_text(saccade_test, 'ns:VideoUID')
        meta['VideoGUID']                  = get_text(saccade_test, 'ns:VideoGUID')

        # 3) <VW_SaccadeVideo> is inside <VW_SaccadeTest>
        saccade_video = saccade_test.find('ns:VW_SaccadeVideo', ns)
        if saccade_video is not None:
            meta['TestUID_video']         = get_text(saccade_video, 'ns:TestUID')
            meta['VideoUID_video']        = get_text(saccade_video, 'ns:VideoUID')
            meta['PatientUID_video']      = get_text(saccade_video, 'ns:PatientUID')
            meta['VideoGUID_video']       = get_text(saccade_video, 'ns:VideoGUID')
            meta['WorkstationName_video'] = get_text(saccade_video, 'ns:WorkstationName')
            meta['IsVideoLeftEye']        = get_text(saccade_video, 'ns:IsVideoLeftEye')
            meta['ModuleName']            = get_text(saccade_video, 'ns:ModuleName')
            meta['Quality']               = get_text(saccade_video, 'ns:Quality')
            meta['Width']                 = get_text(saccade_video, 'ns:Width')
            meta['Height']                = get_text(saccade_video, 'ns:Height')
            meta['FrameRate']             = get_text(saccade_video, 'ns:FrameRate')
            meta['IsVideoAbnormal']       = get_text(saccade_video, 'ns:IsVideoAbnormal')
            meta['IsVideoCompressionOn']  = get_text(saccade_video, 'ns:IsVideoCompressionOn')

    return meta


def parse_txt_for_saccade_data(txt_lines, test_uid):
    """
    From the .txt file lines, we look for the row that begins with <TestUID> 
    AND has 'Laser Saccade' in the row. Then we collect lines until the next 
    <TestUID> line, interpreting them as CSV data for columns 1..3 
    (time_step, stimulus, waveform).
    We specifically look for the block that belongs to the same TestUID as in 
    the XML. If you have multiple test_uids, you might refine logic accordingly.
    """

    # We do a two-pass approach:
    # 1) find a line that starts with <TestUID> and has the test_uid plus 'Laser Saccade'.
    # 2) from there, gather lines until the next <TestUID> or end-of-file.

    start_index = None
    for i, line in enumerate(txt_lines):
        if line.strip().startswith("<TestUID>"):
            # Check if the test_uid and "Laser Saccade" are in the same line
            if (test_uid in line) and ("Laser Saccade" in line):
                start_index = i + 1
                break

    if start_index is None:
        return None  # No block found

    # Collect lines from start_index until next <TestUID> or end
    block_lines = []
    for j in range(start_index, len(txt_lines)):
        line = txt_lines[j]
        if line.strip().startswith("<TestUID>"):
            # next testUID encountered, stop
            break
        block_lines.append(line)

    # Now parse block_lines as CSV, focusing on 3 columns: time_step, stimulus, waveform
    # We'll assume columns are separated by commas or whitespace.
    data = []
    for bl in block_lines:
        # safe split
        cols = bl.split(',')
        if len(cols) < 3:
            # try splitting by whitespace if comma doesn't exist
            if '\t' in bl:
                cols = bl.split('\t')
            else:
                cols = bl.split()
        if len(cols) >= 3:
            time_step = cols[0].strip()
            stimulus  = cols[1].strip()
            waveform  = cols[2].strip()
            # Convert numeric fields if possible
            try:
                time_step_val = float(time_step)
            except:
                time_step_val = 0.0
            try:
                stim_val = float(stimulus)
            except:
                stim_val = 0.0
            try:
                wave_val = float(waveform)
            except:
                wave_val = 0.0

            data.append((time_step_val, stim_val, wave_val))

    return data


def resample_data(data, old_fps, new_fps):
    """
    Resample 'data' from old_fps to new_fps. 
    'data' is a list of tuples: (time, stim, wave).
    We'll do a simple nearest-index approach.
    
    If old_fps <= 0 or new_fps <= 0, we skip and just return the original data.
    """
    if old_fps <= 0 or new_fps <= 0 or len(data) == 0:
        return data

    old_length = len(data)
    ratio = old_fps / new_fps  # how many old frames per 1 new frame
    if ratio <= 0:
        return data

    new_length = int(round(old_length / ratio))
    if new_length < 1:
        # fallback
        return data

    # Build new series by nearest neighbor
    resampled = []
    for i in range(new_length):
        old_index = int(round(i * ratio))
        if old_index >= old_length:
            old_index = old_length - 1
        resampled.append(data[old_index])
    return resampled


def match_data_length(data, num_frames):
    """
    If data is shorter than num_frames, pad by copying the last value.
    If data is longer, cut it. 
    data is a list of tuples (time, stim, wave).
    """
    if num_frames <= 0:
        # no valid frames, just return the data as is
        return data

    curr_len = len(data)
    if curr_len == num_frames:
        return data
    elif curr_len < num_frames:
        # pad
        if curr_len == 0:
            # can't pad from last value if empty
            return data
        last_val = data[-1]
        needed = num_frames - curr_len
        data.extend([last_val]*needed)
        return data
    else:
        # curr_len > num_frames, cut off
        return data[:num_frames]


def save_data_to_csv(data, csv_path):
    """
    Save the list of tuples (time, stim, wave) as a CSV with columns 
    time_step, stimulus, waveform.
    """
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("time_step,stimulus,waveform\n")
        for row in data:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")


# If you want to run this as a script directly:
if __name__ == "__main__":
    """
    Example usage:
       python saccade_script.py /path/to/input monthly_zip /path/to/output
    or
       python saccade_script.py /path/to/unzipped all_unzip /path/to/output
    """
    parser = argparse.ArgumentParser(description="Script that takes three string arguments.")
    parser.add_argument("-i", "--input_path", type=str, help="input path")
    parser.add_argument("-m", "--mode", type=str, choices=["monthly_zip", "all_unzip"])
    parser.add_argument("-o", "--output_path", type=str, help="output_path")
    parser.add_argument("--clip_length", type=float, default=0.0,
                    help="Length (in seconds) of each sub-clip to be created. If <=0, no clipping.")
    
    args = parser.parse_args()

    main(args.input_path, args.mode, args.output_path, args.clip_length)