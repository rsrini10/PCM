# Precision Care Medicine 2024 Fall and 2025 Spring - Penguin

## Clinical Data
### Preprocess
To preprocess, modify the following command as needed. 
The `mode` has two options: `monthly_zip` and `all_unzip`. Both has been tested. 

The `clip_length` parameter is in seconds. If `clip_length <= 0`, will save the entire video/waveform.
```bash
python preprocess_saccade.py \
    --input_path /Volumes/LTY-Photos/DatasetCollection/PCMdata \
    --mode monthly_zip \
    --output_path 2023_7_12_clip5 \
    --clip_length 5
```

Also, an `{$output_path}_saccade_summary.csv` file will be saved.
