# Lestack

A command line stacking script for solar and planetary imaging.

Version  0.2.0 - 20240613  
Author: Ben de Mulder.

Lestack supports various stacking methods and can handle multiple video files in a directory.


## Requirements

The script requires the following Python modules:
- `opencv-python`
- `numpy`
- `scikit-image`
- `scipy`
- `tqdm`

## Installation
You can install these modules using `pip`:

```sh
pip install opencv-python numpy scikit-image scipy tqdm
```
## Usage

### Command-Line Arguments:
***-in / --input***: Input video file (supports .ser or .avi format).  

***-dir***: Directory containing video files to process.

***-rdir***: Directory containing video files to process recursively.

***-out / --output***: Output stacked image filename (default: result.png).

***-log***: Log file to record processing details.

***-pstack***: Percentage of frames to stack (default: 10).

***-resample***: Resampling factor (integer between 2 and 5).

***-drizzlef***: Drizzle factor.

***-drizzlep***: Drizzle pixel size relative to the original pixel size (between 0.1 and 0.9).

***-q / --quiet***: Quiet mode, suppresses all output except the final result code.

***-ncore:*** Number of cores to use (default: all cores).

***-matrix***: Bayer matrix pattern (default: RGGB). Options: RGGB, GRBG, GBRG, BGGR.

***-flat***: Flat frame calibration image file.

***-dark***: Dark frame calibration image file.

***-method***: Stacking method to use (default: mean). Options: mean, median, trimmed_mean, robust_mean.

***-method***: Weight for blur value in the frame selection process.

***-method***: Weight for contrast value in the frame selection process..

***-method***: Weight for SSIM value in the frame selection process..

***-method***: Weight for gradient magnitude in the frame selection process..

***-entropyw***: Weight for entropy value in the frame selection process..

## Examples
Process a single video file:

```sh
python lestack-0.1.5.py -in input_video.avi -out output_image.png -pstack 20 -resample 3 -method median
```
Process all video files in a directory
```sh
python lestack-0.1.5.py -dir /path/to/directory -out output_image -pstack 20 -resample 3 -method median
```
Process all video files in a directory and its subdirectories
```sh
python lestack-0.1.5.py -rdir /path/to/directory -out output_image -pstack 20 -resample 3 -method median
```
Create a log file
```sh
python lestack-0.1.5.py -in input_video.avi -out output_image.png -pstack 20 -resample 3 -method median -log process_log.txt
```
## Output
The script generates a stacked image based on the processed video frames and saves it to the specified output path. If multiple files are processed in a directory, the output files will have numeric suffixes (1, 2, 3, ...). The script will output the image based on the file suffix. It supports now PNG and TIF. 

## Log File
If a log file is specified, it will record the following details for each processed video file:

Complete path and name of the video file
Complete path and name of the generated image
Current date and time (in yyyymmddhh:mm format)
Total execution time to generate the image
Total number of frames in the video
Number of frames stacked
Notes
Ensure the input video files are in a supported format (.ser or .avi).
The script uses SIFT for feature detection and alignment, which requires OpenCV.
The stacking method can be chosen based on your preference and the nature of the video frames.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)