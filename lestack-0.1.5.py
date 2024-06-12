#################################################
#  lestack.py: A command line stacking script
#  for solar and planetary imaging
#  Version  0.1.5 - 20240612
#  By Ben de Mulder
#  This script is distributed under MIT license 
#################################################


import cv2
import numpy as np
from skimage import exposure
from scipy.fftpack import fftshift, fft2
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from tqdm import tqdm
import sys
import multiprocessing
from datetime import datetime
import time

def compute_blur_value(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = fft2(gray_image)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(rows, cols) // 10
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    low_freq_energy = np.sum(magnitude_spectrum * mask)
    high_freq_energy = np.sum(magnitude_spectrum * (1 - mask))
    blur_value = high_freq_energy / (low_freq_energy + 1e-6)
    return blur_value

def compute_contrast_value(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.std()

def align_images(image, reference):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create()
    keypoints_image, descriptors_image = detector.detectAndCompute(gray_image, None)
    keypoints_reference, descriptors_reference = detector.detectAndCompute(gray_reference, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors_image, descriptors_reference)

    matches = sorted(matches, key=lambda x: x.distance)
    top_n_matches = 50
    if len(matches) > top_n_matches:
        matches = matches[:top_n_matches]

    points_image = np.zeros((len(matches), 2), dtype=np.float32)
    points_reference = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_image[i, :] = keypoints_image[match.queryIdx].pt
        points_reference[i, :] = keypoints_reference[match.trainIdx].pt

    warp_matrix, mask = cv2.estimateAffinePartial2D(points_reference, points_image)

    if warp_matrix is not None:
        aligned_image = cv2.warpAffine(image, warp_matrix, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_image
    else:
        print("Affine transformation could not be estimated.")
        return image

def stack_images(images, method='mean'):
    if method == 'mean':
        stacked_image = np.mean(images, axis=0).astype(np.uint8)
    elif method == 'median':
        stacked_image = np.median(images, axis=0).astype(np.uint8)
    elif method == 'trimmed_mean':
        alpha = 0.1
        trimmed_means = [np.mean(np.sort(image.flatten())[int(alpha * len(image.flatten())):-int(alpha * len(image.flatten()))]) for image in images]
        stacked_image = np.mean(trimmed_means, axis=0).reshape(images[0].shape).astype(np.uint8)
    elif method == 'robust_mean':
        sigma = 10.0
        robust_images = np.array([cv2.fastNlMeansDenoisingColored(image, None, sigma, 10, 7, 21) for image in images])
        stacked_image = np.mean(robust_images, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported stacking method: {method}")

    return stacked_image

def resample_image(image, factor):
    height, width = image.shape[:2]
    new_height, new_width = height * factor, width * factor
    resampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resampled_image

def drizzle(images, drizzle_factor, drizzle_pixel_size):
    output_shape = (int(images[0].shape[0] * drizzle_factor), int(images[0].shape[1] * drizzle_factor))
    drizzle_image = np.zeros(output_shape, dtype=np.float32)
    weights = np.zeros(output_shape, dtype=np.float32)

    for image in images:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                y_new = int(y * drizzle_factor)
                x_new = int(x * drizzle_factor)
                pixel_size = int(1 / drizzle_pixel_size)
                drizzle_image[y_new:y_new + pixel_size, x_new:x_new + pixel_size] += image[y, x]
                weights[y_new:y_new + pixel_size, x_new:x_new + pixel_size] += 1

    drizzle_image /= np.maximum(weights, 1)
    return drizzle_image.astype(np.uint8)

def process_video(input_path, output_path, pstack, resample_factor, drizzle_factor, drizzle_pixel_size, quiet, ncore, matrix, flat_path, dark_path, stack_method):
    if not os.path.isfile(input_path):
        if not quiet:
            print("No valid video file given as input")
        return 0, 0, 0
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        if not quiet:
            print("No valid video file given as input")
        return 0, 0, 0

    frames = []
    frame_scores = []

    ret, frame = cap.read()
    if not ret:
        if not quiet:
            print("Failed to read the first frame")
        return 0, 0, 0
    
    is_color = len(frame.shape) == 3 and frame.shape[2] == 3
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def debayer(frame, matrix):
        if matrix == 'RGGB':
            return cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
        elif matrix == 'GRBG':
            return cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
        elif matrix == 'GBRG':
            return cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
        elif matrix == 'BGGR':
            return cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
        else:
            raise ValueError(f"Unsupported Bayer matrix pattern: {matrix}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if is_color:
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = debayer(frame, matrix)
        frames.append(frame)

    cap.release()

    def compute_scores(frame):
        blur_value = compute_blur_value(frame)
        contrast_value = compute_contrast_value(frame)
        return blur_value, contrast_value

    max_cores = multiprocessing.cpu_count()
    if ncore is not None:
        if ncore > max_cores:
            if not quiet:
                print(f"Max number of core available: {max_cores}")
            return 0, 0, 0
    else:
        ncore = max_cores

    with ThreadPoolExecutor(max_workers=ncore) as executor:
        future_to_frame = {executor.submit(compute_scores, frame): frame for frame in frames}
        for future in tqdm(as_completed(future_to_frame), total=len(frames), desc="Computing scores", disable=quiet):
            frame = future_to_frame[future]
            blur_value, contrast_value = future.result()
            frame_scores.append((blur_value, contrast_value))

    frame_scores = np.array(frame_scores)
    scores = frame_scores[:, 1] / frame_scores[:, 0]
    best_frame_index = np.argmax(scores)
    sorted_indices = np.argsort(scores)[::-1]
    num_best_frames = max(1, int(len(frames) * pstack / 100))
    top_best_indices = sorted_indices[:num_best_frames]

    reference_frame = frames[best_frame_index]
    best_frames = [frames[i] for i in top_best_indices]

    aligned_best_frames = []
    with ThreadPoolExecutor(max_workers=ncore) as executor:
        future_to_frame = {executor.submit(align_images, frame, reference_frame): frame for frame in best_frames}
        for future in tqdm(as_completed(future_to_frame), total=len(best_frames), desc="Aligning frames", disable=quiet):
            aligned_image = future.result()
            aligned_best_frames.append(aligned_image)

    final_image = stack_images(aligned_best_frames, method=stack_method)

    if resample_factor is not None:
        final_image = resample_image(final_image, resample_factor)

    if drizzle_factor is not None and drizzle_pixel_size is not None:
        final_image = drizzle(aligned_best_frames, drizzle_factor, drizzle_pixel_size)

    if flat_path:
        flat_frame = cv2.imread(flat_path, cv2.IMREAD_UNCHANGED)
        if flat_frame is None or flat_frame.shape != final_image.shape:
            if not quiet:
                print(f"Flat frame does not match the dimensions of the stacked image: {flat_path}")
            return 0, len(frames), len(aligned_best_frames)
        final_image = cv2.divide(final_image, flat_frame, scale=255.0)

    if dark_path:
        dark_frame = cv2.imread(dark_path, cv2.IMREAD_UNCHANGED)
        if dark_frame is None or dark_frame.shape != final_image.shape:
            if not quiet:
                print(f"Dark frame does not match the dimensions of the stacked image: {dark_path}")
            return 0, len(frames), len(aligned_best_frames)
        final_image = cv2.subtract(final_image, dark_frame)

    cv2.imwrite(output_path, final_image)
    if not quiet:
        print(f"Final stacked image saved at {output_path}")
    return 1, len(frames), len(aligned_best_frames)

def process_directory(directory, recursive, output_filename, stack_method, *args, **kwargs):
    video_extensions = ('.avi', '.ser')
    log_entries = []
    start_time = time.time()

    if recursive:
        video_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if f.endswith(video_extensions)]
    else:
        video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(video_extensions)]
    
    for i, video_file in enumerate(video_files, start=1):
        output_path = f"{output_filename}_{i}.png" if output_filename else f"{os.path.splitext(video_file)[0]}.png"
        start_time_file = time.time()
        result, total_frames, stacked_frames = process_video(video_file, output_path, stack_method=stack_method, *args, **kwargs)
        end_time_file = time.time()
        execution_time_file = end_time_file - start_time_file
        log_entries.append(f"{video_file},{output_path},{datetime.now().strftime('%Y%m%d%H:%M')},{execution_time_file:.2f},{total_frames},{stacked_frames}")

    return log_entries

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video frames, align, and stack them to generate a final image. (Version 0.1.5)")
    parser.add_argument('-in', '--input', help='Input video file (ser or avi format)')
    parser.add_argument('-dir', help='Directory containing video files to process')
    parser.add_argument('-rdir', help='Directory containing video files to process recursively')
    parser.add_argument('-out', '--output', default='result.png', help='Output stacked image filename (default: result.png)')
    parser.add_argument('-log', help='Log file to record processing details')
    parser.add_argument('-pstack', type=float, default=10, help='Percentage of frames to stack (default: 10)')
    parser.add_argument('-resample', type=int, choices=range(2, 6), help='Resampling factor (integer between 2 and 5)')
    parser.add_argument('-drizzlef', type=float, help='Drizzle factor')
    parser.add_argument('-drizzlep', type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help='Drizzle pixel size relative to the original pixel size (between 0.1 and 0.9)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode, suppresses all output except the final result code')
    parser.add_argument('-ncore', type=int, help='Number of cores to use (default: all cores)')
    parser.add_argument('-matrix', choices=['RGGB', 'GRBG', 'GBRG', 'BGGR'], default='RGGB', help='Bayer matrix pattern (default: RGGB)')
    parser.add_argument('-flat', help='Flat frame calibration image file')
    parser.add_argument('-dark', help='Dark frame calibration image file')
    parser.add_argument('-method', choices=['mean', 'median', 'trimmed_mean', 'robust_mean'], default='mean', help='Stacking method to use (default: mean)')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    directory = args.dir
    recursive_directory = args.rdir
    log_file = args.log
    pstack = args.pstack
    resample_factor = args.resample
    drizzle_factor = args.drizzlef
    drizzle_pixel_size = args.drizzlep
    quiet = args.quiet
    ncore = args.ncore
    matrix = args.matrix
    flat_path = args.flat
    dark_path = args.dark
    stack_method = args.method

    log_entries = []

    if directory:
        log_entries = process_directory(directory, recursive=False, output_filename=output_path, stack_method=stack_method, pstack=pstack, resample_factor=resample_factor, drizzle_factor=drizzle_factor, drizzle_pixel_size=drizzle_pixel_size, quiet=quiet, ncore=ncore, matrix=matrix, flat_path=flat_path, dark_path=dark_path)
    elif recursive_directory:
        log_entries = process_directory(recursive_directory, recursive=True, output_filename=output_path, stack_method=stack_method, pstack=pstack, resample_factor=resample_factor, drizzle_factor=drizzle_factor, drizzle_pixel_size=drizzle_pixel_size, quiet=quiet, ncore=ncore, matrix=matrix, flat_path=flat_path, dark_path=dark_path)
    elif input_path:
        start_time = time.time()
        result, total_frames, stacked_frames = process_video(input_path, output_path, pstack, resample_factor, drizzle_factor, drizzle_pixel_size, quiet, ncore, matrix, flat_path, dark_path, stack_method)
        end_time = time.time()
        execution_time = end_time - start_time
        log_entries.append(f"{input_path},{output_path},{datetime.now().strftime('%Y%m%d%H:%M')},{execution_time:.2f},{total_frames},{stacked_frames}")
    else:
        if not quiet:
            print("No input file or directory specified.")
        sys.exit(0)

    if log_file:
        with open(log_file, 'w') as f:
            f.write("\n".join(log_entries))

    if log_entries:
        for entry in log_entries:
            print(entry)
