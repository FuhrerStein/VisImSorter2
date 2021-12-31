import io
import rawpy
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

plain_file_types = ['jpg', 'png', 'jpeg', 'gif']
raw_file_types = ['nef', 'dng']
all_file_types = plain_file_types + raw_file_types

gaussian_filter1d = ndimage.filters.gaussian_filter1d
# median_filter = ndimage.filters.median_filter
gaussian_filter = ndimage.filters.gaussian_filter


ORIENTATION_DB = dict([
    (2, [Image.FLIP_LEFT_RIGHT]),
    (3, [Image.ROTATE_180]),
    (4, [Image.FLIP_TOP_BOTTOM]),
    (5, [Image.ROTATE_90, Image.FLIP_TOP_BOTTOM]),
    (6, [Image.ROTATE_270]),
    (7, [Image.ROTATE_270, Image.FLIP_TOP_BOTTOM]),
    (8, [Image.ROTATE_90])
])


def reorient_image(im):
    image_orientation = 0

    try:
        im_exif = im.getexif()
        image_orientation = im_exif.get(274, 0)
    except (KeyError, AttributeError, TypeError, IndexError):
        # print(KeyError)
        pass

    set_of_operations = ORIENTATION_DB.get(image_orientation, [])
    for operation in set_of_operations:
        im = im.transpose(operation)
    return im


def load_image(image_full_name):
    try:
        if sum([image_full_name.lower().endswith(ex) for ex in raw_file_types]):
            f = open(image_full_name, 'rb', buffering=0)  # This is a workaround for opening cyrillic file names
            thumb = rawpy.imread(f).extract_thumb()
            img_to_read = io.BytesIO(thumb.data)
        else:
            img_to_read = image_full_name
        return reorient_image(Image.open(img_to_read))
    except Exception as e:
        print("Error reading ", image_full_name, e)


def generate_image_vector(image_full_names, selected_color_bands: dict, hg_bands, im_divisions, need_hue):
    bands_gaussian = hg_bands ** .5 / 6
    convert_to = np.uint8 if im_divisions > 64 else np.uint16

    def extract_image_data(im_record):
        img = load_image(im_record.image_paths)
        if img is None:
            return
        im_record["sizes"] = img.size
        img = img.resize((224, 224), resample=Image.BILINEAR)
        for current_color_space, color_subspaces in selected_color_bands.items():
            converted_image = img.convert(current_color_space)
            for band in converted_image.getbands():
                if band in color_subspaces:
                    combined_hg = []
                    band_array = np.asarray(converted_image.getchannel(band))
                    for big_part in np.split(band_array, im_divisions):
                        for small_part in np.split(big_part, im_divisions, 1):
                            partial_hg = np.histogram(small_part, bins=hg_bands, range=(0., 255.))[0]
                            partial_hg = gaussian_filter(partial_hg, bands_gaussian, mode='nearest')
                            combined_hg.append(partial_hg)
                    combined_hg = np.concatenate(combined_hg)
                    combined_hg_size = np.linalg.norm(combined_hg)
                    im_record[current_color_space + "_" + band] = combined_hg.astype(convert_to)
                    im_record[current_color_space + "_" + band + "_size"] = combined_hg_size
        if need_hue:
            hue_hg_one = np.histogram(img.convert("HSV").getchannel("H"), bins=256, range=(0., 255.))[0]
            max_hue = np.argmax(gaussian_filter1d(hue_hg_one, 15, mode='wrap').astype(np.uint16)).tolist()
            im_record["hue_hg"] = hue_hg_one.astype(np.uint16)
            im_record["max_hue"] = max_hue
        return im_record

    image_full_names2 = image_full_names.apply(extract_image_data, axis=1)
    return image_full_names2

