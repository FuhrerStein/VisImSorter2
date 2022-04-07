import io
import rawpy
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from scipy import ndimage
from PyQt5 import QtGui
from PIL import Image, ImageOps, ImageChops, ImageStat

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
    convert_to = np.uint8 if im_divisions >= 15 else np.uint16

    def extract_image_data(im_record):
        img = load_image(im_record.image_paths)
        if img is None:
            return
        im_record["sizes"] = img.size
        im_record["megapixels"] = img.size[0] * img.size[1] / 1e6
        img = img.resize((240, 240), resample=Image.BILINEAR)
        img_stat = ImageStat.Stat(img.convert("RGB"))
        im_record["means"] = int(sum(img_stat.mean) // 3)
        for current_color_space, color_subspaces in selected_color_bands.items():
            converted_image = img.convert(current_color_space)
            for band in set(converted_image.getbands()) & color_subspaces:
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
    return image_full_names2.dropna()


def generate_thumb_pixmap(im_l, im_r, crop_l_orig, crop_r_orig, thm_size, thumb_mode, thumb_colored, thumb_id):

    def make_pixmap(img):
        qimage_obj = QtGui.QImage(img.tobytes(), img.width, img.height, img.width * 3, QtGui.QImage.Format_RGB888)
        pixmap_obj = QtGui.QPixmap(qimage_obj)
        return pixmap_obj

    def make_qimage(img):
        qimage_obj = QtGui.QImage(img.tobytes(), img.width, img.height, img.width * 3, QtGui.QImage.Format_RGB888)
        return qimage_obj

    image_l = load_image(im_l).convert("RGB")
    image_r = load_image(im_r).convert("RGB")
    # print(f"Pair {thumb_id} stage ***")

    if crop_l_orig is not None:
        cropped_size_l = [480 + crop_l_orig[0] + crop_l_orig[2], 480 + crop_l_orig[1] + crop_l_orig[3]]
        cropped_size_r = [480 + crop_r_orig[0] + crop_r_orig[2], 480 + crop_r_orig[1] + crop_r_orig[3]]
        crop_l, crop_r = [0] * 4, [0] * 4
        for i in range(4):
            direction = i & 1
            crop_diff = crop_l_orig[i] - crop_r_orig[i]
            if crop_diff < 0:
                crop_l[i] = crop_diff * [image_l.width, image_l.height][direction] / cropped_size_l[direction]
            elif crop_diff > 0:
                crop_r[i] = -crop_diff * [image_r.width, image_r.height][direction] / cropped_size_r[direction]

            if i > 1:
                crop_l[i] = [image_l.width, image_l.height][direction] - crop_l[i]
                crop_r[i] = [image_r.width, image_r.height][direction] - crop_r[i]

        image_l = image_l.crop(crop_l)
        image_r = image_r.crop(crop_r)
    
    image_l = image_l.resize((thm_size, thm_size), Image.BILINEAR)
    image_r = image_r.resize((thm_size, thm_size), Image.BILINEAR)

    if thumb_mode == -3:
        image_d = Image.blend(ImageOps.invert(image_l), image_r, .5)
    else:
        image_d = ImageChops.difference(image_l, image_r)

    if not thumb_colored:
        image_bands  = image_d.split()
        o = -128 if thumb_mode == -3 else 0
        image_sum_bw = ImageChops.add(ImageChops.add(image_bands[0], image_bands[1], 1, o), image_bands[2], 1, o)
        image_d = image_sum_bw.convert("RGB")

    if thumb_mode == -4:
        image_d = ImageOps.invert(image_d)
    # img_pixmap = make_pixmap(image_d)
    # qimg = make_qimage(image_d)
    print(f"Pair {thumb_id} stage ****")
    # print(f"len(image_d): {len(image_d)}", f"{thumb_id}")
    # return f"Pair {thumb_id} stage *****"
    # return [thumb_id, 5]
    # return [thumb_id, qimg]
    return thumb_id, image_d

