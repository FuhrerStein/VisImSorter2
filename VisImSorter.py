import fnmatch
import glob
import io
import itertools
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
from itertools import chain
from multiprocessing import Process, Queue
from time import sleep
from timeit import default_timer as timer

import PyQt5.QtCore
import matplotlib.animation as animation
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import numpy.ma as ma
from urllib.parse import unquote
import psutil
import rawpy
from PIL import Image, ImageOps, ImageChops
# from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QMainWindow, QSizePolicy, QMenu, QAction, QActionGroup
from PySide2.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from scipy import ndimage
from PaintSheet import PaintSheet, PreviewSheet

import CompareDialog
import VisImSorterInterface

image_DB = []
group_db = []
distance_db = []
angles_db = []
target_groups = 0
target_group_size = 0
progress_value = 0
progress_max = 0
image_count = 0
groups_count = 0

start_folder = ""
new_folder = ""
new_vector_folder = ""
status_text = ""
search_subdirectories = False
enforce_equal_folders = 0.
hg_bands_count = 0
compare_file_percent = 10
selected_color_spaces = []
selected_color_subspaces = []
move_files_not_copy = False
show_histograms = False
export_histograms = False
create_samples_enabled = False
run_index = 0
abort_reason = ""
gaussian_filter1d = ndimage.filters.gaussian_filter1d
median_filter = ndimage.filters.median_filter
gaussian_filter = ndimage.filters.gaussian_filter
la_norm = np.linalg.norm
folder_size_min_files = 0
final_sort = 0
group_with_similar_name = 0
plain_file_types = ['jpg', 'png', 'jpeg', 'gif']
raw_file_types = ['nef', 'dng']
folder_constraint_type = 0
folder_constraint_value = 0
stage1_grouping_type = 0
enable_stage2_grouping = False
enable_multiprocessing = True
image_paths = []
pool = mp.Pool
function_queue = []


# todo: option to change subfolder naming scheme (add color into the name, number of files)
# todo: search similar images to sample image or sample folder with images
# todo: sort final folders by: (average color, average lightness, file count, ...)
# todo: compare using 3D histograms
# todo: fix: in case of low
# todo: figure out what to do if two files have the same name and belong to the same folder

# todo: modify sorting algorithm so that final number of folders is roughly equal to desired
# todo: initial grouping is made only using image pairs rather than group pairs
# todo: when searching for closest pairs, sort also by other bands, not only by HSV hue

# todo: option to group only by pixel count
# todo: asynchronous interface: enable to change settings while scanning
# todo: scan forders and calculate difference between images in each
# todo: convert image_DB into numpy array

# todo: rewrite stage2_regroup into using pool in multiprocessing mode


'''Алгоритм выделения связных компонент

В алгоритме выделения связных компонент задается входной параметр R и в графе удаляются все ребра, для которых 
«расстояния» больше R. Соединенными остаются только наиболее близкие пары объектов. Смысл алгоритма заключается в 
том, чтобы подобрать такое значение R, лежащее в диапазон всех «расстояний», при котором граф «развалится» на 
несколько связных компонент. Полученные компоненты и есть кластеры. 

Для подбора параметра R обычно строится гистограмма распределений попарных расстояний. В задачах с хорошо выраженной 
кластерной структурой данных на гистограмме будет два пика – один соответствует внутрикластерным расстояниям, 
второй – межкластерным расстояния. Параметр R подбирается из зоны минимума между этими пиками. При этом управлять 
количеством кластеров при помощи порога расстояния довольно затруднительно. 


Алгоритм минимального покрывающего дерева

Алгоритм минимального покрывающего дерева сначала строит на графе минимальное покрывающее дерево, 
а затем последовательно удаляет ребра с наибольшим весом. 



'''


def pixmap_from_image(im):
    qimage_obj = QtGui.QImage(im.tobytes(), im.width, im.height, im.width * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap(qimage_obj)


class Config():
    search_duplicates = False
    image_divisions = 1
    compare_by_angles = False
    max_likeness = 100


conf = Config()


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


def run_sequence():
    global progress_value
    global progress_max
    global status_text
    global function_queue
    local_run_index = run_index
    # def none_function(): pass

    if enable_multiprocessing:
        function_queue.append(init_pool)

    function_queue.append(scan_for_images)
    function_queue.append(generate_image_vectors_and_groups)

    if conf.search_duplicates:
        function_queue.append(close_pool)
        # function_queue.append(create_triangle_distance_db)
        function_queue.append(create_triangle_distance_db)
        function_queue.append(show_closest_images_gui)
    else:
        if stage1_grouping_type == 0:
            function_queue.append(create_simple_groups)
        elif stage1_grouping_type == 1:
            function_queue.append(create_groups_by_similarity)
        elif stage1_grouping_type == 2:
            function_queue.append(create_groups_v4)

        if enable_stage2_grouping:
            function_queue.append(stage2_regroup)

        function_queue.append(final_group_sort)
        function_queue.append(choose_destination_folder)
        function_queue.append(move_files)

        if export_histograms:
            function_queue.append(export_image_vectors)

        if create_samples_enabled:
            function_queue.append(create_samples)

    if enable_multiprocessing:
        function_queue.append(close_pool)

    start_time = timer()

    parent = psutil.Process()
    if psutil.LINUX:
        parent.nice(15)
    else:
        parent.nice(psutil.IDLE_PRIORITY_CLASS)

    while len(function_queue):
        single_function = function_queue.pop(0)
        single_function()
        if local_run_index != run_index:
            if enable_multiprocessing:
                close_pool()
            return

    finish_time = timer()

    if show_histograms:
        AnimatedHistogram(image_DB, group_db)

    status_text = "Finished. Elapsed " + "%d:%02d" % divmod(finish_time - start_time, 60) + " minutes."
    print(status_text)
    progress_max = 100
    progress_value = 100


def close_pool():
    global pool
    pool.terminate()
    pool.close()


def init_pool():
    global pool
    pool = mp.Pool()


def scan_for_images():
    global status_text
    # global start_folder
    # global search_subdirectories
    global run_index
    global abort_reason
    global image_paths
    status_text = "Scanning images..."
    print(status_text)
    QApplication.processEvents()

    types = plain_file_types + raw_file_types
    image_paths = []
    image_paths_grouped = []

    all_files = glob.glob(start_folder + "/**/*", recursive=search_subdirectories)

    for file_mask in types:
        image_paths.extend(fnmatch.filter(all_files, "*." + file_mask))

    image_paths = [os.path.normpath(im) for im in image_paths]

    if len(image_paths) == 0:
        abort_reason = "No images found."
        run_index += 1
        QMessageBox.warning(None, "VisImSorter: Error", abort_reason)
        print(abort_reason)
    elif group_with_similar_name > 0:
        file_name_groups = sorted(image_paths, key=lambda x: os.path.basename(x)[:group_with_similar_name])
        image_paths_grouped = [list(it) for k, it in itertools.groupby(file_name_groups,
             key=lambda x: os.path.basename(x)[:group_with_similar_name])]
    else:
        image_paths_grouped = [[it] for it in image_paths]
    image_paths = image_paths_grouped


def generate_image_vectors_and_groups():
    global image_DB
    global image_count
    global target_group_size
    global target_groups
    global progress_max
    global progress_value
    global group_db
    global groups_count
    global run_index
    global abort_reason
    global pool
    global folder_size_min_files
    global folder_constraint_type
    global folder_constraint_value
    global image_paths

    local_run_index = run_index
    image_count = len(image_paths)

    progress_max = image_count
    progress_value = 0
    image_DB = []
    results = []

    for image_path in image_paths:
        result = pool.apply_async(generate_image_vector,
                                  (image_path, selected_color_spaces, selected_color_subspaces, hg_bands_count,
                                   conf.image_divisions, not conf.search_duplicates),
                                  callback=generate_image_vector_callback)
        results.append(result)

    for res in results:
        while not res.ready():
            QApplication.processEvents()
            sleep(.05)
            if local_run_index != run_index:
                return
        res.wait()

    if not conf.search_duplicates:
        image_DB.sort(key=lambda x: x[2])
    image_count = len(image_DB)
    real_image_count = sum([x[3] for x in image_DB])

    if image_count == 0:
        abort_reason = "Failed to create image database."
        run_index += 1
        print(abort_reason)

    if local_run_index != run_index:
        return

    if folder_constraint_type:
        target_groups = int(folder_constraint_value)
        target_group_size = int(round(real_image_count / target_groups))
    else:
        target_group_size = int(folder_constraint_value)
        target_groups = int(round(real_image_count / target_group_size))


def generate_image_vector_callback(vector_pack):
    global image_DB
    global progress_value
    global status_text

    if vector_pack is not None:
        image_DB.append(vector_pack)
    progress_value += 1
    status_text = f"(1/4) Generating image histograms... ({progress_value:d} of {image_count:d}"
    QApplication.processEvents()


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


def generate_image_vector(image_full_names, color_spaces, color_subspaces, hg_bands, im_divisions, need_hue):
    hue_hg = None
    average_hg = None
    bands_gaussian = hg_bands ** .5 / 6
    size = None

    def get_band_hg(image):
        partial_hg = np.histogram(image, bins=hg_bands, range=(0., 255.))[0]
        partial_hg = gaussian_filter(partial_hg, bands_gaussian, mode='nearest')
        combined_hg.append(partial_hg)
        QApplication.processEvents()

    for image_full_name in image_full_names:
        img = load_image(image_full_name)
        if img is None:
            continue
        if not size:
            size = img.size
        img = img.resize((224, 224), resample=Image.BILINEAR)

        combined_hg = []
        for current_color_space in color_spaces:
            converted_image = img.convert(current_color_space)
            for band in converted_image.getbands():
                if band not in color_subspaces:
                    continue
                band_array = np.asarray(converted_image.getchannel(band))

                for big_part in np.split(band_array, im_divisions):
                    for small_part in np.split(big_part, im_divisions, 1):
                        get_band_hg(small_part)
        combined_hg = np.concatenate(combined_hg)
        average_hg = average_hg + combined_hg if average_hg else combined_hg

        if need_hue:
            hue_hg_one = np.histogram(img.convert("HSV").getchannel("H"), bins=256, range=(0., 255.))[0]
            hue_hg = hue_hg + hue_hg_one if hue_hg else hue_hg_one

    images_in_row = len(image_full_names)
    if images_in_row > 1:
        average_hg = average_hg // images_in_row
        hue_hg = hue_hg // images_in_row

    if need_hue:
        max_hue = np.argmax(gaussian_filter1d(hue_hg, 15, mode='wrap').astype(np.uint16)).tolist()
        hue_hg = hue_hg.astype(np.uint16)
    else:
        max_hue = None

    if im_divisions > 64:
        average_hg = average_hg.astype(np.uint8)
    else:
        average_hg = average_hg.astype(np.uint16)

    return [image_full_names, average_hg, max_hue, images_in_row, hue_hg, size]


def create_simple_groups():
    global group_db
    global groups_count

    group_db = []
    group_im_indexes = np.array_split(range(image_count), target_groups)
    for new_group_image_list in group_im_indexes:
        new_group_vector = np.mean(np.array([image_DB[i][1] for i in new_group_image_list]), axis=0)
        new_group_hue = np.argmax(np.sum(np.array([image_DB[i][4] for i in new_group_image_list]), axis=0))
        new_group_image_count = sum([image_DB[i][3] for i in new_group_image_list])
        group_db.append([new_group_image_list.tolist(), new_group_vector, new_group_hue, new_group_image_count])

    group_db = sorted(group_db, key=lambda x: x[2])
    groups_count = len(group_db)


def create_groups_v4():
    global image_count
    global status_text
    global target_groups
    global groups_count
    global run_index
    global pool
    global group_db
    global image_DB
    global distance_db
    global progress_max
    global progress_value

    local_run_index = run_index
    batches = 1
    batch = 0
    compare_limit = int(image_count * .1 + 200)

    groups_history = []
    groups_open_history = []
    group_ratings_history = []
    group_invites_history = []

    status_text = "(2/4) Comparing images..."
    progress_max = image_count

    relations_db = create_relations_db(image_DB, batch, batches, compare_limit)

    status_text = "(3/4) Choosing center images..."

    eef01 = (enforce_equal_folders + .5) ** 2
    eef01_divisor = np.tanh(eef01) + 1

    groups_open = np.ones(image_count, dtype=bool)
    invite_count = np.ones(image_count) * target_group_size
    image_index_list = np.arange(image_count)
    image_belongings = image_index_list
    image_belongings_old = image_belongings
    image_belongings_final = image_belongings
    mutual_ratings = (np.tanh(eef01 * (1 - relations_db['rank'] / target_group_size)) + 1) / eef01_divisor
    group_ratings = np.sum((mutual_ratings - np.eye(image_count)) / (1 + relations_db['dist']), axis=1)
    group_ratings /= group_ratings.max()
    size_corrector = 1.
    minimal_distance = relations_db['dist'][relations_db['dist'] > 1].min()
    mutual_weights = 0

    progress_max = 1500
    for _ in range(progress_max):
        if (image_belongings == image_belongings_old).all():
            actual_groups = len(np.unique(image_belongings))
            group_sizes = np.bincount(image_belongings, minlength=image_count)
            # actual_groups2 = len(group_sizes.nonzero())
            image_belongings_final = image_belongings
            if actual_groups > target_groups:
                # image_belongings = image_index_list
                # todo: make an elegant a_coefficient function
                # todo: maybe reset every invite_count on some occasions
                # todo: use bigger steps for very large number of groups
                # todo: increase attraction to groups that have least mean distance or right file count

                # desmos
                # \tanh\left(1-\left(\frac{x}{b}\right)^{c}-\frac{b}{\left(15-a\right)x}\right)
                # \frac{1}{1+\frac{1}{\exp\left(x\right)}}

                # cheap sigmoid
                # \frac{\left(x - b\right)}{2} / (1 +\operatorname{abs}(x-b))+.5
                # \frac{x-b}{2\left(1+\operatorname{abs}\left(x-b\right)\right)}+.5
                # \frac{.5}{\operatorname{sign}\left(x-b\right)+\frac{1}{x-b}}+.5

                # mega
                # \frac{1}{1+20^{5\left(\frac{x}{b}-1\right)}}\cdot\left(\frac{-.5}{\operatorname{sign}\left(x-b\right)+\frac{ab}{x-b}}+.5\right)
                # \frac{1}{1+20^{\frac{5\left(x-b\right)}{b}}}\cdot\left(\frac{-.5}{\operatorname{sign}\left(x-b\right)+\frac{ab}{x-b}}+.5\right)

                groups_open = (image_belongings == image_index_list)
                size_corrector *= 1.1
                distance_limit = minimal_distance * size_corrector ** .2
                mutual_distance = relations_db['dist'] - distance_limit
                distance_weights = .5 - .5 / (np.sign(mutual_distance) + 1 / mutual_distance)
                # distance_weights = (.5 - .5 / (np.sign(mutual_distance) + .05 * distance_limit / mutual_distance))
                # distance_weights /= (1 + 20 ** (5 * mutual_distance / distance_limit))
                mutual_weights = distance_weights  # * group_size_weights
                group_ratings = np.sum(mutual_weights / (relations_db['dist'] + 500) * (1 - np.eye(image_count)),
                                       axis=1)
                group_ratings /= group_ratings.max()
                image_belongings = image_index_list
                # print(_, actual_groups, size_corrector)

            else:
                break

        # groups_history.append(image_belongings)
        # groups_open_history.append(groups_open)
        # group_ratings_history.append(group_ratings)
        # group_invites_history.append(group_invites)

        image_belongings_old = image_belongings
        groups_open_ratings = (image_belongings == image_index_list) * group_ratings
        group_invites = mutual_weights * groups_open_ratings
        image_belongings = np.argmax(group_invites, axis=1)

        closed_groups_indexes = (image_belongings != image_index_list).nonzero()
        images_in_closed_groups = np.isin(image_belongings, closed_groups_indexes)
        # invite_count = ma.array(invite_count, mask=images_in_closed_groups).filled(np.ones(image_count) * target_group_size)
        image_belongings = ma.array(image_belongings, mask=images_in_closed_groups).filled(image_index_list)

        progress_value = _
        QApplication.processEvents()

    # base_image_names = np.array([os.path.basename(i[0][0]) for i in image_DB])
    # save_log(groups_history, "groups_history", base_image_names)
    # save_log(groups_open_history, "groups_open_history", base_image_names)

    status_text = "(4/4) Grouping images..."
    progress_value = 0
    QApplication.processEvents()

    center_index_list = np.unique(image_belongings_final)

    group_db = []
    for im_n in center_index_list:
        indexes_in_group = (image_belongings_final == im_n).nonzero()[0].tolist()
        # indexes_in_group = [im_n]
        group_db.append([indexes_in_group, None, None, None])

    for grp in group_db:
        grp[1] = np.mean(np.array([image_DB[i][1] for i in grp[0]]), axis=0)
        grp[2] = np.argmax(np.sum(np.array([image_DB[i][4] for i in grp[0]]), axis=0))
        grp[3] = np.sum(np.array([image_DB[i][3] for i in grp[0]]), axis=0)
    groups_count = len(group_db)


def show_closest_images_gui():
    global status_text
    global progress_max
    global progress_value
    global abort_reason
    global run_index

    progress_value = 0

    if len(distance_db) == 0:
        abort_reason = "No similar enough images found"
        run_index += 1
        return

    status_text = "Preparation complete. Showing image pairs."
    QApplication.processEvents()

    compare_wnd = CompareGUI()
    compare_wnd.show()
    while compare_wnd.isVisible():
        sleep(.1)
        QApplication.processEvents()
    compare_wnd.image_delete_candidates.clear()
    compare_wnd.marked_pairs.clear()

def create_triangle_distance_db():
    global progress_value
    global status_text
    global distance_db
    status_text = "(2/4) Comparing images "
    if conf.compare_by_angles:
        status_text += "(angles)..."
    else:
        status_text += "(distances)..."
    im_vectors = np.array([i[1] for i in image_DB], dtype=np.int32)
    im_vector_sizes = la_norm(im_vectors, axis=1) if conf.compare_by_angles else 1
    triangle_distance_db_list = []

    def make_angles(idx):
        im_db = np.zeros(idx, dtype=[('dist', float), ('im_1', int), ('im_2', int)])
        if conf.compare_by_angles:
            distances = 1 - np.dot(im_vectors[:idx], im_vectors[idx]) / im_vector_sizes[:idx] / im_vector_sizes[idx]
        else:
            distances = la_norm((im_vectors[:idx] - im_vectors[idx]), axis=1).astype(int)
        im_db['dist'] = distances
        im_db['im_1'] = idx
        im_db['im_2'] = range(idx)
        if conf.max_likeness < 100 or not conf.compare_by_angles:
            im_db = im_db[distances < conf.max_likeness]
        triangle_distance_db_list.append(im_db)
        if image_count - idx > idx:
            make_angles(image_count - idx)

    for idx in range(1, image_count // 2):
        make_angles(idx)
        progress_value = idx * 2
        QApplication.processEvents()
    status_text = "(3/4) Final resorting ..."
    QApplication.processEvents()
    distance_db = np.sort(np.concatenate(triangle_distance_db_list), order='dist')


def create_relations_db(im_db, batch, batches, compare_lim):
    global progress_value
    im_count = len(im_db)
    image_index_list = np.arange(im_count)
    compare_count = min(compare_lim * 2 + 1, im_count)
    relations_db = []
    base_image_coordinates = np.array([i[1] for i in image_DB], dtype=np.int32)

    for im_1 in np.array_split(image_index_list, batches)[batch]:
        second_range = np.roll(image_index_list, compare_lim - im_1)
        im_1_db = np.zeros(im_count, dtype=[('dist', np.float), ('rank', np.int32), ('im_1', np.int), ('im_2', np.int)])
        im_1_db['im_1'] = im_1
        im_1_db['im_2'] = second_range
        im_1_db['dist'] = np.inf
        # im_1_coordinates = im_db[im_1][1].astype(np.int32)
        # im_2_coordinates = base_image_coordinates[second_range[:compare_count]].astype(np.int32)
        im_1_coordinates = im_db[im_1][1]
        im_2_coordinates = base_image_coordinates[second_range[:compare_count]]
        # im_2_coordinates = im_db[second_range[:compare_count]][1]
        vector_distance_line = la_norm((im_2_coordinates - im_1_coordinates), axis=1).astype(int)
        im_1_db[:compare_count]['dist'] = vector_distance_line
        im_1_db.sort(order='dist')
        im_1_db['rank'] = image_index_list.astype(int)
        im_1_db.sort(order='im_2')
        relations_db.append(im_1_db)
        progress_value = im_1
        QApplication.processEvents()
    relations_db = np.array(relations_db).T

    return relations_db


# noinspection PyTypeChecker
def save_log(log_values, log_name, base_image_names):
    logs_dir = "y:\\logs\\"
    if not os.path.isdir(logs_dir):
        try:
            os.makedirs(logs_dir)
        except Exception as e:
            print("Could not create folder ", e)
    np.savetxt(logs_dir + log_name + ".csv", np.vstack([base_image_names, log_values]), fmt='%s', delimiter=";",
               encoding="utf-8")


def create_groups_by_similarity():
    global image_count
    global status_text
    global target_groups
    global groups_count
    global run_index
    global pool
    global group_db
    global distance_db

    local_run_index = run_index
    step = 0

    group_db = [[[counter]] + image_record[1:4] for counter, image_record in enumerate(image_DB)]
    groups_count = len(group_db)

    feedback_queue = mp.Manager().Queue()
    while groups_count > target_groups:
        step += 1
        status_text = "Joining groups, pass %d. Now joined %d of %d . To join %d groups"
        status_text %= (step, image_count - groups_count, image_count - target_groups, groups_count - target_groups)

        fill_distance_db(feedback_queue, step)
        if local_run_index != run_index:
            return

        merge_group_batches()
        if local_run_index != run_index:
            return


def fill_distance_db(feedback_queue, step):
    global group_db
    global distance_db
    global image_count
    global target_groups
    global target_group_size
    global progress_max
    global progress_value
    global groups_count
    global run_index
    global enforce_equal_folders
    global pool

    local_run_index = run_index
    progress_max = groups_count

    distance_db = []
    results = []
    batches = min(groups_count // 200 + 1, mp.cpu_count())
    # batches = 1
    progress_value = 0

    compare_file_limit = int(compare_file_percent * (image_count + 2000) / 100)

    entries_limit = groups_count // 4 + 1
    entries_limit = groups_count - target_groups

    feedback_list = [0] * batches
    task_running = False
    if batches == 1:
        distance_db += fill_distance_db_sub(group_db, 1, 0, enforce_equal_folders, compare_file_limit,
                                            target_group_size, entries_limit)
    else:
        for batch in range(batches):
            task_running = True
            args = [group_db, batches, batch, enforce_equal_folders, compare_file_limit, target_group_size,
                    entries_limit, feedback_queue]
            result = pool.apply_async(fill_distance_db_sub, args=args, callback=fill_distance_db_callback)
            results.append(result)
            QApplication.processEvents()
            if local_run_index != run_index:
                return

    while task_running:
        task_running = False
        for res in results:
            if not res.ready():
                task_running = True
        while not feedback_queue.empty():
            feedback_bit = feedback_queue.get()
            feedback_list[feedback_bit[0]] = feedback_bit[1]
            progress_value = sum(feedback_list) // batches
        QApplication.processEvents()

    distance_db_unfiltered = sorted(distance_db, key=lambda x: x[0])  # [:entries_limit]

    used_groups = []
    distance_db = []
    for dist, g1, g2 in distance_db_unfiltered:
        if not ((g1 in used_groups) or (g2 in used_groups)):
            distance_db.append([g1, g2])
            used_groups += [g1, g2]
        if len(distance_db) > entries_limit:
            break


def fill_distance_db_callback(in_result):
    global progress_value
    global distance_db
    distance_db += in_result


def fill_distance_db_sub(group_db_l, batches, batch, enforce_ef, compare_lim, target_gs, entries_lim, feedback=None):
    global progress_value
    groups_count_l = len(group_db_l)
    compute_result = []
    simple_counter = 10
    for index_1 in range(batch, groups_count_l, batches):
        best_pair = None
        min_distance = float("inf")
        end_of_search_window = index_1 + compare_lim
        if groups_count_l >= end_of_search_window:
            second_range = range(index_1 + 1, end_of_search_window)
        else:
            rest_of_search_window = min(end_of_search_window - groups_count_l, index_1)
            second_range = chain(range(index_1 + 1, groups_count_l), range(0, rest_of_search_window))
        for index_2 in second_range:
            vector_distance = la_norm(group_db_l[index_1][1].astype(np.int32) - group_db_l[index_2][1].astype(np.int32))
            size_factor = (group_db_l[index_1][3] + group_db_l[index_2][3]) / target_gs
            vector_distance *= (size_factor ** enforce_ef)
            if min_distance > vector_distance:
                min_distance = vector_distance
                best_pair = [index_1, index_2]
            simple_counter -= 1
            if simple_counter == 0:
                simple_counter = 5000
                if feedback is None:
                    progress_value = index_1
                    QApplication.processEvents()
                else:
                    feedback.put([batch, index_1])
        if best_pair is not None:
            compute_result.append([min_distance] + best_pair)

    return compute_result


def merge_group_batches():
    global group_db
    global distance_db
    global target_groups
    global groups_count
    global run_index
    global image_DB

    local_run_index = run_index
    removed_count = 0

    for g1, g2 in distance_db:
        if groups_count - removed_count <= target_groups:
            break
        new_group_image_list = group_db[g1][0] + group_db[g2][0]
        new_group_vector = np.mean(np.array([image_DB[i][1] for i in new_group_image_list]), axis=0)
        new_group_hue = np.argmax(np.sum(np.array([image_DB[i][4] for i in new_group_image_list]), axis=0))
        new_group_image_count = sum([image_DB[i][3] for i in new_group_image_list])
        group_db.append([new_group_image_list, new_group_vector, new_group_hue, new_group_image_count])
        group_db[g1][3] = 0
        group_db[g2][3] = 0
        removed_count += 1
        if local_run_index != run_index:
            return
        QApplication.processEvents()
    group_db_filtered = filter(lambda x: x[3] > 0, group_db)
    group_db = sorted(group_db_filtered, key=lambda x: x[2])
    groups_count = len(group_db)


def stage2_sort_search(im_db, target_group_sz, enforce_ef, gr_db, work_group, target_groups_list):
    groups_count_l = len(gr_db)
    return_db = []
    for im_index in gr_db[work_group][0]:
        current_distance = la_norm(gr_db[work_group][1].astype(np.int32) - im_db[im_index][1].astype(np.int32))
        current_distance *= (gr_db[work_group][3] / target_group_sz) ** enforce_ef
        best_distance = current_distance
        new_group = -1
        distance_difference = 0
        for target_group in range(groups_count_l):
            if target_group == work_group:
                continue
            if not target_groups_list[work_group][target_group]:
                continue
            distance_to_target = la_norm(gr_db[target_group][1].astype(np.int32) - im_db[im_index][1].astype(np.int32))
            distance_to_target *= (gr_db[target_group][3] / target_group_sz) ** enforce_ef
            if distance_to_target < best_distance:
                new_group = target_group
                best_distance = distance_to_target
                distance_difference = current_distance - distance_to_target
        if new_group != -1:
            return_db.append([im_index, work_group, new_group, distance_difference])
            gr_db[work_group][0].remove(im_index)
            gr_db[new_group][0].append(im_index)
            gr_db[work_group][1] = np.mean(np.array([im_db[i][1] for i in gr_db[work_group][0]]), axis=0)
            gr_db[new_group][1] = np.mean(np.array([im_db[i][1] for i in gr_db[new_group][0]]), axis=0)
            gr_db[work_group][3] -= im_db[im_index][3]
            gr_db[new_group][3] += im_db[im_index][3]
    if len(return_db) > 0:
        return_db = sorted(return_db, key=lambda x: x[3], reverse=True)
        return return_db
    else:
        return [[None, work_group]]


def stage2_sort_worker(input_q, output_q, im_db, target_group_sz, enforce_ef):
    for args in iter(input_q.get, 'STOP'):
        result = stage2_sort_search(im_db, target_group_sz, enforce_ef, args[0], args[1], args[2])
        output_q.put(result)


def stage2_regroup():
    global image_DB
    global group_db
    global groups_count
    global progress_value
    global progress_max
    global target_group_size
    global enforce_equal_folders
    global run_index
    global status_text

    local_run_index = run_index

    moved_images = 0
    progress_value = 0
    progress_max = groups_count * groups_count
    idle_runs = 0

    status_text = "Resorting started... "
    QApplication.processEvents()

    parallel_processes = mp.cpu_count()
    task_index = 0
    lookup_order = list(range(groups_count))
    target_groups_list = np.ones((groups_count, groups_count), dtype=np.bool)

    task_queue = Queue()
    done_queue = Queue()

    for i in range(parallel_processes):
        Process(target=stage2_sort_worker, args=(task_queue, done_queue, image_DB, target_group_size,
                                                 enforce_equal_folders)).start()
        task_queue.put([group_db, 0, target_groups_list])
        QApplication.processEvents()

    while idle_runs < groups_count * 2:
        for i in range(parallel_processes):
            task_index += 1
            if task_index == groups_count:
                task_index = 0
                random.shuffle(lookup_order)
            task_queue.put([group_db, lookup_order[task_index], target_groups_list])

        for group_n in range(parallel_processes):
            next_result = done_queue.get()
            moved_in_this_step = 0
            if next_result[0][0] is not None:
                for move_order in next_result:
                    move_result = move_image_between_groups(move_order)
                    if move_result:
                        moved_in_this_step += 1
                        target_groups_list[move_order[2]] = True
                        target_groups_list[:, move_order[1]] = True
            if moved_in_this_step:
                moved_images += moved_in_this_step
                idle_runs = 0
            else:
                idle_runs += 1
                target_groups_list[next_result[0][1]] = False
        progress_value = progress_value * .99 + (progress_max - np.sum(target_groups_list)) * .01
        status_text = "Resorting. Relocated " + str(moved_images) + " files"
        QApplication.processEvents()

        if local_run_index != run_index:
            break

    for i in range(parallel_processes):
        task_queue.put('STOP')


def final_group_sort():
    global image_DB
    global group_db

    if final_sort == 0:
        for grp in group_db:
            grp[2] = np.argmax(np.sum(np.array([image_DB[i][3] for i in grp[0]]), axis=0))
        group_db.sort(key=lambda x: x[2])
    elif final_sort == 1:
        group_db.sort(key=lambda x: x[3])
    elif final_sort == 2:
        group_db.sort(key=lambda x: x[3], reverse=True)


def move_image_between_groups(task):
    global group_db
    global image_DB
    global target_group_size
    global enforce_equal_folders

    im_index = task[0]
    g1 = task[1]
    g2 = task[2]

    if im_index not in group_db[g1][0]:
        return False

    distance_to_g1 = la_norm(group_db[g1][1].astype(np.int32) - image_DB[im_index][1].astype(np.int32))
    distance_to_g1 *= (group_db[g1][3] / target_group_size) ** enforce_equal_folders
    distance_to_new_g2 = la_norm(group_db[g2][1].astype(np.int32) - image_DB[im_index][1].astype(np.int32))
    distance_to_new_g2 *= ((group_db[g2][3] + image_DB[im_index][3]) / target_group_size) ** enforce_equal_folders
    if distance_to_g1 > distance_to_new_g2:
        group_db[g1][0].remove(im_index)
        group_db[g2][0].append(im_index)
        for g in [g1, g2]:
            group_db[g][1] = np.mean(np.array([image_DB[i][1] for i in group_db[g][0]]), axis=0)
            group_db[g][3] = sum([image_DB[i][3] for i in group_db[g][0]])
        return True
    else:
        return False


def choose_destination_folder():
    global new_vector_folder
    global new_folder
    base_folder = start_folder
    destination_folder_index = -1
    if re.match(".*_sorted_\d\d\d$", start_folder):
        base_folder = start_folder[:-11]
    elif re.match(".*_sorted$", start_folder):
        base_folder = start_folder[:-7]
        destination_folder_index = 0

    list_of_indices = sorted(glob.glob(base_folder + "_sorted_[0-9][0-9][0-9]"), reverse=True)
    if len(list_of_indices) == 0:
        if len(glob.glob(base_folder + "_sorted")) > 0:
            destination_folder_index = 0
    if len(list_of_indices) > 0:
        destination_folder_index = int(list_of_indices[0][-3:]) + 1

    destination_folder_index_suffix = "/"
    if destination_folder_index >= 0:
        destination_folder_index_suffix = "_" + "%03d" % destination_folder_index + "/"

    new_folder = base_folder + "_sorted" + destination_folder_index_suffix
    new_vector_folder = base_folder + "_histograms" + destination_folder_index_suffix


def move_files():
    global group_db
    global image_DB
    global move_files_not_copy
    global status_text
    global progress_max
    global progress_value
    global groups_count
    global run_index
    global new_folder
    global start_folder
    global folder_size_min_files

    local_run_index = run_index

    progress_max = groups_count
    status_text = "Sorting done. " + ["Copying ", "Moving "][move_files_not_copy] + "started"
    print(status_text)
    progress_value = 0
    QApplication.processEvents()

    new_folder_digits = int(math.log(target_groups, 10)) + 1

    action = shutil.copy
    if move_files_not_copy:
        action = shutil.move

    try:
        os.makedirs(new_folder)
    except Exception as e:
        print("Could not create folder ", e)

    ungroupables = []
    for group_index, grp in enumerate(group_db):
        if grp[3] >= folder_size_min_files:
            dir_name = get_group_color_name(group_index)
            dir_name = new_folder + str(group_index + 1).zfill(new_folder_digits) + " - (%03d)" % grp[3] + dir_name
            try:
                os.makedirs(dir_name)
            except Exception as e:
                print("Could not create folder ", e)
            for i in grp[0]:
                for im_path in image_DB[i][0]:
                    try:
                        action(im_path, dir_name + "/")
                    except Exception as e:
                        print("Could not complete file operation ", e)
                if local_run_index != run_index:
                    return
                QApplication.processEvents()
        else:
            ungroupables += grp[0]
        progress_value = group_index

    if len(ungroupables) > 0:
        files_count = sum([image_DB[i][3] for i in ungroupables])
        dir_name = new_folder + "_ungroupped" + " - (%03d)" % files_count
        try:
            os.makedirs(dir_name)
        except Exception as e:
            print("Could not create folder", e)
        for i in ungroupables:
            for im_path in image_DB[i][0]:
                try:
                    action(im_path, dir_name)
                except Exception as e:
                    print("Could not complete file operation ", e)
            if local_run_index != run_index:
                return
            QApplication.processEvents()

    if move_files_not_copy:
        start_folder = new_folder


def get_group_color_name(group_n):
    global group_db
    global new_folder
    if "HSV" not in selected_color_spaces:
        return ""
    for subspace in "HSV":
        if subspace not in selected_color_subspaces:
            return ""

    # colors_list = mc.CSS4_COLORS
    colors_list = mc.XKCD_COLORS

    group_color = np.split(group_db[group_n][1][:hg_bands_count * 3], 3)
    group_color = ndimage.zoom(group_color, [1, 256 / hg_bands_count], mode='constant')
    group_color_one = np.zeros(3)
    group_color_one[0] = np.argmax(gaussian_filter1d(group_color[0], 30, mode="wrap")) / 256
    for i in [1, 2]:
        median_value = np.percentile(group_color[i], 70)
        filtered_channel = np.where(group_color[i] > median_value, group_color[i], 0)
        group_color_one[i] = ndimage.center_of_mass(filtered_channel)[0] / 256

    min_colours = {}
    for color_name, color_value in colors_list.items():
        color_value_hsv = mc.rgb_to_hsv(mc.to_rgb(color_value))
        min_colours[la_norm(color_value_hsv - group_color_one)] = color_name

    return " " + min_colours[min(min_colours.keys())][5:].replace("/", "-")


def export_image_vectors():
    global group_db
    global groups_count
    global run_index
    global new_vector_folder

    local_run_index = run_index

    try:
        os.makedirs(new_vector_folder)
    except Exception as e:
        print("Folder already exists", e)

    save_db = [row[1] for row in group_db]
    np.savetxt(new_vector_folder + "Groups.csv", save_db, fmt='%1.4f', delimiter=";")

    for group_n in range(groups_count):
        save_db = []
        for image_index in group_db[group_n][0]:
            save_db.append(
                [image_DB[image_index][0][0]] + image_DB[image_index][1] + [image_DB[image_index][2]])
        np.savetxt(new_vector_folder + "Group_vectors_" + str(group_n + 1) + ".csv", save_db, delimiter=";",
                   fmt='%s', newline='\n')
        if local_run_index != run_index:
            return


def create_samples():
    global group_db
    global groups_count
    global run_index
    global new_vector_folder
    global progress_max
    global progress_value
    global status_text

    if "HSV" not in selected_color_spaces:
        return
    for subspace in "HSV":
        if subspace not in selected_color_subspaces:
            return

    local_run_index = run_index

    roll_settings = [[0, 1500], [1, 1500],
                     [0, 1500], [1, 1500],
                     [0, 1500], [1, 1500],
                     [0, 1500], [1, 1500],
                     [0, 1500], [1, 1500]]

    try:
        os.makedirs(new_vector_folder + "/subvectors")
    except Exception as e:
        print("Folder already exists", e)

    status_text = "Generating folder images."
    QApplication.processEvents()

    progress_max = groups_count * 3
    progress_value = 0

    for group_n in range(groups_count):
        im_size = 2000, 1200
        random.seed()

        img_data = [Image] * 3
        for color_band in range(3):
            img_data_band = np.array([], dtype='uint8')
            bins_sum = 0.
            for hg_bin in range(hg_bands_count):
                bins_sum += group_db[group_n][1][color_band * hg_bands_count + hg_bin]
            band_coefficient = im_size[0] * im_size[1] / bins_sum / 2
            for hg_bin in range(hg_bands_count):
                bin_start = 256 * hg_bin // hg_bands_count
                bin_end = 256 * (hg_bin + 1) // hg_bands_count
                band_index = color_band * hg_bands_count + hg_bin
                bin_value = int(group_db[group_n][1][band_index] * band_coefficient)
                img_data_chunk = np.linspace(bin_start, bin_end, bin_value, endpoint=False, dtype='uint8')
                img_data_band = np.append(img_data_band, img_data_chunk)
            if im_size[0] * im_size[1] // 2 > len(img_data_band):
                img_data_chunk = np.full(im_size[0] * im_size[1] // 2 - len(img_data_band), 255, dtype='uint8')
                img_data_band = np.append(img_data_band, img_data_chunk)
            if im_size[0] * im_size[1] // 2 < len(img_data_band):
                img_data_band = np.resize(img_data_band, im_size[0] * im_size[1] // 2)

            img_data_band = np.reshape(img_data_band, (im_size[1] // 2, im_size[0]))
            img_data_band = np.vstack([img_data_band, np.flipud(img_data_band)])
            for one_set in roll_settings:
                img_data_band = roll_rows2(img_data_band, one_set[0], one_set[1])
                if local_run_index != run_index:
                    return

            gaussian_filter(img_data_band, 2, output=img_data_band, mode='wrap')
            median_filter(img_data_band, 7, output=img_data_band, mode='wrap')

            img_data[color_band] = Image.frombytes("L", im_size, img_data_band.copy(order='C'))

            progress_value += 1
            QApplication.processEvents()

            img_data[color_band].save(
                new_vector_folder + "/subvectors/" + "Group_" + "%03d_s_%d" % ((group_n + 1), color_band) + ".png")

        img = Image.merge("HSV", img_data)

        image_rgb = img.convert("RGB")
        image_rgb.save(new_vector_folder + "Group_" + "%03d" % (group_n + 1) + ".png")

        status_text = "Generating forlder images. " + "%1d of %1d done." % ((group_n + 1), groups_count)


def roll_rows2(input_layer, axis, max_speed):
    if axis == 1:
        input_layer = input_layer.T

    speed_row = np.linspace(-max_speed, max_speed, input_layer.shape[0])
    random.shuffle(speed_row)
    speed_row = gaussian_filter1d(speed_row, math.sqrt(max_speed) * 5, mode='wrap')
    displacement_row = np.zeros_like(speed_row)

    for i in range(10):
        displacement_row = np.roll(np.add(displacement_row * .9, speed_row), 1)
    displacement_row = displacement_row.astype('int32')

    moved_list = map(np.roll, input_layer, displacement_row)
    input_layer = np.array(list(moved_list), copy=False)

    if axis == 1:
        input_layer = input_layer.T

    return input_layer.copy(order='C')


# This function is currently not used
def rotate_squares(input_layer, square_size):
    result = []
    for chunk in np.split(input_layer, input_layer.shape[0] // square_size, 0):
        rows = []
        for square in np.split(chunk, input_layer.shape[1] // square_size, 1):
            rows.append(np.rot90(square, random.randrange(4)))
        result.append(rows)
        QApplication.processEvents()
    result = np.roll(np.block(result), square_size // 2, axis=(0, 1))
    return result


# This function is currently not used
def shuffle_single_layer(input_layer, square_size):
    w = input_layer.shape[1]
    h = input_layer.shape[0]
    squares = input_layer
    squares = np.hsplit(squares, w // square_size)
    squares = np.vstack(squares)
    squares = np.reshape(squares, -1)
    squares = np.hsplit(squares, w * h // (square_size * square_size))
    random.shuffle(squares)
    squares = np.reshape(squares, -1)
    squares = np.hsplit(squares, w * h // square_size)
    squares = np.vstack(squares)
    squares = np.vsplit(squares, w // square_size)
    squares = np.hstack(squares)
    squares = np.reshape(squares, input_layer.shape)
    return squares


class AnimatedHistogram:
    band_max = 0

    def __init__(self, im_db, gr_db):
        bands_count = len(im_db[0][1])
        AnimatedHistogram.band_max = 0
        left = np.arange(bands_count)
        right = left + 1
        bottom = np.zeros(bands_count)
        top = bottom

        verts_count = bands_count * (1 + 3 + 1)
        codes = np.ones(verts_count, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY

        img_verts = np.zeros((verts_count, 2))
        img_verts[0::5, 0] = left
        img_verts[0::5, 1] = bottom
        img_verts[1::5, 0] = left
        img_verts[1::5, 1] = top
        img_verts[2::5, 0] = right
        img_verts[2::5, 1] = top
        img_verts[3::5, 0] = right
        img_verts[3::5, 1] = bottom

        grp_verts = np.zeros((verts_count, 2))
        grp_verts[0::5, 0] = left + .4
        grp_verts[0::5, 1] = bottom
        grp_verts[1::5, 0] = left + .4
        grp_verts[1::5, 1] = top
        grp_verts[2::5, 0] = right - .4
        grp_verts[2::5, 1] = top
        grp_verts[3::5, 0] = right - .4
        grp_verts[3::5, 1] = bottom

        patch_img = None

        def animate(_):
            try:
                next_pair = next(AnimatedHistogram.image_n_group_iterator)
            except Exception as e:
                AnimatedHistogram.image_n_group_iterator = chain.from_iterable(
                    [[(grp[1], im_db[im][1], grp_n, im_db[im][0]) for im in grp[0]] for grp_n, grp in enumerate(gr_db)])
                return []
            gr_top = next_pair[0]
            im_top = next_pair[1]
            AnimatedHistogram.band_max = max(AnimatedHistogram.band_max, max(gr_top), max(im_top))
            ax.set_ylim(0, AnimatedHistogram.band_max)
            t.set_position((bands_count // 10, AnimatedHistogram.band_max * .9))
            new_font_size = fig.get_size_inches() * fig.dpi // 50
            t.set_fontsize(new_font_size[1])
            new_text = "Group " + "%03d " % (next_pair[2] + 1)
            for im_path in next_pair[3]:
                new_text += "\n" + os.path.basename(im_path)
            t.set_text(new_text)
            img_verts[1::5, 1] = im_top
            img_verts[2::5, 1] = im_top
            grp_verts[1::5, 1] = gr_top
            grp_verts[2::5, 1] = gr_top
            return [patch_img, patch_grp, t]

        fig, ax = plt.subplots()
        ax.set_xlim(0, bands_count)

        t = mtext.Text(3, 2.5, 'text label', ha='left', va='bottom', axes=ax)

        patch_img = patches.PathPatch(path.Path(img_verts, codes), facecolor='darkgreen', edgecolor='darkgreen',
                                      alpha=1)
        patch_grp = patches.PathPatch(path.Path(grp_verts, codes), facecolor='royalblue', edgecolor='royalblue',
                                      alpha=1)
        ax.add_patch(patch_img)
        ax.add_patch(patch_grp)
        ax.add_artist(t)

        # ani = animation.FuncAnimation(fig, animate, interval=800, save_count=10, repeat=False, blit=True)
        animation.FuncAnimation(fig, animate, interval=800, save_count=10, repeat=False, blit=True)
        plt.show()


class CompareGUI(QMainWindow, CompareDialog.Ui_MainWindow):
    current_pair = 0
    draw_diff_mode = False
    image_l = QtGui.QImage
    image_r = QtGui.QImage
    image_d = QtGui.QImage
    image_pixmaps = [QtGui.QPixmap] * 4
    paint_sheet = None
    animation_frame = 0
    thumb_mode = -2
    suggested_deletes = dict()
    marked_pairs = dict()
    image_delete_candidates = {}
    image_delete_final = set()
    auto_selection_mode = 2
    size_comparison = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.main_window = main_window

        self.label_percent.mousePressEvent = self.toggle_diff_mode

        self.paint_sheet = PaintSheet(self)
        self.paint_sheet.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        self.preview_sheet = PreviewSheet(self, self.show_pair, self.mark_pair_visual)
        self.preview_sheet.setMaximumHeight(155)
        self.preview_sheet.setMinimumHeight(155)
        self.label_menu.mousePressEvent = self.show_main_menu

        # self.sub_menu_selection = PyQt5.QtWidgets.QMenu()

        self.main_menu = QMenu("Settings")
        self.main_menu.addAction("Move files and close", self.move_files)
        self.main_menu.addAction("Apply marks (no file moves)", self.process_pairs)
        self.main_menu.addAction("Auto mark up to this", self.mark_pairs_automatic)
        self.main_menu.addAction("Resort pairs with accurate diff", self.resort_pairs)
        # self.main_menu.addAction("Select up to this (include equals)", self.mark_pairs_automatic_max)

        self.sub_menu_selection = self.main_menu.addMenu("When files have equal size")
        self.sub_menu_selection_ag = QActionGroup(self.sub_menu_selection)
        sub_menu_names = ["Do not mark anything", "Mark to leave both",
                          "Mark left for deletion", "Mark right for deletion",
                          "Mark file with longer name for deletion", "Mark file with shorter name for deletion"]
        self.sub_menu_selection_items = []
        for item in sub_menu_names:
            item_qmenu = QAction(item, self.sub_menu_selection_ag)
            item_qmenu.setCheckable(True)
            self.sub_menu_selection.addAction(item_qmenu)
            self.sub_menu_selection_items.append(item_qmenu)
        self.sub_menu_selection_items[2].setChecked(True)
        self.sub_menu_selection_ag.triggered.connect(self.change_auto_selection_mode)

        self.main_menu.addSeparator()
        self.main_menu.addAction("Close", self.close)

        self.mode_buttons.buttonClicked.connect(self.reset_thumbs)
        self.push_colored.clicked.connect(self.reset_thumbs)

        self.frame_img.layout().addWidget(self.paint_sheet)
        self.frame_img.layout().addWidget(self.preview_sheet)
        self.thumb_timer = QTimer()
        self.thumb_timer.timeout.connect(self.load_new_thumb)
        self.thumb_timer.setInterval(150)
        self.show_pair()
        self.animate_timer = QTimer()
        self.animate_timer.timeout.connect(self.redraw_animation)
        self.animate_timer.setInterval(800)

        # self.label_name_r.mousePressEvent.connect(self.redraw_animation)
        # self.label_name_r.mousePressEvent.connect(self.next_image)
        self.label_name_r.mousePressEvent = self.right_label_click
        self.label_name_l.mousePressEvent = self.left_label_click

    def resort_pairs(self):
        global progress_value
        global progress_max
        global status_text
        global distance_db
        global main_window
        status_text = "(4/4) Resorting pairs..."
        progress_value = 0
        progress_max = len(distance_db)
        thumb_cache = {}

        QApplication.processEvents()

        def image_thumb_array(im_id):
            img, img_norm = thumb_cache.get(im_id, (None, None))
            if img is None:
                img = load_image(image_DB[im_id][0][0]).convert("RGB").resize((256, 256), resample=Image.BILINEAR)
                if conf.compare_by_angles:
                    img = np.asarray(img).reshape(-1)
                    img_norm = la_norm(img)
                thumb_cache[im_id] = img, img_norm
            return img, img_norm

        for i, pair in enumerate(distance_db):
            id_l = pair['im_1']
            id_r = pair['im_2']

            image_l, image_l_norm = image_thumb_array(id_l)
            image_r, image_r_norm = image_thumb_array(id_r)

            if conf.compare_by_angles:
                distance = 1 - np.dot(image_l[np.newaxis].astype(np.int64), image_r) / image_l_norm / image_r_norm
            else:
                image_plus = ImageChops.subtract(image_l, image_r)
                image_minus = ImageChops.subtract(image_r, image_l)
                image_d = ImageChops.add(image_plus, image_minus)
                image_bands = image_d.split()
                image_sum_bw = ImageChops.add(ImageChops.add(image_bands[0], image_bands[1], 1, 0), image_bands[2], 1, 0)
                distance = la_norm(np.asarray(image_sum_bw)).astype(int)

            pair['dist'] = distance
            progress_value = i
            main_window.redraw_dialog()
            QApplication.processEvents()

        distance_db = np.sort(distance_db, order='dist')

        progress_value = 0
        main_window.redraw_dialog()
        status_text = "Resort complete. Showing image pairs."
        QApplication.processEvents()
        # conf.compare_by_angles = False
        self.image_delete_candidates.clear()
        self.marked_pairs.clear()
        self.reset_thumbs(True)
        self.update_all_delete_marks()
        self.current_pair = 0
        self.preview_sheet.central_pixmap = 0
        self.show_pair()
        self.update_central_lbl()

    def right_label_click(self, a0: QtGui.QMouseEvent):
        if a0.button() == 1:
            self.show_pair(1)
        elif a0.button() == 2:
            self.mark_pair_visual(0, -1, 1)

    def left_label_click(self, a0: QtGui.QMouseEvent):
        if a0.button() == 1:
            self.show_pair(-1)
        elif a0.button() == 2:
            self.mark_pair_visual(0, 1, -1)

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        if not a0.spontaneous():
            self.marked_pairs.clear()
            self.image_delete_candidates.clear()
            self.image_delete_final.clear()
            self.reset_thumbs(True)
            self.update_central_lbl()


    # def show(self) -> None:

    def change_auto_selection_mode(self):
        selected_item = self.sub_menu_selection_items.index(self.sub_menu_selection_ag.checkedAction())
        self.auto_selection_mode = selected_item

    def mark_pairs_automatic(self):
        for work_pair in range(self.current_pair + 1):
            id_l = distance_db[work_pair]['im_1']
            id_r = distance_db[work_pair]['im_2']
            if self.image_delete_candidates.get(id_l, False) or self.image_delete_candidates.get(id_r, False):
                continue
            im_l_size = image_DB[id_l][5]
            im_r_size = image_DB[id_r][5]
            size_difference = im_l_size[0] * im_l_size[1] / (im_r_size[0] * im_r_size[1])
            if size_difference > 1:
                left_order, right_order = 1, -1
            elif size_difference < 1:
                left_order, right_order = -1, 1
            elif self.auto_selection_mode == 0:
                continue
            elif self.auto_selection_mode == 1:
                left_order, right_order = (1, 1)
            elif self.auto_selection_mode == 2:
                left_order, right_order = (-1, 1)
            elif self.auto_selection_mode == 3:
                left_order, right_order = (1, -1)
            elif self.auto_selection_mode == 4 or self.auto_selection_mode == 5:
                im_l_name = image_DB[id_l][0][0]
                im_r_name = image_DB[id_r][0][0]
                im_l_len = len(os.path.basename(im_l_name))
                im_r_len = len(os.path.basename(im_r_name))
                if im_l_len == im_r_len:
                    continue
                delete_left = (self.auto_selection_mode == 5) != (im_l_len > im_r_len)
                left_order, right_order = (-1, 1) if delete_left else (1, -1)
            else:
                print("Error at selecting")
                continue

            if not self.marked_pairs.get(work_pair, None):
                self.mark_pair(work_pair, left_order, right_order)

        self.update_all_delete_marks()
        self.preview_sheet.update()
        self.update_central_lbl()
        self.preview_sheet.pos = None

    def mark_pair_visual(self, shift, left_order, right_order):
        work_pair = (self.current_pair + shift) % len(distance_db)

        self.mark_pair(work_pair, left_order, right_order)

        self.update_all_delete_marks()
        self.preview_sheet.update()
        self.update_central_lbl()

    def mark_pair(self, work_pair, left_order, right_order):
        print(work_pair, left_order, right_order)
        id_l = distance_db[work_pair]['im_1']
        id_r = distance_db[work_pair]['im_2']
        if (left_order, right_order) == (0, 0):
            self.marked_pairs.pop(work_pair, None)
            for img_index in [id_l, id_r]:
                old_set = self.image_delete_candidates.get(img_index, None)
                if old_set is not None:
                    old_set.discard(work_pair)
                    if len(old_set) == 0:
                        self.image_delete_candidates.pop(img_index, None)

        else:
            self.marked_pairs[work_pair] = (left_order, right_order)
            old_set = self.image_delete_candidates.get(id_l, set())
            if left_order == -1:
                old_set.add(work_pair)
                self.image_delete_candidates[id_l] = old_set
            else:
                old_set.discard(work_pair)
                if len(old_set) == 0:
                    self.image_delete_candidates.pop(id_l, None)

            old_set = self.image_delete_candidates.get(id_r, set())
            if right_order == -1:
                old_set.add(work_pair)
                self.image_delete_candidates[id_r] = old_set
            else:
                old_set.discard(work_pair)
                if len(old_set) == 0:
                    self.image_delete_candidates.pop(id_r, None)

    def process_pairs(self):
        global distance_db
        print("process_pairs")
        pairs_to_delete = set()
        old_image_delete_count = len(self.image_delete_final)
        self.image_delete_final.update(self.image_delete_candidates.keys())
        pairs_to_delete.update(self.marked_pairs.keys())

        for i, pair in enumerate(distance_db):
            if pair[1] in self.image_delete_candidates or pair[2] in self.image_delete_candidates:
                pairs_to_delete.add(i)

        distance_db = np.delete(distance_db, list(pairs_to_delete))

        print(f"Processed {len(self.marked_pairs)} pairs, deleted {len(pairs_to_delete)} pairs")
        print(f"Added {len(self.image_delete_final) - old_image_delete_count} images to delete, totaling {len(self.image_delete_final)}")

        self.image_delete_candidates.clear()
        self.marked_pairs.clear()
        self.preview_sheet.pos = None
        self.reset_thumbs(True)
        self.update_all_delete_marks()
        self.current_pair = 0
        self.preview_sheet.central_pixmap = 0
        self.show_pair()
        self.update_central_lbl()

    def move_files(self):
        print("Files to be moved")
        self.process_pairs()

        for im_index in self.image_delete_final:
            full_name = image_DB[im_index][0][0]
            parent_folder = os.path.dirname(full_name)
            parent_start_folder = os.path.dirname(start_folder)
            own_subfolder = parent_folder[len(parent_start_folder):]
            new_delete_folder = os.path.join(parent_start_folder, "---") + own_subfolder
            if not os.path.isdir(new_delete_folder):
                try:
                    os.makedirs(new_delete_folder)
                except Exception as e:
                    print("Could not create folder", e)
            try:
                shutil.move(full_name, new_delete_folder)
                if not os.listdir(parent_folder):
                    os.rmdir(parent_folder)

            except Exception as e:
                # todo good message here
                print("Could not complete file move ", e)
                return

        list_images_to_delete = list(self.image_delete_final)
        list_images_to_delete.sort(reverse=True)
        for im_index in list_images_to_delete:
            image_DB.pop(im_index)

        global image_count
        image_count = len(image_DB)

        self.close()

    def show_main_menu(self, *args, **kwargs):
        self.main_menu.popup(self.label_menu.mapToGlobal(QtCore.QPoint(0, -self.main_menu.sizeHint().height())))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Left:
            self.show_pair(-1)
        elif e.key() == QtCore.Qt.Key_Right:
            self.show_pair(1)

    def toggle_diff_mode(self, *args, **kwargs):
        self.draw_diff_mode = not self.draw_diff_mode
        self.paint_sheet.separate_pictures = self.draw_diff_mode
        if self.draw_diff_mode:
            self.animate_timer.start()
            self.label_percent.setFrameShadow(QtWidgets.QFrame.Sunken)
        else:
            self.animate_timer.stop()
            self.paint_sheet.pixmap_l = self.image_pixmaps[0]
            self.paint_sheet.pixmap_r = self.image_pixmaps[1]
            self.label_name_l.setLineWidth(1)
            self.label_name_r.setLineWidth(1)
            self.label_percent.setFrameShadow(QtWidgets.QFrame.Raised)
            self.paint_sheet.active_frame = None
        self.update_central_lbl()
        self.paint_sheet.update()

    def reset_thumbs(self, button_id):
        if self.thumb_mode != self.mode_buttons.checkedId() or type(button_id) is bool:
            for i in range(len(self.preview_sheet.pixmaps)):
                self.preview_sheet.pixmaps[i] = None
        self.thumb_mode = self.mode_buttons.checkedId()

    def load_image_and_pixmap(self, path):
        im = load_image(path).convert("RGB")
        qimage_obj = QtGui.QImage(im.tobytes(), im.width, im.height, im.width * 3, QtGui.QImage.Format_RGB888)
        pixmap_obj = QtGui.QPixmap(qimage_obj)
        return im, pixmap_obj

    def update_central_lbl(self):
        im_dist = distance_db[self.current_pair]['dist']
        first_label_text = f" Pair {self.current_pair + 1} of {len(distance_db)} \nDifference "
        first_label_text += f"{im_dist * 100:.2f}%" if conf.compare_by_angles else f"{im_dist:.0f}"
        first_label_text += "\nDifference mode" if self.draw_diff_mode else "\nSide by side mode"
        first_label_text += f"\nMarked {len(self.marked_pairs)} pairs, {len(self.image_delete_candidates)} + {len(self.image_delete_final)} images"
        self.label_percent.setText(first_label_text)
    #
    # def size_text(self, img: Image, idx: int, other_img, label):
    #     color_template = "QLabel { \
    #                     border-width: 3px;\
    #                     border-style: solid;\
    #                     border-color: "
    #     red_color = "rgb(200, 70, 70);}"
    #     green_color = "rgb(70, 200, 70);}"
    #     gray_color = "rgb(70, 70, 70);}"
    #     size = img.width * img.height
    #     text = f"Image # {idx + 1}\nof {image_count}"
    #     text += f"\n {img.width} X {img.height}"
    #     text += f"\n {size / 1e6:.02f}Mpix"
    #     bigger = size / other_img.width / other_img.height
    #     inequality = abs(bigger - 1 / bigger)
    #
    #     if inequality < .01:
    #         text += "\n equal sizes"
    #         color_template += gray_color
    #     elif bigger > 3:
    #         text += "\n much bigger"
    #         color_template += green_color
    #     elif bigger > 1.1:
    #         text += "\n bigger"
    #         color_template += green_color
    #     elif bigger > 1:
    #         text += "\n a bit bigger"
    #         color_template += green_color
    #     elif bigger < .33:
    #         text += "\n much smaller"
    #         color_template += red_color
    #     elif bigger < .9:
    #         text += "\n smaller"
    #         color_template += red_color
    #     elif bigger < 1:
    #         text += "\n a bit smaller"
    #
    #     label.setText(text)
    #     label.setStyleSheet(color_template + red_color)

    def size_label_text(self, idx: str, label, img_size):
        color_template = "QLabel { \
                        border-width: 3px;\
                        border-style: solid;\
                        border-color: "
        red_color = "rgb(200, 70, 70);}"
        green_color = "rgb(70, 200, 70);}"
        gray_color = "rgb(70, 70, 70);}"

        im_id = distance_db[self.current_pair]['im_' + idx] + 1

        text = f"Image # {im_id}\nof {image_count}"
        text += f"\n {img_size[0]} X {img_size[1]}"
        text += f"\n {img_size[0] * img_size[1] / 1e6:.02f} Mpix\n "

        size_labels = {0: "equal sizes",
                       1: "a bit bigger",
                       2: "bigger",
                       3: "much bigger",
                       -1: "a bit smaller",
                       -2: "smaller",
                       -3: "much smaller"
                       }

        text += size_labels[self.size_comparison if idx == "1" else -self.size_comparison]
        if self.size_comparison == 0:
            color_template += gray_color
        elif (self.size_comparison > 0) != (idx == "2"):
            color_template += green_color
        else:
            color_template += red_color

        label.setText(text)
        label.setStyleSheet(color_template)

    def compare_image_sizes(self):
        img_l_size = self.image_l.size
        img_r_size = self.image_r.size

        bigger_l = img_l_size[0] * img_l_size[1] / img_r_size[0] / img_r_size[1]
        bigger_r = 1 / bigger_l
        if bigger_l == 1:
            self.size_comparison = 0
        elif bigger_l > 3:
            self.size_comparison = 3
        elif bigger_l > 1.1:
            self.size_comparison = 2
        elif bigger_l > 1:
            self.size_comparison = 1
        elif bigger_r > 3:
            self.size_comparison = -3
        elif bigger_r > 1.1:
            self.size_comparison = -2
        elif bigger_r > 1:
            self.size_comparison = -1

        self.size_label_text("1", self.label_size_l, img_l_size)
        self.size_label_text("2", self.label_size_r, img_r_size)

    def generate_diff_image(self):
        if self.image_l.height > self.image_r.height:
            image_r = self.image_r.resize(self.image_l.size, Image.BILINEAR)
            image_l = self.image_l
        elif self.image_l.height < self.image_r.height or self.image_l.width != self.image_r.width:
            image_l = self.image_l.resize(self.image_r.size, Image.BILINEAR)
            image_r = self.image_r
        else:
            image_r = self.image_r
            image_l = self.image_l

        invert_l = ImageOps.invert(image_l)
        self.image_d = Image.blend(invert_l, image_r, .5)

    def show_pair(self, shift=0):
        self.current_pair += shift
        self.current_pair %= len(distance_db)
        one_step = -1 if shift < 0 else 1
        for i in range(abs(shift)):
            self.preview_sheet.central_pixmap = (self.preview_sheet.central_pixmap + one_step) % 15
            thumb_id = (self.preview_sheet.central_pixmap + 7 * one_step) % 15
            self.preview_sheet.pixmaps[thumb_id] = None
            self.preview_sheet.delete_marks[thumb_id] = None

        id_l = distance_db[self.current_pair]['im_1']
        id_r = distance_db[self.current_pair]['im_2']
        self.label_name_l.setText(image_DB[id_l][0][0])
        self.label_name_r.setText(image_DB[id_r][0][0])
        self.image_l, self.image_pixmaps[0] = self.load_image_and_pixmap(image_DB[id_l][0][0])
        self.image_r, self.image_pixmaps[1] = self.load_image_and_pixmap(image_DB[id_r][0][0])
        # self.size_text(self.image_l, id_l, self.image_r, self.label_size_l)
        # self.size_text(self.image_r, id_r, self.image_l, self.label_size_r)
        self.compare_image_sizes()
        self.update_central_lbl()
        self.generate_diff_image()

        self.image_pixmaps[2] = pixmap_from_image(self.image_d)
        self.image_pixmaps[3] = pixmap_from_image(ImageOps.invert(self.image_d))
        self.paint_sheet.pos = None

        self.preview_sheet.update()

        if self.draw_diff_mode:
            self.redraw_animation()
        else:
            self.paint_sheet.pixmap_l = self.image_pixmaps[0]
            self.paint_sheet.pixmap_r = self.image_pixmaps[1]

            self.paint_sheet.update()
            self.preview_sheet.update()
        self.thumb_timer.start()

    def load_new_thumb(self):
        if not self.isVisible():
            self.thumb_timer.stop()
            return
        for i in range(1, 16):
            thumb_id = i // 2 if i % 2 else -i // 2
            if self.generate_diff_thumb(thumb_id):
                return

    def generate_diff_thumb(self, shift):
        thumb_id = (self.preview_sheet.central_pixmap + shift) % 15
        if self.preview_sheet.pixmaps[thumb_id]:
            return
        work_pair = (self.current_pair + shift) % len(distance_db)
        id_l = distance_db[work_pair]['im_1']
        id_r = distance_db[work_pair]['im_2']
        image_l = load_image(image_DB[id_l][0][0]).convert("RGB").resize((240, 240), Image.BILINEAR)
        image_r = load_image(image_DB[id_r][0][0]).convert("RGB").resize((240, 240), Image.BILINEAR)
        if self.thumb_mode == -3:
            image_d = Image.blend(ImageOps.invert(image_l), image_r, .5)
        else:
            image_plus = ImageChops.subtract(image_l, image_r)
            image_minus = ImageChops.subtract(image_r, image_l)
            image_d = ImageChops.add(image_plus, image_minus)

        if not self.push_colored.isChecked():
            image_bands = image_d.split()
            if self.thumb_mode == -3:
                o = -128
            else:
                o = 0
            image_sum_bw = ImageChops.add(ImageChops.add(image_bands[0], image_bands[1], 1, o), image_bands[2], 1, o)
            image_d = image_sum_bw.convert("RGB")

        if self.thumb_mode == -4:
            image_d = ImageOps.invert(image_d)

        self.preview_sheet.pixmaps[thumb_id] = pixmap_from_image(image_d)
        self.update_preview_delete_marks(work_pair, thumb_id)
        self.preview_sheet.update()
        return True

    def update_all_delete_marks(self):
        for i in range(1, 16):
            shift = i // 2 if i % 2 else -i // 2
            thumb_id = (self.preview_sheet.central_pixmap + shift) % 15
            work_pair = (self.current_pair + shift) % len(distance_db)
            self.update_preview_delete_marks(work_pair, thumb_id)

    def update_preview_delete_marks(self, work_pair, thumb_id):
        id_l = distance_db[work_pair]['im_1']
        id_r = distance_db[work_pair]['im_2']
        mark_left, mark_right = 0, 0
        mark_set = self.marked_pairs.get(work_pair, None)
        if mark_set:
            mark_left, mark_right = mark_set
        if self.image_delete_candidates.get(id_l, None) and not mark_left:
            mark_left = -2
        if self.image_delete_candidates.get(id_r, None) and not mark_right:
            mark_right = -2

        if mark_left or mark_right:
            self.preview_sheet.delete_marks[thumb_id] = mark_left, mark_right
        else:
            self.preview_sheet.delete_marks[thumb_id] = None
        # self.preview_sheet.delete_marks[thumb_id] = None

    def redraw_animation(self):
        if not self.isVisible():
            self.animate_timer.stop()
            return
        self.animation_frame = 1 - self.animation_frame
        self.paint_sheet.pixmap_l = self.image_pixmaps[3 - self.animation_frame]
        self.paint_sheet.pixmap_r = self.image_pixmaps[self.animation_frame]
        self.label_name_r.setLineWidth(1 + self.animation_frame * 3)
        self.label_name_l.setLineWidth(1 + 3 - self.animation_frame * 3)
        self.paint_sheet.active_frame = self.animation_frame
        self.paint_sheet.update()


class VisImSorterGUI(QMainWindow, VisImSorterInterface.Ui_VisImSorter):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.timer = QTimer()
        self.drag_timer = QTimer()
        self.font = PyQt5.QtGui.QFont()
        self.init_elements()

    def init_elements(self):
        global start_folder
        global main_window

        main_window = self

        self.font.setPointSize(14)

        self.timer.timeout.connect(self.redraw_dialog)
        self.timer.setInterval(50)
        self.drag_timer.timeout.connect(self.drag_timeout)
        self.drag_timer.setInterval(3000)

        self.select_folder_button.clicked.connect(self.select_folder)
        self.select_folder_button.resizeEvent = self.directory_changed
        self.select_folder_button.dragLeaveEvent = self.directory_changed
        self.select_folder_button.dragEnterEvent = self.directory_entered
        self.select_folder_button.dropEvent = self.directory_dropped
        self.directory_changed()

        self.slider_histogram_bands.valueChanged.connect(self.slider_histogram_bands_changed)
        self.slider_histogram_bands_changed()

        self.slider_image_split_steps.valueChanged.connect(self.slider_image_split_steps_changed)
        self.slider_image_split_steps_changed()

        self.combo_compare_vectors.currentIndexChanged.connect(self.slider_image_max_likeness_changed)
        self.slider_image_max_likeness.valueChanged.connect(self.slider_image_max_likeness_changed)
        self.slider_image_max_likeness_changed()

        self.slider_enforce_equal.valueChanged.connect(self.slider_equal_changed)
        self.slider_equal_changed()

        self.list_color_spaces.itemSelectionChanged.connect(self.color_spaces_reselected)
        self.color_spaces_reselected()
        self.tabWidget.currentChanged.connect(self.duplicate_tab_switched)

        self.init_color_space(self.list_color_spaces_CMYK)
        self.init_color_space(self.list_color_spaces_HSV)
        self.init_color_space(self.list_color_spaces_RGB)
        self.init_color_space(self.list_color_spaces_YCbCr)

        self.btn_stop.clicked.connect(self.stop_button_pressed)
        self.btn_start.clicked.connect(self.start_button_pressed)
        self.progressBar.setVisible(False)
        self.progressBar.valueChanged.connect(self.update)
        self.enable_elements()

        if len(sys.argv) > 1:
            if os.path.isdir(sys.argv[1]):
                start_folder = sys.argv[1]
                self.directory_changed()

    def start_button_pressed(self):
        global target_groups
        global target_group_size
        global search_subdirectories
        global selected_color_spaces
        global selected_color_subspaces
        global move_files_not_copy
        global show_histograms
        global export_histograms
        global run_index
        global status_text
        global abort_reason
        global start_folder
        global new_folder
        global create_samples_enabled
        global folder_size_min_files
        global folder_size_min_files
        global final_sort
        global group_with_similar_name
        global enable_stage2_grouping
        global folder_constraint_type
        global folder_constraint_value
        global stage1_grouping_type
        global enable_multiprocessing

        local_run_index = run_index

        folder_constraint_type = self.combo_folder_constraints.currentIndex()
        folder_constraint_value = self.spin_num_constraint.value()
        search_subdirectories = self.check_subdirs_box.isChecked()
        move_files_not_copy = self.radio_move.isChecked()
        show_histograms = self.check_show_histograms.isChecked()
        export_histograms = self.check_export_histograms.isChecked()
        create_samples_enabled = self.check_create_samples.isChecked()
        enable_stage2_grouping = self.check_stage2_grouping.isChecked()
        stage1_grouping_type = self.combo_stage1_grouping.currentIndex()
        enable_multiprocessing = self.check_multiprocessing.isChecked()
        conf.search_duplicates = not self.tab_unduplicate.isHidden()
        conf.compare_by_angles = self.combo_compare_vectors.currentIndex() == 1

        final_sort = self.combo_final_sort.currentIndex()
        if self.check_equal_name.isChecked():
            group_with_similar_name = self.spin_equal_first_symbols.value()
        else:
            group_with_similar_name = 0

        selected_color_spaces = []
        selected_color_subspaces = []

        for item in self.list_color_spaces.selectedItems():
            selected_color_spaces.append(str(item.text()))
            selected_color_subspaces += self.get_sub_color_space(item.text())

        if len(selected_color_subspaces) == 0:
            QMessageBox.warning(None, "VisImSorter: Error",
                                "Please select at least one color space and at least one sub-color space")
            return

        self.disable_elements()
        run_sequence()
        if local_run_index != run_index:
            status_text = abort_reason
            self.btn_stop.setText(status_text)
            self.btn_stop.setStyleSheet("background-color: rgb(250,150,150)")
        else:
            self.btn_stop.setText(status_text)
            self.btn_stop.setStyleSheet("background-color: rgb(150,250,150)")
            run_index += 1
        self.btn_stop.setFont(self.font)
        self.progressBar.setVisible(False)

    def stop_button_pressed(self):
        global run_index
        global abort_reason
        if self.progressBar.isVisible():
            run_index += 1
            abort_reason = "Process aborted by user."
        else:
            self.directory_changed(True)
            self.enable_elements()

    def get_sub_color_space(self, color_space_name):
        w = self.group_colorspaces_all.findChild(PyQt5.QtWidgets.QListWidget, "list_color_spaces_" + color_space_name)
        subspaces = []
        for item in w.selectedItems():
            subspaces.append(item.text())
        return subspaces

    def color_spaces_reselected(self):
        for index in range(self.horizontalLayout_7.count()):
            short_name = self.horizontalLayout_7.itemAt(index).widget().objectName()[18:]
            items = self.list_color_spaces.findItems(short_name, PyQt5.QtCore.Qt.MatchExactly)
            if len(items) > 0:
                enabled = items[0].isSelected()
                self.horizontalLayout_7.itemAt(index).widget().setEnabled(enabled)

    def init_color_space(self, color_space_list):
        for i in range(color_space_list.count()):
            item = color_space_list.item(i)
            if item.text() != "K":
                item.setSelected(True)

    def enable_elements(self):
        global progress_value
        self.group_input.setEnabled(True)
        self.group_analyze.setEnabled(True)
        self.tabWidget.setEnabled(True)
        self.group_move.setEnabled(True)
        self.group_final.setEnabled(True)
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)
        self.btn_stop.setStyleSheet("background-color: rgb(200,150,150)")
        self.btn_stop.setFont(self.font)
        progress_value = 0
        self.redraw_dialog()
        self.timer.stop()
        self.duplicate_tab_switched()

    def disable_elements(self):
        self.group_input.setEnabled(False)
        self.group_analyze.setEnabled(False)
        self.tabWidget.setEnabled(False)
        self.group_move.setEnabled(False)
        self.group_final.setEnabled(False)
        self.btn_start.setVisible(False)
        self.btn_stop.setText("Stop")
        self.btn_stop.setVisible(True)
        self.progressBar.setVisible(True)
        self.duplicate_tab_switched()
        self.timer.start()

    def duplicate_tab_switched(self):
        self.group_move.setEnabled(self.tab_grouping.isVisible())
        self.group_final.setEnabled(self.tab_grouping.isVisible())
        self.groupBox_5.setEnabled(self.tab_grouping.isVisible())

    def slider_histogram_bands_changed(self):
        global hg_bands_count
        hg_bands_count = self.slider_histogram_bands.value()
        self.lbl_histogram_bands.setText(str(hg_bands_count))

    def slider_image_split_steps_changed(self):
        raw_splits = self.slider_image_split_steps.value()
        conf.image_divisions = 2 ** raw_splits
        self.lbl_image_split_steps.setText(str(conf.image_divisions ** 2))

    def slider_equal_changed(self):
        global enforce_equal_folders
        enforce_equal_folders = self.slider_enforce_equal.value() / 100
        enforce_equal_folders *= enforce_equal_folders
        self.lbl_slider_enforce_equal.setText("%d" % (self.slider_enforce_equal.value() / 3) + "%")

    def slider_image_max_likeness_changed(self):
        raw_slider = self.slider_image_max_likeness.value()
        if self.combo_compare_vectors.currentIndex():
            self.lbl_image_max_likeness.setText(f"{raw_slider}%")
            conf.max_likeness = raw_slider / 100
        else:
            raw_slider *= 655.36
            conf.max_likeness = raw_slider
            self.lbl_image_max_likeness.setText(f"{int(raw_slider)}")

        if raw_slider == 100:
            self.lbl_image_max_likeness.setText("off")

    def drag_timeout(self):
        self.directory_changed()
        self.drag_timer.stop()

    def directory_entered(self, input_object):
        if input_object.mimeData().hasUrls():
            url_text = unquote(input_object.mimeData().text())
            firt_path = url_text[8:]
            firt_path_start = url_text[:8]

            if firt_path_start == "file:///":
                if os.path.isdir(firt_path):
                    self.select_folder_button.setText("Drop folder here")
                    input_object.accept()
                    return
        self.select_folder_button.setText("Only directory accepted")
        input_object.ignore()
        self.drag_timer.start()

    def directory_dropped(self, in_dir):
        global start_folder
        url_text = unquote(in_dir.mimeData().text())
        firt_path = url_text[8:]
        firt_path_start = url_text[:8]
        if firt_path_start == "file:///":
            if os.path.isdir(firt_path):
                start_folder = firt_path
                self.directory_changed()

    def directory_changed(self, suppress_text=False):
        global start_folder
        global status_text

        if os.path.exists(start_folder):
            start_folder = os.path.normpath(start_folder)
            if not suppress_text:
                status_text = "Ready to start"

            start_folder_parts = [a + "\\" for a in start_folder.split("\\")]
            start_folder_parts2 = [(a + " ") if a[-1] != "\\" else a for b in start_folder_parts for a in b.split(" ")]
            line_length = 0
            lines_count = max(round(len(start_folder) / 35), 1)
            line_max = max(len(start_folder) // lines_count, 20)
            button_text = ""

            for part_path in start_folder_parts2:
                extended_line_length = line_length + len(part_path)
                if extended_line_length > line_max:
                    button_text += "\n" + part_path
                    line_length = len(part_path)
                else:
                    button_text += part_path
                    line_length += len(part_path)

            # button_text = start_folder
            # if len(start_folder) > line_length:
            #     lines_count = len(start_folder) // line_length
            #     line_max = len(start_folder) // lines_count
            #     path_list = re.split("\\\\", start_folder)
            #     button_text = ""
            #     line_length = 0
            #     for part_path in path_list:
            #         extended_line_length = line_length + len(part_path) + 1
            #         if (extended_line_length > line_max) and (extended_line_length > 5):
            #             button_text += "\n" + part_path + "\\"
            #             line_length = len(part_path + "\\")
            #         else:
            #             button_text += part_path + "\\"
            #             line_length += len(part_path + "\\")

            self.select_folder_button.setText(button_text)
            self.btn_start.setEnabled(True)
        else:
            if not suppress_text:
                status_text = "Please select folder"
            self.select_folder_button.setText("Select folder")
            self.btn_start.setEnabled(False)

    def select_folder(self):
        global start_folder
        start_folder = QFileDialog.getExistingDirectory(self, "Choose directory", start_folder or "Y:/")
        self.directory_changed()

    def redraw_dialog(self):
        global progress_value
        global status_text
        global progress_max
        if self.progressBar.maximum() != progress_max:
            self.progressBar.setRange(0, progress_max)
        self.progressBar.setValue(int(progress_value))
        self.statusbar.showMessage(status_text)
        self.update()
        QApplication.processEvents()


def main():
    app = QApplication(sys.argv)
    sorter_window = VisImSorterGUI()
    sorter_window.show()
    sys.exit(app.exec_())


main_window = VisImSorterGUI

if __name__ == '__main__':
    main()
