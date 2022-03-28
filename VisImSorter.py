import fnmatch
import glob
import itertools
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
# import gc
from time import sleep, localtime, time, gmtime, strftime
from timeit import default_timer as timer

import matplotlib.animation as animation
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import numpy.ma as ma
import pandas as pd
from urllib.parse import unquote
import psutil
from PIL import Image, ImageOps, ImageChops, ImageStat
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QMainWindow, QMenu, QAction, QActionGroup
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy import ndimage
from PaintSheet import PaintSheet, ThumbSheet
from ImageLoader import load_image, generate_image_vector, all_file_types
# from collections.abc import Mapping, Container

import CompareDialog
import VisImSorterInterface

pool = mp.Pool

image_paths_db = pd.Series
image_DB = pd.DataFrame
distance_db = pd.DataFrame
image_count = 0

group_db = []
groups_count = 0

new_folder = ""
new_vector_folder = ""
enforce_equal_folders = 0.
compare_file_percent = 10
move_files_not_copy = False
show_histograms = False
export_histograms = False
create_samples_enabled = False
gaussian_filter1d = ndimage.gaussian_filter1d
median_filter = ndimage.median_filter
gaussian_filter = ndimage.gaussian_filter
la_norm = np.linalg.norm
final_sort = 0
function_queue = []


class Config:
    search_duplicates = False
    image_divisions = 1
    compare_by_angles = False
    max_likeness = 100
    search_subdirectories = False
    hg_bands_count = 0

    enable_stage2_grouping = False
    enable_multiprocessing = True
    stage1_grouping_type = 0
    folder_constraint_value = 0
    folder_constraint_type = 0
    group_with_similar_name = 0

    target_groups = 0
    target_group_size = 0
    folder_size_min_files = 0
    selected_color_bands = dict()
    compare_resort = False

    max_pairs = 0
    start_folder = ""


class Data:
    status_text = ""
    progress_max = 100
    progress_value = 0
    last_iteration_progress_value = 0
    run_index = 0
    working_bands = set()
    raw_time_left = 0
    smooth_end_time = [0.] * 50
    smooth_speed = None
    steps_buffer = [(0., 0)] * 50
    smooth_time_iterations = 0
    event_start_time = 0
    passed_percent = 0

    def set_smooth_time(self, iterations: int):
        self.smooth_time_iterations = iterations
        self.smooth_end_time = [0.] * 50
        self.event_start_time = time()
        self.passed_percent = 0
        data.last_iteration_progress_value = 0
        data.smooth_speed = None
        data.steps_buffer = [(0., 0)] * 50


conf = Config()
data = Data()


def mix(a, b, amount=.5):
    return a * (1 - amount) + b * amount


def pixmap_from_image(im):
    qimage_obj = QtGui.QImage(im.tobytes(), im.width, im.height, im.width * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap(qimage_obj)


def set_status(status=None, progress_value=None, progress_max=None, abort=False, muted=False):
    if status is not None:
        data.status_text = status
        if not muted:
            print(status)
    if progress_max is not None:
        data.progress_max = progress_max
    if progress_value is not None:
        data.progress_value = progress_value
    if abort:
        data.run_index += 1

#
# def deep_getsizeof(o, ids=None):
#     """Find the memory footprint of a Python object
#
#     This is a recursive function that drills down a Python object graph
#     like a dictionary holding nested dictionaries with lists of lists
#     and tuples and sets.
#
#     The sys.getsizeof function does a shallow size of only. It counts each
#     object inside a container as pointer only regardless of how big it
#     really is.
#
#     :param o: the object
#     :param ids:
#     :return:
#     """
#     if ids == None:
#         ids = set()
#     d = deep_getsizeof
#     if id(o) in ids:
#         return 0
#
#     r = sys.getsizeof(o)
#     ids.add(id(o))
#
#     if isinstance(o, str): # or isinstance(0, unicode):
#         return r
#
#     if isinstance(o, Mapping):
#         return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
#
#     if isinstance(o, Container):
#         return r + sum(d(x, ids) for x in o)
#
#     return r


def run_sequence():
    global function_queue
    local_run_index = data.run_index

    if conf.enable_multiprocessing:
        function_queue.append(init_pool)

    function_queue.append(scan_for_images)
    function_queue.append(collect_image_vectors)

    if conf.search_duplicates:
        function_queue.append(close_pool)
        # function_queue.append(create_distance_db)
        function_queue.append(create_distance_db_multi)
        function_queue.append(show_compare_images_gui)
    else:
        if conf.stage1_grouping_type == 0:
            function_queue.append(create_simple_groups)
        elif conf.stage1_grouping_type == 1:
            function_queue.append(create_groups_by_similarity)
        elif conf.stage1_grouping_type == 2:
            function_queue.append(create_groups_v4)

        if conf.enable_stage2_grouping:
            function_queue.append(stage2_regroup)

        function_queue.append(final_group_sort)
        function_queue.append(choose_destination_folder)
        function_queue.append(move_files)

        if export_histograms:
            function_queue.append(export_image_vectors)

        if create_samples_enabled:
            function_queue.append(create_samples)

    if conf.enable_multiprocessing:
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
        if local_run_index != data.run_index:
            if conf.enable_multiprocessing:
                close_pool()
            return

    finish_time = timer()

    if show_histograms:
        AnimatedHistogram(image_DB, group_db)

    main_win.status_text = "Finished. Elapsed {0}:{1:02d} minutes.".format(*divmod(int(finish_time - start_time), 60))
    print(main_win.status_text)
    main_win.progress_max = 100
    main_win.progress_value = 100


def close_pool():
    global pool
    pool.terminate()
    pool.close()


def init_pool():
    global pool
    pool = mp.Pool()


def scan_for_images():
    global image_paths_db
    set_status("Scanning images...")
    QApplication.processEvents()
    image_paths = []
    all_files = glob.glob(conf.start_folder + "/**/*", recursive=conf.search_subdirectories)

    for file_extention in all_file_types:
        image_paths.extend(fnmatch.filter(all_files, "*." + file_extention))

    if not len(image_paths):
        set_status("No images found.", abort=True)
        return
    image_paths = pd.Series(image_paths).apply(os.path.normpath).rename("image_paths")
    # image_paths = pd.Series([os.path.normpath(im) for im in image_paths], name="image_paths")
    image_names = image_paths.apply(os.path.basename).rename("image_names")
    image_paths_db = pd.DataFrame((image_paths, image_names)).T

    if conf.group_with_similar_name > 0:
        image_name_groups = image_names.apply(lambda x: x[:conf.group_with_similar_name])
        image_paths_db["image_name_groups"] = image_name_groups


def collect_image_vectors():
    global image_DB
    global image_count
    global pool

    local_run_index = data.run_index
    image_count = len(image_paths_db)
    data.working_bands.clear()

    for current_color_space, color_subspaces in conf.selected_color_bands.items():
        for band in color_subspaces:
            band_full_name = current_color_space + "_" + band
            image_paths_db[band_full_name] = None
            image_paths_db[band_full_name + "_size"] = None
            data.working_bands.add(band_full_name)

    set_status(f"(1/4) Generating image histograms...", progress_max=image_count, progress_value=0)
    image_DB = []
    results = []
    links_per_task = 32
    path_db_chunks = [image_paths_db[i:i + links_per_task] for i in range(0, image_count, links_per_task)]

    for path_db_chunk in path_db_chunks:
        task_parameters = (path_db_chunk, conf.selected_color_bands,
                           conf.hg_bands_count, conf.image_divisions, not conf.search_duplicates)
        result = pool.apply_async(generate_image_vector, task_parameters, callback=image_vector_callback)
        results.append(result)

    for res in results:
        while not res.ready():
            QApplication.processEvents()
            sleep(.05)
            if local_run_index != data.run_index:
                return
        res.wait()

    image_DB = pd.concat(image_DB)
    image_count = len(image_DB)
    if image_count == 0:
        set_status("Failed to create image database.", abort=True)

    if not conf.search_duplicates:
        image_DB.sort_values("max_hue")
        if conf.folder_constraint_type:
            conf.target_groups = int(conf.folder_constraint_value)
            conf.target_group_size = int(round(image_count / conf.target_groups))
        else:
            conf.target_group_size = int(conf.folder_constraint_value)
            conf.target_groups = int(round(image_count / conf.target_group_size))

    image_DB.reset_index(drop=True, inplace=True)


def image_vector_callback(vector_pack):
    if vector_pack is not None:
        image_DB.append(vector_pack)
    set_status(f"\r(1/4) Generating image histograms ({data.progress_value:d} of "
               f"{image_count:d})", data.progress_value + len(vector_pack), muted=True)


# def create_distance_db():
#     global distance_db
#     local_run_index = data.run_index

#     status = "(angles)..." if conf.compare_by_angles else "(distances)..."
#     set_status("(2/4) Comparing image " + status)
#     data.set_smooth_time(20)
#     triangle_distance_db_list = []
#     max_likeness = conf.max_likeness * 1e-2 if conf.compare_by_angles else 300

#     work_indexes = zip(range(1, image_count // 2 + 2), range(image_count - 1, image_count // 2 - 1, -1))
#     work_indexes = list(dict.fromkeys([s for t in work_indexes for s in t]))
#     im_vectors = {}
#     for color_band in data.working_bands:
#         im_vectors[color_band] = np.stack(image_DB[color_band]).astype(np.int32)

#     for i, im_index in enumerate(work_indexes):
#         diff_db = pd.DataFrame({'im_2': range(im_index)})
#         diff_db['im_1'] = im_index
#         diff_db['mean_diffs'] = (image_DB.means[:im_index] - image_DB.means[im_index]).abs()
#         if conf.max_likeness:
#             diff_db = diff_db[diff_db.mean_diffs < conf.max_likeness * 1.35].copy()
#         if len(diff_db) == 0:
#             continue
#         for color_band in data.working_bands:
#             im_vector_sizes = image_DB[color_band + "_size"]
#             if conf.compare_by_angles:
#                 distances = 1 - np.dot(im_vectors[color_band][diff_db.im_2], im_vectors[color_band][im_index]) \
#                             / im_vector_sizes[diff_db.im_2] / im_vector_sizes[im_index]
#             else:
#                 distances = la_norm((im_vectors[color_band][diff_db.im_2] - im_vectors[color_band][im_index]),
#                                     axis=1).astype(int)
#             diff_db['dist_' + color_band] = distances
#         only_dist_db = diff_db.iloc[:, 3:]
#         diff_db["dist"] = only_dist_db.mean(axis=1)
#         diff_db["size_compare"] = image_DB.megapixels[:im_index] / image_DB.megapixels[im_index]
#         if max_likeness:
#             diff_db = diff_db[diff_db["dist"] < max_likeness]

#         triangle_distance_db_list.append(diff_db)
#         data.passed_percent = (i + 1) / len(work_indexes)
#         text_to_go = f"(2/4) Comparing image {i + 1:d} of {len(work_indexes):d}."
#         set_status(text_to_go, progress_value=i, muted=True)
#         QApplication.processEvents()
#         if local_run_index != data.run_index:
#             return

#     data.set_smooth_time(0)
#     set_status("(3/4) Final resorting ...")
#     QApplication.processEvents()
#     distance_db = pd.concat(triangle_distance_db_list)
#     distance_db.sort_values("dist", inplace=True)
#     distance_db.reset_index(drop=True, inplace=True)
#     if conf.max_pairs:
#         distance_db = distance_db[:conf.max_pairs].copy()

#     distance_db["size_compare_category"] = 0
#     distance_db["crops"] = None
#     distance_db["crop_compare_category"] = 0
#     distance_db["del_im_1"] = 0
#     distance_db["del_im_2"] = 0
#     categories = {1: 1, 2: 1.1, 3: 3}
#     for category, threshold in categories.items():
#         distance_db.loc[distance_db["size_compare"] > threshold, "size_compare_category"] = -category
#         distance_db.loc[1 / distance_db["size_compare"] > threshold, "size_compare_category"] = category


def create_distance_db_multi():
    global distance_db
    local_run_index = data.run_index
    distance_db = None
    triangle_distance_db_list = []
    new_lines_count = 0
    max_likeness = 500000

    def compact_distance_db():
        nonlocal max_likeness, new_lines_count
        global distance_db
        if distance_db is not None:
            triangle_distance_db_list.append(distance_db)
        distance_db = pd.concat(triangle_distance_db_list)
        distance_db.sort_values("dist", inplace=True)
        triangle_distance_db_list.clear()
        new_lines_count = 0
        if distance_db.shape[0] > conf.max_pairs:
            distance_db = distance_db[:conf.max_pairs].copy()
            max_likeness = min(max_likeness, distance_db.dist.iloc[-1])

    status = "(angles)..." if conf.compare_by_angles else "(distances)..."
    set_status("(2/4) Comparing image " + status)
    data.set_smooth_time(20)

    work_indexes = zip(range(1, image_count // 2 + 2), range(image_count - 1, image_count // 2 - 1, -1))
    work_indexes = list(dict.fromkeys([s for t in work_indexes for s in t]))
    im_vectors = {}
    for color_band in data.working_bands:
        im_vectors[color_band] = np.stack(image_DB[color_band]).astype(np.int32)

    for i, im_index in enumerate(work_indexes):
        mean_diffs = (image_DB.means[:im_index] - image_DB.means[im_index]).abs()
        working_pairs = mean_diffs < max_likeness
        diff_list = {}
        if not working_pairs.any():
            continue
        for color_band in data.working_bands:
            color_band_s = color_band + "_size"
            if conf.compare_by_angles:
                distances = (1 - np.dot(im_vectors[color_band][:im_index][working_pairs], im_vectors[color_band][im_index]) \
                            / image_DB[color_band_s][:im_index][working_pairs] / image_DB[color_band_s][im_index]) * 45000
                # distances = 1 - np.dot(im_vectors[color_band][diff_db.index], im_vectors[color_band][im_index]) \
                #             / im_vector_sizes[diff_db.index] / im_vector_sizes[im_index]
            else:
                distances = la_norm((im_vectors[color_band][:im_index][working_pairs] - im_vectors[color_band][im_index]),
                                    axis=1).astype(int)
                # distances = la_norm((im_vectors[color_band][diff_db.im_2] - im_vectors[color_band][im_index]),
                #                     axis=1).astype(int)
            diff_list['dist_' + color_band] = distances
        diff_db = pd.DataFrame(diff_list, index=image_DB[:im_index][working_pairs].index)
        # diff_db = pd.DataFrame(diff_list)
        diff_db["dist"] = diff_db.mean(axis=1)
        diff_db["size_compare"] = image_DB.megapixels[:im_index] / image_DB.megapixels[im_index]
        diff_db = diff_db[diff_db["dist"] < max_likeness]

        diff_db = diff_db.set_index(pd.MultiIndex.from_product([diff_db.index, [im_index]], names=["im_2", "im_1"]))
        triangle_distance_db_list.append(diff_db)
        if conf.max_pairs:
            new_lines_count += len(diff_db)
            if new_lines_count > conf.max_pairs:
                compact_distance_db()
        data.passed_percent = (i + 1) / len(work_indexes)
        text_to_go = f"(2/4) Comparing image {i + 1:d} of {len(work_indexes):d}."
        set_status(text_to_go, progress_value=i, muted=True)
        QApplication.processEvents()
        if local_run_index != data.run_index:
            return

    data.set_smooth_time(0)
    set_status("(3/4) Final resorting ...")
    QApplication.processEvents()
    compact_distance_db()

    distance_db.reset_index(inplace=True)

    distance_db["size_compare_category"] = 0
    distance_db["crops"] = None
    distance_db["crop_compare_category"] = 0
    distance_db["del_im_1"] = 0
    distance_db["del_im_2"] = 0
    categories = {1: 1, 2: 1.1, 3: 3}
    for category, threshold in categories.items():
        distance_db.loc[distance_db["size_compare"] > threshold, "size_compare_category"] = -category
        distance_db.loc[1 / distance_db["size_compare"] > threshold, "size_compare_category"] = category


def show_compare_images_gui():
    global compare_wnd
    if len(distance_db) == 0:
        set_status("No similar enough images found", abort=True)
        return
    local_run_index = data.run_index

    compare_wnd = CompareGUI()
    if conf.compare_resort:
        compare_wnd.resort_pairs()
        if local_run_index != data.run_index:
            return

    set_status("Preparation complete.", 0)
    QApplication.processEvents()
    compare_wnd_toggle()


def compare_wnd_toggle():
    if compare_wnd == CompareGUI:
        return
    if main_win.btn_show_compare_wnd.isChecked():
        compare_wnd.show()
    else:
        compare_wnd.hide()


def create_simple_groups():
    global group_db
    global groups_count

    group_db = []
    group_im_indexes = np.array_split(range(image_count), conf.target_groups)
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
    # global target_groups
    global groups_count
    # global run_index
    global pool
    global group_db
    global image_DB
    global distance_db
    global progress_max
    global progress_value

    local_run_index = data.run_index
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
    invite_count = np.ones(image_count) * conf.target_group_size
    image_index_list = np.arange(image_count)
    image_belongings = image_index_list
    image_belongings_old = image_belongings
    image_belongings_final = image_belongings
    mutual_ratings = (np.tanh(eef01 * (1 - relations_db['rank'] / conf.target_group_size)) + 1) / eef01_divisor
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
            if actual_groups > conf.target_groups:
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
    # global target_groups
    global groups_count
    # global run_index
    global pool
    global group_db
    global distance_db

    local_run_index = data.run_index
    step = 0

    group_db = [[[counter]] + image_record[1:4] for counter, image_record in enumerate(image_DB)]
    groups_count = len(group_db)

    feedback_queue = mp.Manager().Queue()
    while groups_count > conf.target_groups:
        step += 1
        status_text = "Joining groups, pass %d. Now joined %d of %d . To join %d groups"
        status_text %= (
            step, image_count - groups_count, image_count - conf.target_groups, groups_count - conf.target_groups)

        fill_distance_db(feedback_queue, step)
        if local_run_index != data.run_index:
            return

        merge_group_batches()
        if local_run_index != data.run_index:
            return


def fill_distance_db(feedback_queue, step):
    global group_db
    global distance_db
    global image_count
    # global target_groups
    # global target_group_size
    global progress_max
    global progress_value
    global groups_count
    # global run_index
    global enforce_equal_folders
    global pool

    local_run_index = data.run_index
    progress_max = groups_count

    distance_db = []
    results = []
    batches = min(groups_count // 200 + 1, mp.cpu_count())
    # batches = 1
    progress_value = 0

    compare_file_limit = int(compare_file_percent * (image_count + 2000) / 100)

    entries_limit = groups_count // 4 + 1
    entries_limit = groups_count - conf.target_groups

    feedback_list = [0] * batches
    task_running = False
    if batches == 1:
        distance_db += fill_distance_db_sub(group_db, 1, 0, enforce_equal_folders, compare_file_limit,
                                            conf.target_group_size, entries_limit)
    else:
        for batch in range(batches):
            task_running = True
            args = [group_db, batches, batch, enforce_equal_folders, compare_file_limit, conf.target_group_size,
                    entries_limit, feedback_queue]
            result = pool.apply_async(fill_distance_db_sub, args=args, callback=fill_distance_db_callback)
            results.append(result)
            QApplication.processEvents()
            if local_run_index != data.run_index:
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
            second_range = itertools.chain(range(index_1 + 1, groups_count_l), range(0, rest_of_search_window))
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
    # global target_groups
    global groups_count
    # global run_index
    global image_DB

    local_run_index = data.run_index
    removed_count = 0

    for g1, g2 in distance_db:
        if groups_count - removed_count <= conf.target_groups:
            break
        new_group_image_list = group_db[g1][0] + group_db[g2][0]
        new_group_vector = np.mean(np.array([image_DB[i][1] for i in new_group_image_list]), axis=0)
        new_group_hue = np.argmax(np.sum(np.array([image_DB[i][4] for i in new_group_image_list]), axis=0))
        new_group_image_count = sum([image_DB[i][3] for i in new_group_image_list])
        group_db.append([new_group_image_list, new_group_vector, new_group_hue, new_group_image_count])
        group_db[g1][3] = 0
        group_db[g2][3] = 0
        removed_count += 1
        if local_run_index != data.run_index:
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
    # global target_group_size
    global enforce_equal_folders
    # global run_index
    global status_text

    local_run_index = data.run_index

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

    task_queue = mp.Queue()
    done_queue = mp.Queue()

    for i in range(parallel_processes):
        mp.Process(target=stage2_sort_worker, args=(task_queue, done_queue, image_DB, conf.target_group_size,
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

        if local_run_index != data.run_index:
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
    # global target_group_size
    global enforce_equal_folders

    im_index = task[0]
    g1 = task[1]
    g2 = task[2]

    if im_index not in group_db[g1][0]:
        return False

    distance_to_g1 = la_norm(group_db[g1][1].astype(np.int32) - image_DB[im_index][1].astype(np.int32))
    distance_to_g1 *= (group_db[g1][3] / conf.target_group_size) ** enforce_equal_folders
    distance_to_new_g2 = la_norm(group_db[g2][1].astype(np.int32) - image_DB[im_index][1].astype(np.int32))
    distance_to_new_g2 *= ((group_db[g2][3] + image_DB[im_index][3]) / conf.target_group_size) ** enforce_equal_folders
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
    base_folder = conf.start_folder
    destination_folder_index = -1
    if re.match(".*_sorted_\d\d\d$", conf.start_folder):
        base_folder = conf.start_folder[:-11]
    elif re.match(".*_sorted$", conf.start_folder):
        base_folder = conf.start_folder[:-7]
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
    # global run_index
    global new_folder
    # global start_folder
    # global folder_size_min_files

    local_run_index = data.run_index

    progress_max = groups_count
    status_text = "Sorting done. " + ["Copying ", "Moving "][move_files_not_copy] + "started"
    print(status_text)
    progress_value = 0
    QApplication.processEvents()

    new_folder_digits = int(math.log(conf.target_groups, 10)) + 1

    action = shutil.copy
    if move_files_not_copy:
        action = shutil.move

    try:
        os.makedirs(new_folder)
    except Exception as e:
        print("Could not create folder ", e)

    ungroupables = []
    for group_index, grp in enumerate(group_db):
        if grp[3] >= conf.folder_size_min_files:
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
                if local_run_index != data.run_index:
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
            if local_run_index != data.run_index:
                return
            QApplication.processEvents()

    if move_files_not_copy:
        conf.start_folder = new_folder


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

    group_color = np.split(group_db[group_n][1][:conf.hg_bands_count * 3], 3)
    group_color = ndimage.zoom(group_color, [1, 256 / conf.hg_bands_count], mode='constant')
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
    # global run_index
    global new_vector_folder

    local_run_index = data.run_index

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
        if local_run_index != data.run_index:
            return


def create_samples():
    global group_db
    global groups_count
    # global run_index
    global new_vector_folder
    global progress_max
    global progress_value
    global status_text

    if "HSV" not in selected_color_spaces:
        return
    for subspace in "HSV":
        if subspace not in selected_color_subspaces:
            return

    local_run_index = data.run_index

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
            for hg_bin in range(conf.hg_bands_count):
                bins_sum += group_db[group_n][1][color_band * conf.hg_bands_count + hg_bin]
            band_coefficient = im_size[0] * im_size[1] / bins_sum / 2
            for hg_bin in range(conf.hg_bands_count):
                bin_start = 256 * hg_bin // conf.hg_bands_count
                bin_end = 256 * (hg_bin + 1) // conf.hg_bands_count
                band_index = color_band * conf.hg_bands_count + hg_bin
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
                if local_run_index != data.run_index:
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
                AnimatedHistogram.image_n_group_iterator = itertools.chain.from_iterable(
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


def get_dominant_image(work_pair):
    crop_compare = distance_db.crop_compare_category.iloc[work_pair]
    if crop_compare and crop_compare < 0:
        return 1

    size_compare = distance_db.size_compare_category.iloc[work_pair]
    if size_compare and size_compare > 0:
        return 1
    return 0


def get_image_pair(work_pair):
    id_l = distance_db.im_1.iloc[work_pair]
    id_r = distance_db.im_2.iloc[work_pair]
    if get_dominant_image(work_pair) > 0:
        return id_r, id_l
    else:
        return id_l, id_r


def get_image_pair_crops(work_pair):
    crops = distance_db.crops.iloc[work_pair]
    if type(crops) is not list:
        return None, None

    if get_dominant_image(work_pair) > 0:
        return crops[4:], crops[:4]
    else:
        return crops[:4], crops[4:]


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
    suggest_mode = 0
    # suggested_deletes = dict()
    image_delete_candidates = {}
    image_delete_final = set()
    size_comparison = 0
    # resort_with_crops = True
    list_filtered = False
    original_distance_db = None

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.label_percent.mousePressEvent = self.central_label_click

        self.paint_sheet = PaintSheet(self)
        self.paint_sheet.setSizePolicy(QtWidgets.QSizePolicy(*[QtWidgets.QSizePolicy.Expanding]*2))
        self.thumb_sheet = ThumbSheet(self, self.show_pair, self.mark_pair_visual)
        self.thumb_sheet.setMaximumHeight(155)
        self.thumb_sheet.setMinimumHeight(155)
        self.paint_sheet.thumb_sheet = self.thumb_sheet
        # self.thumb_sheet.resize(1000, 155)

        self.mode_buttons.buttonClicked.connect(self.reset_thumbs)
        self.push_colored.clicked.connect(self.reset_thumbs)
        self.suggest_buttons.buttonClicked.connect(self.switch_suggested)
        self.filter_buttons.buttonClicked.connect(self.toggle_filter_pairs)

        self.push_mark_suggested.clicked.connect(self.mark_suggested_pairs)
        self.push_apply_marked.clicked.connect(self.apply_marked)
        self.push_move_applied.clicked.connect(self.move_files)
        self.push_mark_clear.clicked.connect(self.clear_marks)
        self.push_goto_first.clicked.connect(self.goto_first_pair)
        self.push_resort_pairs.clicked.connect(self.resort_pairs)

        self.frame_img.layout().addWidget(self.paint_sheet)
        self.frame_img.layout().addWidget(self.thumb_sheet)
        self.thumb_timer = QtCore.QTimer()
        self.thumb_timer.timeout.connect(self.load_new_thumb)
        self.thumb_timer.setInterval(150)
        self.show_pair()
        self.animate_timer = QtCore.QTimer()
        self.animate_timer.timeout.connect(self.redraw_animation)
        self.animate_timer.setInterval(800)
        self.switch_suggested(True)

        self.label_name_r.mousePressEvent = self.right_label_click
        self.label_name_l.mousePressEvent = self.left_label_click

    def toggle_filter_pairs(self, check=None):
        global distance_db
        self.list_filtered = -2 if check is False else self.filter_buttons.checkedId()

        if self.list_filtered == -2:
            if self.original_distance_db is not None:
                distance_db = distance_db.combine_first(self.original_distance_db)
                self.original_distance_db = None
        else:
            if self.original_distance_db is None:
                self.original_distance_db = distance_db
            else:
                distance_db = self.original_distance_db

            undelete_list = distance_db.apply(self.get_filtered_pair, axis=1)  # todo: new procedure here
            distance_db = distance_db[undelete_list]

        self.adjust_suggettions_block()
        self.reset_thumbs(True)
        self.goto_first_pair()

    def adjust_suggettions_block(self):
        def enable_buttons(bool_list):
            for button, option in zip(self.suggest_buttons.buttons(), bool_list):
                button.setEnabled(option)

        if self.list_filtered == -2:
            self.push_suggest_size_and_crop.click()
            enable_buttons([1, 1, 1, 1, 1])
        elif self.list_filtered == -3:
            self.push_suggest_any.click()
            enable_buttons([0, 0, 0, 1, 1])
        elif self.list_filtered == -4:
            enable_buttons([1, 1, 1, 1, 1])
            self.push_suggest_size_and_crop.click()
        elif self.list_filtered == -5:
            enable_buttons([0, 1, 0, 1, 1])
            self.push_suggest_crop_only.click()
        elif self.list_filtered == -6:
            enable_buttons([0, 0, 1, 1, 1])
            self.push_suggest_size_only.click()
        elif self.list_filtered == -7:
            enable_buttons([0, 1, 1, 1, 1])
            self.push_suggest_size_and_crop.click()

    def switch_suggested(self, button_id):
        # if self.suggest_mode != self.suggest_buttons.checkedId() or type(button_id) is bool:
        self.suggest_mode = self.suggest_buttons.checkedId()
        self.update_all_delete_marks()
        self.show_pair()

    def clear_marks(self):
        distance_db.del_im_1 = 0
        distance_db.del_im_2 = 0
        self.push_apply_marked.setEnabled(False)
        self.push_mark_clear.setEnabled(False)
        self.image_delete_candidates.clear()
        self.update_all_delete_marks()
        self.thumb_sheet.update()
        self.update_central_lbl()

    def mark_suggested_pairs(self):
        for work_pair in range(self.current_pair + 1):
            im_1_mark, im_2_mark = self.get_delete_suggest(work_pair)
            self.mark_pair_in_db(work_pair, im_1_mark, im_2_mark, False)
        self.update_all_delete_marks()
        self.thumb_sheet.update()
        self.update_central_lbl()

    def mark_pair_in_db(self, work_pair, im_1_order, im_2_order, force=True):
        pair_index = distance_db.index[work_pair]
        id_1 = distance_db.im_1.iloc[work_pair]
        id_2 = distance_db.im_2.iloc[work_pair]

        if not force:
            if self.image_delete_candidates.get(id_1, False):
                return
            if self.image_delete_candidates.get(id_2, False):
                return
            if distance_db.del_im_1.iloc[work_pair] != 0:
                return
            if distance_db.del_im_2.iloc[work_pair] != 0:
                return

        for im_id, im_order in [(id_1, im_1_order), (id_2, im_2_order)]:
            old_set = self.image_delete_candidates.get(im_id, set())
            if im_order == -1:
                old_set.add(work_pair)
                self.image_delete_candidates[im_id] = old_set
            else:
                if len(old_set):
                    old_set.discard(work_pair)
                    if len(old_set) == 0:
                        self.image_delete_candidates.pop(im_id, None)
        distance_db.loc[pair_index, "del_im_1"] = im_1_order
        distance_db.loc[pair_index, "del_im_2"] = im_2_order
        if self.image_delete_candidates and not self.push_apply_marked.isEnabled():
            self.push_apply_marked.setEnabled(True)
            self.push_mark_clear.setEnabled(True)

    def mark_pair_visual(self, shift, left_order, right_order):
        if not len(distance_db):
            return
        work_pair = (self.current_pair + shift) % len(distance_db)

        if get_dominant_image(work_pair):
            left_order, right_order = right_order, left_order

        self.mark_pair_in_db(work_pair, left_order, right_order)

        self.update_all_delete_marks()
        self.thumb_sheet.update()
        self.update_central_lbl()

    def resort_pairs(self):
        global distance_db
        set_status("(4/4) Resorting pairs...", 0, len(distance_db))
        local_run_index = data.run_index
        data.set_smooth_time(50)
        thumb_cache = {}
        resized_thumb_cache = {}
        tasks_count = len(distance_db)
        QApplication.processEvents()
        self.hide()
        self.animate_timer.stop()
        main_win.redraw_timer.start()
        resolutions = [15, 30, 60, 120, 240, 480]

        def get_image_thumb(im_id):
            img = thumb_cache.get(im_id, None)
            if img is None:
                if len(thumb_cache) > 1000:
                    thumb_cache.pop(next(iter(thumb_cache)))
                img = load_image(image_DB.image_paths[im_id])
                img = img.convert("RGB").resize((480, 480), resample=Image.BILINEAR)
                thumb_cache[im_id] = img
            return img

        def get_resized_image_thumb(im_id, width, height, res_id):
            img = resized_thumb_cache.get((im_id, width, height), None)
            if img is None:
                if res_id < len(resolutions) - 1:
                    im_res = resolutions[res_id]
                    img = get_resized_image_thumb(im_id, im_res, im_res, res_id + 1)
                else:
                    img = get_image_thumb(im_id)
                img = img.resize((width, height), resample=Image.BILINEAR)
                resized_thumb_cache[(im_id, width, height)] = img
            return img

        def get_best_crop2(im_id_l, im_id_r, res_id, crops, best_dist=None):
            im_res = resolutions[res_id]

            def get_cropped_img(im_id, crop_tuple):
                resized_width = im_res + crop_tuple[0] + crop_tuple[2]
                resized_height = im_res + crop_tuple[1] + crop_tuple[3]
                resized_img = get_resized_image_thumb(im_id, resized_width, resized_height, res_id + 1)
                croping_tuple = crop_tuple[0], crop_tuple[1], crop_tuple[0] + im_res, crop_tuple[1] + im_res
                return resized_img.crop(croping_tuple)

            def make_crops_row2(im_id, sub_crops, image_2_raw):
                crops_list = []
                for crop_part in range(4):
                    crop_tuple_plus = sub_crops.copy()
                    crop_tuple_plus[crop_part] += 1
                    diff = ImageChops.difference(get_cropped_img(im_id, crop_tuple_plus), image_2_raw)
                    crops_list.append(np.ravel(diff))
                return np.stack(crops_list)

            image_l_c = get_cropped_img(im_id_l, crops[:4])
            image_r_c = get_cropped_img(im_id_r, crops[4:])

            if best_dist is None:
                best_dist = la_norm(np.asarray(image_l_c).astype(int) - np.asarray(image_r_c))
            if sum(crops) > im_res * .7:
                return best_dist, crops

            dist1 = la_norm(make_crops_row2(im_id_l, crops[:4], image_r_c), axis=1).tolist()
            dist2 = la_norm(make_crops_row2(im_id_r, crops[4:], image_l_c), axis=1).tolist()
            cropped_distances = dist1 + dist2

            best_crop_dist = min(cropped_distances)
            best_crop_id = cropped_distances.index(best_crop_dist)

            QApplication.processEvents()
            if best_crop_dist > best_dist:
                resized_thumb_cache.clear()
                # gc.collect()
                return best_dist, crops
            else:
                crops[best_crop_id] += 1
                return get_best_crop2(im_id_l, im_id_r, res_id, crops, best_crop_dist)

        def recalc_distance_with_crops(pair_record):
            if local_run_index != data.run_index:
                return
            id_r, id_l = pair_record[:2]

            crops = [0] * 8
            distance = None
            for res_id in range(len(resolutions)):
                crops = [i * 2 - 1 * (i > 0) for i in crops]
                distance, crops = get_best_crop2(id_l, id_r, res_id, crops)

            if sum(crops) == 0:
                crops = None

            pair_record['dist'] = int(distance)
            pair_record['crops'] = crops
            categories = {1: 1, 2: 1.1, 3: 3}
            if crops:
                crop_size_l = (480 - crops[0] - crops[2]) * (480 - crops[1] - crops[3])
                crop_size_r = (480 - crops[4] - crops[6]) * (480 - crops[5] - crops[7])
                crop_size_compare = crop_size_r / crop_size_l
                for category, threshold in categories.items():
                    if crop_size_compare > threshold:
                        pair_record.crop_compare_category = - category
                    if 1 / crop_size_compare > threshold:
                        pair_record.crop_compare_category = category

            tasks_passed = pair_record.name + 1
            data.passed_percent = tasks_passed / tasks_count
            text_to_go = f"Resorting pairs... ({tasks_passed}/{tasks_count})."
            set_status(text_to_go, tasks_passed, muted=True)

            QApplication.processEvents()
            return pair_record

        distance_db['crops'] = None
        distance_db = distance_db.apply(recalc_distance_with_crops, axis=1)

        if local_run_index != data.run_index:
            return

        distance_db.sort_values("dist", inplace=True)
        distance_db.reset_index(drop=True, inplace=True)
        conf.compare_by_angles = False

        thumb_cache.clear()
        data.set_smooth_time(0)
        set_status("Resort complete. Showing image pairs.", 0, 100)
        main_win.redraw_dialog()
        self.push_resort_pairs.setEnabled(False)
        QApplication.processEvents()
        self.thumb_sheet.central_pixmap = 0
        self.current_pair = 0
        self.image_delete_candidates.clear()
        # self.marked_pairs.clear()
        self.reset_thumbs(True)
        self.show()
        # self.update_all_delete_marks()
        # self.show_pair()

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

    def hideEvent(self, a0: QtGui.QHideEvent) -> None:
        if self.draw_diff_mode:
            self.toggle_diff_mode()
        main_win.btn_show_compare_wnd.setChecked(False)
        # self.animate_timer.stop()

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        if not a0.spontaneous():
            self.image_delete_candidates.clear()
            self.image_delete_final.clear()
            self.push_move_applied.setEnabled(False)
            self.push_apply_marked.setEnabled(False)
            self.push_mark_clear.setEnabled(False)
            self.reset_thumbs(True)

    def apply_marked(self):
        self.toggle_filter_pairs(False)
        old_image_delete_count = len(self.image_delete_final)
        self.image_delete_final.update(self.image_delete_candidates.keys())

        pairs_to_delete_1 = distance_db.im_1.isin(self.image_delete_candidates)
        pairs_to_delete_2 = distance_db.im_2.isin(self.image_delete_candidates)
        pairs_to_delete_3 = distance_db.del_im_1 != 0
        pairs_to_delete_4 = distance_db.del_im_2 != 0
        pairs_to_delete = pairs_to_delete_1 | pairs_to_delete_2 | pairs_to_delete_3 | pairs_to_delete_4

        distance_db.drop(distance_db[pairs_to_delete].index, inplace=True)

        print(f"Added {len(self.image_delete_final) - old_image_delete_count} images to delete list, "
              f"totaling {len(self.image_delete_final)}")

        self.image_delete_candidates.clear()
        self.current_pair = 0
        self.thumb_sheet.central_pixmap = 0
        self.toggle_filter_pairs()
        self.update_all_delete_marks()
        self.reset_thumbs(True)

        self.push_apply_marked.setEnabled(False)
        self.push_move_applied.setEnabled(True)
        self.push_mark_clear.setEnabled(False)

    def goto_first_pair(self):
        self.show_pair(-self.current_pair)

    def move_files(self):
        print("Files to be moved")
        self.push_move_applied.setEnabled(False)

        for im_index in self.image_delete_final:
            full_name = image_DB.image_paths[im_index]
            parent_folder = os.path.dirname(full_name)
            parent_start_folder = os.path.dirname(conf.start_folder)
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
                # return

        list_images_to_delete = list(self.image_delete_final)
        list_images_to_delete.sort(reverse=True)
        image_DB.drop(index=list_images_to_delete, inplace=True)

        self.image_delete_final.clear()
        global image_count
        image_count = len(image_DB)

        self.show_pair()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Left:
            self.show_pair(-1)
        elif e.key() == QtCore.Qt.Key_Right:
            self.show_pair(1)

    def central_label_click(self, a0: QtGui.QMouseEvent):
        if a0.button() == QtCore.Qt.RightButton:
            frames = [self.frame_1, self.frame_2, self.frame_3, self.frame_4]
            option_state = self.frame_1.isVisible()
            for frame in frames:
                frame.setVisible(not option_state)
        elif a0.button() == QtCore.Qt.LeftButton:
            self.toggle_diff_mode()

    def toggle_diff_mode(self):
        self.draw_diff_mode = not self.draw_diff_mode
        self.paint_sheet.separate_pictures = self.draw_diff_mode
        if self.draw_diff_mode:
            self.animate_timer.start()
            self.label_percent.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.paint_sheet.separator_line = .2
        else:
            self.animate_timer.stop()
            self.paint_sheet.pixmap_l = self.image_pixmaps[0]
            self.paint_sheet.pixmap_r = self.image_pixmaps[1]
            self.label_name_l.setLineWidth(1)
            self.label_name_r.setLineWidth(1)
            self.label_percent.setFrameShadow(QtWidgets.QFrame.Raised)
            self.paint_sheet.active_frame = None
            self.paint_sheet.separator_line = .5
        self.update_central_lbl()
        self.paint_sheet.update()

    def reset_thumbs(self, button_id):
        if self.thumb_mode != self.mode_buttons.checkedId() or type(button_id) is bool:
            self.thumb_sheet.pixmaps = [None] * 15
        self.thumb_mode = self.mode_buttons.checkedId()
        self.update_all_delete_marks()
        self.show_pair()

    def load_image_and_pixmap(self, path):
        im = load_image(path).convert("RGB")
        qimage_obj = QtGui.QImage(im.tobytes(), im.width, im.height, im.width * 3, QtGui.QImage.Format_RGB888)
        pixmap_obj = QtGui.QPixmap(qimage_obj)
        return im, pixmap_obj

    def update_central_lbl(self):
        if not len(distance_db):
            self.label_percent.setText(f"\nMarked  for deletion {len(self.image_delete_final)} images")
            return
        im_dist = distance_db.dist.iloc[self.current_pair]
        # mean_diffs = distance_db.mean_diffs.iloc[self.current_pair]
        marks_count = len(distance_db[distance_db.del_im_1 != 0])
        first_label_text = f" Pair {self.current_pair + 1} of {len(distance_db)} \nDifference "
        first_label_text += f"{im_dist * 100:.2f}%" if im_dist < 1 else f"{im_dist:.0f}"
        # first_label_text += f"\nMean diffs: {mean_diffs:d}"
        first_label_text += "\nDifference mode" if self.draw_diff_mode else "\nSide by side mode"
        first_label_text += f"\nMarked {marks_count} pairs"
        first_label_text += f"\n{len(self.image_delete_candidates)} + {len(self.image_delete_final)} images"
        self.label_percent.setText(first_label_text)

    def size_label_text(self, idx: int, im_id=None):
        color_template = """QLabel {
                        border-width: 3px;
                        border-style: solid;
                        border-color: """
        red_color = "rgb(200, 70, 70);}"
        green_color = "rgb(70, 200, 70);}"
        gray_color = "rgb(70, 70, 70);}"

        label = [self.label_size_l, self.label_size_r][idx]
        if not len(distance_db) or im_id is None:
            label.setStyleSheet(color_template + gray_color)
            label.setText("No image")
            return

        img_size = image_DB.sizes[im_id]

        text = f"Image {im_id + 1} of {image_count}"
        text += f"\n {img_size[0]} X {img_size[1]}"
        text += f"\n {image_DB.megapixels[im_id]:.02f} Mpix"

        size_labels = {0: "equal sizes",
                       1: "a bit bigger",
                       2: "bigger",
                       3: "much bigger",
                       -1: "a bit smaller",
                       -2: "smaller",
                       -3: "much smaller",
                       }

        crop_labels = {0: "equal crops",
                       1: "bit less to crop",
                       2: "less to crop",
                       3: "much less to crop",
                       -1: "bit more to crop",
                       -2: "more to crop",
                       -3: "much more to crop",
                       }

        work_record = distance_db.iloc[self.current_pair]

        # size_comparison = (distance_db.size_compare_category.iloc[self.current_pair])
        # crop_comparison = (distance_db.crop_compare_category.iloc[self.current_pair])
        size_comparison = work_record.size_compare_category
        crop_comparison = work_record.crop_compare_category
        if get_dominant_image(self.current_pair) == (idx == 0):
            size_comparison *= -1
            crop_comparison *= -1

        text += "\n" + size_labels[size_comparison]

        if crop_comparison:
            text += "\n" + crop_labels[crop_comparison]

        crop_list = get_image_pair_crops(self.current_pair)[idx]
        if crop_list and sum(crop_list):
            text += f"\n To crop ({crop_list[0]}, {crop_list[1]}, {crop_list[2]}, {crop_list[3]})"

        label.setText(text)

        color_category = self.get_delete_suggest(self.current_pair, True)[idx]
        # if get_dominant_image(self.current_pair):
        #     color_category *= -1

        color_template += (red_color, gray_color, green_color)[color_category + 1]

        label.setStyleSheet(color_template)

    def generate_diff_image2(self, image_l, image_r, do_thumb=False, work_pair=None):
        def make_pixmap(img):
            qimage_obj = QtGui.QImage(img.tobytes(), img.width, img.height, img.width * 3, QtGui.QImage.Format_RGB888)
            pixmap_obj = QtGui.QPixmap(qimage_obj)
            return pixmap_obj

        if work_pair is None:
            work_pair = self.current_pair
        crop_l_orig, crop_r_orig = get_image_pair_crops(work_pair)

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

            if not do_thumb:
                self.image_l = image_l
                self.image_r = image_r
                self.image_pixmaps[0] = make_pixmap(image_l)
                self.image_pixmaps[1] = make_pixmap(image_r)

        if do_thumb:
            thm_size = self.thumb_sheet.thumb_size
            image_r = image_r.resize((thm_size, thm_size), Image.BILINEAR)
            image_l = image_l.resize((thm_size, thm_size), Image.BILINEAR)
        else:
            if image_l.height > image_r.height:
                image_r = image_r.resize(image_l.size, Image.BILINEAR)
            elif image_l.height < image_r.height or image_l.width != image_r.width:
                image_l = image_l.resize(image_r.size, Image.BILINEAR)

        if self.thumb_mode == -3:
            image_d = Image.blend(ImageOps.invert(image_l), image_r, .5)
        else:
            image_d = ImageChops.difference(image_l, image_r)

        if not self.push_colored.isChecked():
            image_bands = image_d.split()
            o = -128 if self.thumb_mode == -3 else 0
            image_sum_bw = ImageChops.add(ImageChops.add(image_bands[0], image_bands[1], 1, o), image_bands[2], 1, o)
            image_d = image_sum_bw.convert("RGB")

        if self.thumb_mode == -4:
            image_d = ImageOps.invert(image_d)

        return image_d

    def show_pair(self, shift=0):
        if not len(distance_db):
            self.size_label_text(0)
            self.size_label_text(1)
            self.label_name_l.setText("")
            self.label_name_r.setText("")
            self.paint_sheet.pixmap_l = None
            self.paint_sheet.pixmap_r = None
            self.image_pixmaps = [QtGui.QPixmap] * 4
            for i in range(15):
                self.thumb_sheet.pixmaps[i] = None
                self.thumb_sheet.delete_marks[i] = None
                self.thumb_sheet.suggest_marks[i] = None
            self.paint_sheet.update()
            self.thumb_sheet.update()
            return

        self.current_pair += shift
        self.current_pair %= len(distance_db)
        one_step = -1 if shift < 0 else 1
        for i in range(abs(shift)):
            self.thumb_sheet.central_pixmap = (self.thumb_sheet.central_pixmap + one_step) % 15
            thumb_id = (self.thumb_sheet.central_pixmap + 7 * one_step) % 15
            self.thumb_sheet.pixmaps[thumb_id] = None
            self.thumb_sheet.delete_marks[thumb_id] = None
            self.thumb_sheet.suggest_marks[thumb_id] = None

        id_l, id_r = get_image_pair(self.current_pair)
        path_l = image_DB.image_paths[id_l]
        path_r = image_DB.image_paths[id_r]

        self.label_name_l.setText(path_l)
        self.label_name_r.setText(path_r)
        self.image_l, self.image_pixmaps[0] = self.load_image_and_pixmap(path_l)
        self.image_r, self.image_pixmaps[1] = self.load_image_and_pixmap(path_r)
        self.size_label_text(0, id_l)
        self.size_label_text(1, id_r)
        self.update_central_lbl()
        self.image_d = self.generate_diff_image2(self.image_l, self.image_r)
        if self.thumb_mode == -3:
            image_d_second = ImageOps.invert(self.image_d)
        else:
            image_d_second = self.image_d

        self.image_pixmaps[2] = pixmap_from_image(self.image_d)
        self.image_pixmaps[3] = pixmap_from_image(image_d_second)
        self.paint_sheet.pos = None
        self.paint_sheet.img_zoom = 1
        self.paint_sheet.img_xy = QtCore.QPoint(0, 0)
        self.paint_sheet.img_xy = QtCore.QPoint(0, 0)
        self.paint_sheet.suggest_marks = self.get_delete_suggest(self.current_pair, True)
        self.thumb_sheet.update()

        if self.draw_diff_mode:
            self.redraw_animation()
        else:
            self.paint_sheet.pixmap_l = self.image_pixmaps[0]
            self.paint_sheet.pixmap_r = self.image_pixmaps[1]

            self.paint_sheet.update()
            # self.preview_sheet.update()
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
        if not len(distance_db):
            return
        thumb_id = (self.thumb_sheet.central_pixmap + shift) % 15
        if self.thumb_sheet.pixmaps[thumb_id]:
            return
        work_pair = (self.current_pair + shift) % len(distance_db)
        id_l, id_r = get_image_pair(work_pair)
        image_l = load_image(image_DB.image_paths[id_l]).convert("RGB")
        image_r = load_image(image_DB.image_paths[id_r]).convert("RGB")
        image_d = self.generate_diff_image2(image_l, image_r, True, work_pair)

        self.thumb_sheet.pixmaps[thumb_id] = pixmap_from_image(image_d)
        self.update_preview_delete_marks(work_pair, thumb_id)
        self.thumb_sheet.update()
        return True

    def update_all_delete_marks(self):
        if not len(distance_db):
            return
        for i in range(1, 16):
            shift = i // 2 if i % 2 else -i // 2
            thumb_id = (self.thumb_sheet.central_pixmap + shift) % 15
            work_pair = (self.current_pair + shift) % len(distance_db)
            self.update_preview_delete_marks(work_pair, thumb_id)

    def get_delete_suggest(self, work_pair, check_flip=False):
        distance_db_row = distance_db.iloc[work_pair]
        size_comparison = distance_db_row.size_compare_category
        crop_comparison = distance_db_row.crop_compare_category
        delete_left = -1, 1
        delete_right = 1, -1

        if check_flip and get_dominant_image(work_pair):
            delete_left, delete_right = delete_right, delete_left

        if self.suggest_mode == -2:
            if crop_comparison < 0 < size_comparison:
                return delete_right
            if size_comparison < 0 < crop_comparison:
                return delete_left
        elif self.suggest_mode == -3:
            if crop_comparison < 0:
                return delete_right
            if 0 < crop_comparison:
                return delete_left
        elif self.suggest_mode == -4:
            if 0 < size_comparison:
                return delete_right
            if size_comparison < 0:
                return delete_left
        elif self.suggest_mode == -5:
            return delete_left
        return 0, 0

    def get_filtered_pair(self, distance_db_row):
        size_comparison = distance_db_row.size_compare_category
        crop_comparison = distance_db_row.crop_compare_category

        if self.list_filtered == -3:
            return crop_comparison == 0 == size_comparison
        elif self.list_filtered == -4:
            return crop_comparison * size_comparison < 0
        elif self.list_filtered == -5:
            return crop_comparison != 0 and size_comparison == 0
        elif self.list_filtered == -6:
            return crop_comparison == 0 and size_comparison != 0
        elif self.list_filtered == -7:
            return crop_comparison * size_comparison > 0
        return False

    def update_preview_delete_marks(self, work_pair, thumb_id):
        id_l = distance_db.im_1.iloc[work_pair]
        id_r = distance_db.im_2.iloc[work_pair]
        mark_left = distance_db.del_im_1.iloc[work_pair]
        mark_right = distance_db.del_im_2.iloc[work_pair]
        if self.image_delete_candidates.get(id_l, None) and not mark_left:
            mark_left = -2
        if self.image_delete_candidates.get(id_r, None) and not mark_right:
            mark_right = -2

        if mark_left or mark_right:
            if get_dominant_image(work_pair):
                mark_left, mark_right = mark_right, mark_left
            self.thumb_sheet.delete_marks[thumb_id] = mark_left, mark_right
        else:
            self.thumb_sheet.delete_marks[thumb_id] = None

        mark_left, mark_right = self.get_delete_suggest(work_pair, True)
        self.thumb_sheet.suggest_marks[thumb_id] = mark_left, mark_right

    def redraw_animation(self):
        if not self.isVisible():
            self.animate_timer.stop()
            return
        self.animation_frame = 1 - self.animation_frame
        self.paint_sheet.pixmap_l = self.image_pixmaps[3 - self.animation_frame]
        self.paint_sheet.pixmap_r = self.image_pixmaps[self.animation_frame]
        if self.image_pixmaps[0] is not None and self.image_pixmaps[1] is not None:
            self.paint_sheet.force_sizes = self.image_pixmaps[0].size(), self.image_pixmaps[1].size()
        self.label_name_r.setLineWidth(1 + self.animation_frame * 3)
        self.label_name_l.setLineWidth(1 + 3 - self.animation_frame * 3)
        self.paint_sheet.active_frame = self.animation_frame
        self.paint_sheet.update()


class VisImSorterGUI(QMainWindow, VisImSorterInterface.Ui_VisImSorter):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.redraw_timer = QtCore.QTimer()
        self.drag_timer = QtCore.QTimer()
        self.font = QtGui.QFont()
        self.init_elements()

    def init_elements(self):
        # global start_folder

        self.font.setPointSize(14)

        self.redraw_timer.timeout.connect(self.redraw_dialog)
        self.redraw_timer.setInterval(50)
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

        # self.btn_angles.clicked.connect(self.slider_image_max_likeness_changed)
        # self.btn_distances.clicked.connect(self.slider_image_max_likeness_changed)

        self.btn_show_compare_wnd.clicked.connect(compare_wnd_toggle)

        # self.slider_image_max_likeness.valueChanged.connect(self.slider_image_max_likeness_changed)
        # self.slider_image_max_likeness_changed()

        self.slider_max_pairs.valueChanged.connect(self.slider_max_pairs_changed)
        self.slider_max_pairs_changed()

        self.slider_enforce_equal.valueChanged.connect(self.slider_equal_changed)
        self.slider_equal_changed()

        self.list_color_spaces.itemSelectionChanged.connect(self.color_spaces_reselected)
        self.color_spaces_reselected()
        self.tabWidget.currentChanged.connect(self.duplicate_tab_switched)

        list_color_spaces = [self.list_color_spaces_CMYK, self.list_color_spaces_HSV,
                             self.list_color_spaces_RGB, self.list_color_spaces_YCbCr]
        for c_list in list_color_spaces:
            [c_list.item(i).setSelected(True) for i in range(3)]
        [self.list_color_spaces.item(i).setSelected(True) for i in range(2)]

        self.btn_stop.clicked.connect(self.stop_button_pressed)
        self.btn_start.clicked.connect(self.start_button_pressed)
        self.progressBar.setVisible(False)
        # self.progressBar.valueChanged.connect(self.update)
        self.enable_elements()

        if len(sys.argv) > 1:
            if os.path.isdir(sys.argv[1]):
                conf.start_folder = sys.argv[1]
                self.directory_changed()

    def start_button_pressed(self):
        global move_files_not_copy
        global show_histograms
        global export_histograms
        # global start_folder
        global new_folder
        global create_samples_enabled
        global final_sort

        local_run_index = data.run_index

        conf.folder_constraint_type = self.combo_folder_constraints.currentIndex()
        conf.folder_constraint_value = self.spin_num_constraint.value()
        conf.search_subdirectories = self.check_subdirs_box.isChecked()
        move_files_not_copy = self.radio_move.isChecked()
        show_histograms = self.check_show_histograms.isChecked()
        export_histograms = self.check_export_histograms.isChecked()
        create_samples_enabled = self.check_create_samples.isChecked()
        conf.enable_stage2_grouping = self.check_stage2_grouping.isChecked()
        conf.stage1_grouping_type = self.combo_stage1_grouping.currentIndex()
        conf.enable_multiprocessing = self.check_multiprocessing.isChecked()
        conf.search_duplicates = self.tab_unduplicate.isVisible()
        conf.compare_by_angles = self.btn_angles.isChecked()
        conf.compare_resort = self.btn_compare_resort.isChecked()

        final_sort = self.combo_final_sort.currentIndex()
        if self.check_equal_name.isChecked():
            conf.group_with_similar_name = self.spin_equal_first_symbols.value()
        else:
            conf.group_with_similar_name = 0

        conf.selected_color_bands.clear()

        for item in self.list_color_spaces.selectedItems():
            space_name = str(item.text())
            w = self.group_colorspaces_all.findChild(QtWidgets.QListWidget, "list_color_spaces_" + space_name)
            subspaces = {item.text() for item in w.selectedItems()}
            conf.selected_color_bands[space_name] = subspaces
        selected_bands_count = sum([len(i) for i in conf.selected_color_bands.values()])

        if not selected_bands_count:
            QMessageBox.warning(None, "VisImSorter: Error",
                                "Please select at least one color space and at least one sub-color space")
            return

        self.enable_elements(False)
        self.redraw_dialog()
        run_sequence()
        if local_run_index != data.run_index:
            self.btn_stop.setText(data.status_text)
            self.btn_stop.setStyleSheet("background-color: rgb(250,150,150)")
        else:
            self.btn_stop.setText(data.status_text)
            self.btn_stop.setStyleSheet("background-color: rgb(150,250,150)")
            data.run_index += 1
        self.btn_stop.setFont(self.font)
        self.progressBar.setVisible(False)

    def stop_button_pressed(self):
        if self.progressBar.isVisible():
            set_status("Process aborted by user.", abort=True)
        else:
            self.directory_changed(True)
            self.enable_elements()

    def color_spaces_reselected(self):
        for index in range(self.horizontalLayout_7.count()):
            short_name = self.horizontalLayout_7.itemAt(index).widget().objectName()[18:]
            items = self.list_color_spaces.findItems(short_name, QtCore.Qt.MatchExactly)
            if len(items) > 0:
                enabled = items[0].isSelected()
                self.horizontalLayout_7.itemAt(index).widget().setEnabled(enabled)

    def enable_elements(self, standby_mode=True):
        standby_controls = [self.group_input, self.group_analyze, self.tabWidget,
                            self.group_move, self.group_final]
        [control.setEnabled(standby_mode) for control in standby_controls]
        self.btn_start.setVisible(standby_mode)
        self.btn_stop.setVisible(not standby_mode)
        self.btn_show_compare_wnd.setVisible(not standby_mode)

        if standby_mode:
            self.btn_stop.setStyleSheet("background-color: rgb(200,150,list_color_spaces)")
            self.btn_stop.setFont(self.font)
            data.progress_value = 0
            self.redraw_dialog()
            self.redraw_timer.stop()
        else:
            self.btn_stop.setText("Stop")
            self.progressBar.setVisible(True)
            self.redraw_timer.start()

        self.duplicate_tab_switched()

    def duplicate_tab_switched(self):
        grouping = self.tab_grouping.isVisible()
        self.group_move.setEnabled(grouping)
        self.group_final.setEnabled(grouping)
        self.groupBox_5.setEnabled(grouping)

    def slider_histogram_bands_changed(self):
        conf.hg_bands_count = self.slider_histogram_bands.value()
        self.lbl_histogram_bands.setText(str(conf.hg_bands_count))

    def slider_image_split_steps_changed(self):
        raw_splits = self.slider_image_split_steps.value()
        raw_split_steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30]
        conf.image_divisions = raw_split_steps[raw_splits]
        self.lbl_image_split_steps.setText(str(conf.image_divisions ** 2))

    def slider_equal_changed(self):
        global enforce_equal_folders
        enforce_equal_folders = self.slider_enforce_equal.value() / 100
        enforce_equal_folders *= enforce_equal_folders
        self.lbl_slider_enforce_equal.setText("%d" % (self.slider_enforce_equal.value() / 3) + "%")

    def slider_max_pairs_changed(self):
        raw_slider = self.slider_max_pairs.value()
        conf.max_pairs = int(10 * 3 ** (0.04 * raw_slider))
        self.lbl_max_pairs.setText(f"{conf.max_pairs}")
        if raw_slider == 200:
            conf.max_pairs = 0
            self.lbl_max_pairs.setText("all")

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
        url_text = unquote(in_dir.mimeData().text())
        firt_path = url_text[8:]
        firt_path_start = url_text[:8]
        if firt_path_start == "file:///":
            if os.path.isdir(firt_path):
                conf.start_folder = firt_path
                self.directory_changed()

    def directory_changed(self, suppress_text=False):

        if os.path.exists(conf.start_folder):
            conf.start_folder = os.path.normpath(conf.start_folder)
            if not suppress_text:
                set_status("Ready to start")

            start_folder_parts = [a + "\\" for a in conf.start_folder.split("\\")]
            start_folder_parts2 = [(a + " ") if a[-1] != "\\" else a for b in start_folder_parts for a in b.split(" ")]
            line_length = 0
            lines_count = max(round(len(conf.start_folder) / 35), 1)
            line_max = max(len(conf.start_folder) // lines_count, 20)
            button_text = ""

            for part_path in start_folder_parts2:
                extended_line_length = line_length + len(part_path)
                if extended_line_length > line_max:
                    button_text += "\n" + part_path
                    line_length = len(part_path)
                else:
                    button_text += part_path
                    line_length += len(part_path)

            self.select_folder_button.setText(button_text)
            self.btn_start.setEnabled(True)
        else:
            if not suppress_text:
                set_status("Please select folder")
            self.select_folder_button.setText("Select folder")
            self.btn_start.setEnabled(False)

    def select_folder(self):
        conf.start_folder = QFileDialog.getExistingDirectory(self, "Choose directory", conf.start_folder or "Y:/")
        self.directory_changed()

    def redraw_dialog(self):
        if self.progressBar.maximum() != data.progress_max:
            self.progressBar.setRange(0, data.progress_max)
        self.progressBar.setValue(int(data.progress_value))
        text_to_go, text_end_time, text_to_go2, text_end_time2 = "", "", "", ""
        step_one_time, step_one_step = data.steps_buffer[0]
        last_time, last_step = data.steps_buffer[-1]
        if (data.last_iteration_progress_value != data.progress_value) and time() - last_time > 3:
            data.last_iteration_progress_value = data.progress_value
            data.steps_buffer.pop(0)
            data.steps_buffer.append((time(), data.progress_value))

        if step_one_time > 0:
            last_period = (time() - step_one_time) / (data.progress_value - step_one_step)
            if data.smooth_speed is None:
                data.smooth_speed = np.ones(50) * last_period
            data_shift = np.roll(data.smooth_speed, 1)
            data_shift[0] = last_period
            data.smooth_speed = mix(data.smooth_speed, data_shift, .03)
            secs_to_go = data.smooth_speed[-1] * (data.progress_max - last_step)
            finish_time = secs_to_go + last_time
            time_to_go = gmtime(finish_time - time())
            if time_to_go.tm_hour:
                time_text = " To go %H:%M:%S hours."
            else:
                time_text = " To go %M:%S minutes."
            text_to_go2 = strftime(time_text, time_to_go)
            text_end_time2 = strftime(" Finish at %H:%M:%S.", localtime(finish_time))

        self.statusbar.showMessage(data.status_text + text_to_go + text_end_time + text_to_go2 + text_end_time2)
        QApplication.processEvents()


def main():
    global main_win
    app = QApplication(sys.argv)
    main_win = VisImSorterGUI()
    main_win.show()
    sys.exit(app.exec_())


main_win = VisImSorterGUI
compare_wnd = CompareGUI

if __name__ == '__main__':
    main()
