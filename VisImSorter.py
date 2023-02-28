import fnmatch
import glob
import itertools
import math
import multiprocessing as mp
import os
import shutil
import sys
from time import sleep, localtime, time, gmtime, strftime
from timeit import default_timer as timer
from prettytable import PrettyTable
# import random
# import matplotlib.animation as animation
# import matplotlib.patches as patches
# import matplotlib.path as path
# import matplotlib.pyplot as plt
# import matplotlib.text as mtext
# import numpy.ma as ma
import matplotlib.colors as mc
import numpy as np
import pandas as pd
from urllib.parse import unquote
import psutil
from PIL import Image, ImageOps, ImageChops
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy import ndimage
from PaintSheet import PaintSheet, ThumbSheet_Scene, recieve_loaded_thumb, restrict, smootherstep_ease
from ImageLoader import load_image, generate_image_vector, all_file_types, generate_thumb_pixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import CompareDialog
import VisImSorterInterface

pool = mp.Pool

image_paths_db = pd.Series
image_DB = pd.DataFrame
distance_db = pd.DataFrame
image_to_group = np.ones(0)
group_lists = []
image_count = 0


gaussian_filter1d = ndimage.gaussian_filter1d
median_filter = ndimage.median_filter
gaussian_filter = ndimage.gaussian_filter
la_norm = np.linalg.norm

function_queue = []


class Constants():
    final_sort_by_color = 150
    final_sort_by_brightness_asc = 151
    final_sort_by_brightness_desc = 152
    final_sort_by_file_count_asc = 153
    final_sort_by_file_count_desc = 154


class Config:
    search_duplicates = False
    image_divisions = 1
    max_likeness = 100
    search_subdirectories = False
    hg_bands_count = 0
    move_files_not_copy = False
    final_sort = 0
    folder_constraint_value = 0
    folder_constraint_type = 0
    group_with_similar_name = 0
    enforce_equal_folders = 0.
    target_groups = 0
    target_group_size = 0
    folder_size_min_files = 0
    selected_color_bands = dict()
    compare_resort = False
    ignore_ext = False

    max_pairs = 0
    start_folder = ""
    start_folder_2 = ""
    second_set_size = 0
    max_threads = 3

    export_histograms = False
    create_samples = False



class Data:
    status_text = ""
    progress_max = 100
    progress_value = 0
    last_iteration_progress_value = 0
    run_index = 0
    working_bands = list()
    raw_time_left = 0
    smooth_end_time = [0.] * 50
    smooth_speed = None
    steps_buffer = [(0., 0)] * 50
    smooth_time_iterations = 0
    event_start_time = 0
    passed_percent = 0
    HSV_color_codes = None
    HSV_color_names = None
    check_crop_levels = 1
    new_folder = ""
    new_vector_folder = ""

    def set_smooth_time(self, iterations: int = 1):
        self.smooth_time_iterations = iterations
        self.smooth_end_time = [0.] * iterations
        self.event_start_time = time()
        self.passed_percent = 0
        data.last_iteration_progress_value = 0
        data.smooth_speed = None
        data.steps_buffer = [(0., 0)] * iterations


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=15, height=15, dpi=100):
        # fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained', facecolor="lightblue")
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor="lightblue")
        self.axes = fig.add_subplot(polar=True)
        super(MplCanvas, self).__init__(fig)


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
        if not muted: print(status)
    if progress_max is not None: data.progress_max = progress_max
    if progress_value is not None: data.progress_value = progress_value
    if abort: data.run_index += 1


def run_sequence():
    global function_queue
    local_run_index = data.run_index

    function_queue.append(init_pool)
    function_queue.append(scan_for_images_new)
    if conf.start_folder_2:
        function_queue.append(scan_for_images_second)
    function_queue.append(collect_image_vectors)

    if conf.search_duplicates:
        function_queue += [create_distance_db_duplicates, show_compare_images_gui]
    else:
        function_queue += [
            close_pool,
            # create_groups_reverse_tree_simple,
            create_groups_reverse_tree_smart,
            optimize_groups_kmeans9,
            choose_destination_folder,
            move_files_pd
        ]

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
            close_pool()
            return

    finish_time = timer()

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
    pool = mp.Pool(processes=conf.max_threads)


def run_fast_scandir(directory_in, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(directory_in):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1][1:].lower() in ext or conf.ignore_ext:
                files.append(f.path)

    for directory in list(subfolders):
        sf, f = run_fast_scandir(directory, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def scan_for_images_new(second_scan=False):
    global image_paths_db
    set_status("Scanning images...")
    QApplication.processEvents()
    scan_folder = conf.start_folder_2 if second_scan else conf.start_folder
    subfolders, image_paths = run_fast_scandir(scan_folder, all_file_types)

    if not len(image_paths):
        set_status("No images found.", abort=True)
        return
    else:
        print(f"Found {len(image_paths)} images.")
    image_paths = pd.Series(image_paths).apply(os.path.normpath).rename("image_paths")
    image_names = image_paths.apply(os.path.basename).rename("image_names")
    image_paths_db = pd.DataFrame((image_paths, image_names)).T.copy()

    if conf.group_with_similar_name > 0:
        image_name_groups = image_names.apply(lambda x: x[:conf.group_with_similar_name])
        image_paths_db["image_name_groups"] = image_name_groups


def scan_for_images_second():
    global image_paths_db
    path_db_1 = image_paths_db
    scan_for_images_new(True)
    conf.second_set_size = len(image_paths_db)
    image_paths_db = pd.concat([image_paths_db, path_db_1])


def collect_image_vectors():
    global image_DB
    global image_count
    global pool

    local_run_index = data.run_index
    image_count = len(image_paths_db)
    data.working_bands.clear()

    if conf.search_duplicates:
        for current_color_space, color_subspaces in conf.selected_color_bands.items():
            for band in color_subspaces:
                band_full_name = current_color_space + "_" + band
                image_paths_db[band_full_name] = None
                data.working_bands.append(band_full_name)
    else:
        image_paths_db['hg'] = None

    set_status(f"(1/4) Generating image histograms...", progress_max=image_count, progress_value=0)
    image_DB = []
    results = []
    links_per_task = 32
    path_dbs = []
    if conf.second_set_size:
        path_dbs.append([image_paths_db[:conf.second_set_size][i:i + links_per_task] for i in
                         range(0, conf.second_set_size, links_per_task)])
    path_dbs.append(
        [image_paths_db[i:i + links_per_task] for i in range(conf.second_set_size, image_count, links_per_task)])

    for path_db_chunks in path_dbs:
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
    image_DB = image_DB.astype({"means": int})
    image_count = len(image_DB)
    if image_count == 0:
        set_status("Failed to create image database.", abort=True)
    else:
        print(f"Loaded {image_count} image histograms.")

    if not conf.search_duplicates:
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


def create_distance_db_duplicates():
    global distance_db
    local_run_index = data.run_index
    im_histograms = image_DB[data.working_bands]

    def compact_compare_db():
        compare_db_sorted = compare_db.sort_values(by="mean_diffs")
        highest_diff = compare_db_sorted.mean_diffs.iloc[min(conf.max_pairs, len(compare_db) - 1)] * 3 + 5
        compare_sub_db_size = compare_db_sorted.mean_diffs.values.searchsorted(highest_diff)
        compare_sub_db_size = max(compare_sub_db_size, conf.max_pairs * 20 + 10000)
        return compare_db.iloc[:compare_sub_db_size].copy()

    if conf.second_set_size:
        im_id_combinations = itertools.product(range(conf.second_set_size), range(conf.second_set_size, len(image_DB)))
        compare_db = pd.DataFrame(im_id_combinations, columns=["im_1", "im_2"])
        compare_db["mean_diffs"] = np.abs(
            image_DB.means[compare_db.im_1].values - image_DB.means[compare_db.im_2].values)
        # compare_db.sort_values(by="mean_diffs", inplace=True)
        # highest_diff = compare_db.mean_diffs.iloc[min(conf.max_pairs, len(compare_db) - 1)] * 1.5 + 5
        # compare_sub_db_size = compare_db.mean_diffs.values.searchsorted(highest_diff)
        # compare_sub_db_size = max(compare_sub_db_size, conf.max_pairs * 3 + 10000)
        # compare_db_rest = compare_db.iloc[:compare_sub_db_size]
        compare_db_rest = compact_compare_db()
    else:
        # im_id_combinations = itertools.combinations(range(len(image_DB)), 2)
        set_status("(2/4) Fast compare image vectors", 0, len(image_DB) - 1)
        compare_db = None
        compare_db_steps_list = []
        for compare_step in range(1, len(image_DB) - 1):
            im_2_idx = image_DB.index.values[compare_step:]
            im_1_idx = image_DB.index.values[:len(im_2_idx)]
            compare_db_step = pd.DataFrame({"im_1": im_1_idx, "im_2": im_2_idx})
            # compare_db_step["mean_diffs"] = np.abs(image_DB.means[im_1_idx].values - image_DB.means[im_2_idx].values)
            compare_db_step["mean_diffs"] = np.abs(image_DB.means.values[:len(im_2_idx)] -
                                                   image_DB.means.values[compare_step:])
            compare_db_steps_list.append(compare_db_step)
            if not compare_step % 100 or compare_step == len(image_DB) - 2:
                if compare_db is not None:
                    compare_db_steps_list = [compare_db] + compare_db_steps_list
                compare_db = pd.concat(compare_db_steps_list, ignore_index=True)
                compare_db = compact_compare_db()
                compare_db_steps_list.clear()

            set_status(progress_value=compare_step)
            QApplication.processEvents()
        compare_db_rest = compare_db

    data.set_smooth_time(20)
    set_status("(2/4) Comparing image vectors", 0, len(compare_db_rest))
    distance_db = None
    dist_db_list = []

    while len(compare_db_rest):
        compare_db_work = compare_db_rest.iloc[:1000].copy()
        compare_db_rest = compare_db_rest.iloc[1000:]
        compare_db_work[data.working_bands] = im_histograms.loc[compare_db_work.im_1].values - im_histograms.loc[
            compare_db_work.im_2].values
        compare_db_work["dist"] = compare_db_work[data.working_bands].applymap(la_norm).mean(axis=1)
        dist_db_list.append(compare_db_work[["im_1", "im_2", "mean_diffs", "dist"]])
        if len(dist_db_list) > 10:
            dist_db_list = [pd.concat(dist_db_list).sort_values("dist").iloc[:conf.max_pairs].copy()]

        set_status(progress_value=(data.progress_max - len(compare_db_rest)))
        QApplication.processEvents()
        if local_run_index != data.run_index:
            return

    distance_db = pd.concat(dist_db_list).sort_values("dist").iloc[:conf.max_pairs].copy()
    distance_db.reset_index(drop=True, inplace=True)
    im_mpx = image_DB.megapixels
    distance_db["size_compare"] = im_mpx.loc[distance_db.im_2].values / im_mpx.loc[distance_db.im_1].values
    distance_db["size_compare_category"] = 0
    distance_db["crops"] = None
    distance_db["crop_compare_category"] = 0
    distance_db["del_im_1"] = 0
    distance_db["del_im_2"] = 0
    categories = {1: 1, 2: 1.1, 3: 3}
    for category, threshold in categories.items():
        distance_db.loc[distance_db["size_compare"] > threshold, "size_compare_category"] = -category
        distance_db.loc[1 / distance_db["size_compare"] > threshold, "size_compare_category"] = category
    name_sizes = image_DB.image_names.apply(len)
    name_difference = name_sizes.loc[distance_db.im_1].values - name_sizes.loc[distance_db.im_2].values
    distance_db["name_difference"] = name_difference


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

#
# def create_groups_reverse_tree_simple():
#     global group_lists, image_to_group
#     local_run_index = data.run_index
#     set_status("(2/4) Creating basic groups.", progress_max=(image_count - conf.target_groups))
#     data.set_smooth_time(20)
#     group_hg_sums = np.hstack([np.vstack(image_DB.means.values), np.vstack(image_DB.hg.values), [[1]] * image_count])
#     group_hg_means = group_hg_sums.copy()
#     group_lists = [[i] for i in range(image_count)]
#
#     while len(group_hg_sums) > conf.target_groups:
#         if local_run_index != data.run_index: return
#         set_status(progress_value=(image_count - len(group_hg_sums)))
#         QApplication.processEvents()
#
#         joining_group = np.argmin(group_hg_sums[:, -1])
#         joining_group_hg = group_hg_means[joining_group]
#
#         if len(group_hg_sums) > 160:
#             mean_diffs = abs(group_hg_means[:, 0] - joining_group_hg[0])
#             closest_groups = np.argpartition(mean_diffs, 100)[:100]
#         else:
#             closest_groups = np.arange(len(group_hg_sums))
#         closest_groups = closest_groups[closest_groups != joining_group]
#
#         diff_to_others = np.square(group_hg_means[closest_groups] - joining_group_hg).sum(axis=1)
#         closest_group = closest_groups[np.argmin(diff_to_others)]
#
#         groups_sum = group_hg_sums[[joining_group, closest_group]].sum(axis=0)
#         group_hg_sums[joining_group] = groups_sum
#         group_hg_means[joining_group] = groups_sum // groups_sum[-1]
#         group_lists[joining_group] = group_lists[joining_group] + group_lists[closest_group]
#
#         group_hg_sums = np.delete(group_hg_sums, closest_group, axis=0)
#         group_hg_means = np.delete(group_hg_means, closest_group, axis=0)
#         del group_lists[closest_group]
#
#     image_to_group = -np.ones(image_count, dtype=int)
#     for i, images in enumerate(group_lists):
#         image_to_group[images] = i


def create_groups_reverse_tree_smart():
    global group_lists, image_to_group
    local_run_index = data.run_index
    set_status("(2/4) Creating basic groups.", progress_max=(image_count - conf.target_groups))
    data.set_smooth_time(20)
    group_hg_sums = np.hstack([np.vstack(image_DB.means.values), np.vstack(image_DB.hg.values), [[1]] * image_count])
    group_hg_means = group_hg_sums.copy()
    group_lists = [[i] for i in range(image_count)]

    while len(group_hg_sums) > conf.target_groups:
        if local_run_index != data.run_index: return
        set_status(progress_value=(image_count - len(group_hg_sums)))
        QApplication.processEvents()

        joining_group = np.argmin(group_hg_sums[:, -1])
        joining_group_hg = group_hg_means[joining_group]

        if len(group_hg_sums) > 160:
            mean_diffs = abs(group_hg_means[:, 0] - joining_group_hg[0])
            closest_groups = np.argpartition(mean_diffs, 100)[:100]
        else:
            closest_groups = np.arange(len(group_hg_sums))
        closest_groups = closest_groups[closest_groups != joining_group]

        diff_to_others = np.square(group_hg_means[closest_groups] - joining_group_hg).sum(axis=1)
        closest_group = closest_groups[np.argmin(diff_to_others)]

        groups_sum = group_hg_sums[[joining_group, closest_group]].sum(axis=0)
        group_hg_sums[joining_group] = groups_sum
        group_hg_means[joining_group] = groups_sum // groups_sum[-1]
        group_lists[joining_group] = group_lists[joining_group] + group_lists[closest_group]

        group_hg_sums = np.delete(group_hg_sums, closest_group, axis=0)
        group_hg_means = np.delete(group_hg_means, closest_group, axis=0)
        del group_lists[closest_group]

    image_to_group = -np.ones(image_count, dtype=int)
    for i, images in enumerate(group_lists):
        image_to_group[images] = i


def optimize_groups_kmeans9():
    data.set_smooth_time(10)
    iterations = 80
    moved_images = 0
    checked_per_iter, moved_per_iter, moved_per_split_join, join_split_gain = 0, 0, 0, 0
    all_groups_list = list(range(conf.target_groups))
    im_hg_arr = np.hstack([np.vstack(image_DB.hg.values).astype(np.int64), [[1]] * image_count])
    dist_to_group = -np.ones(image_count)
    active_images = np.ones((image_count, conf.target_groups), dtype=bool)
    group_hg_sums = np.ones((conf.target_groups, im_hg_arr.shape[1]), dtype=np.int64)
    group_hg_means = group_hg_sums.copy()
    group_internal_dist = pd.Series(0, index=all_groups_list)
    check_limits = []
    split_cache = dict()
    split_cache_size = 1000
    images_in_check = image_count
    groups_in_check = conf.target_groups
    check_limit_stage_movement = 0
    while True:
        check_limits.append([max(images_in_check, 30), groups_in_check])
        images_in_check = images_in_check // 5 + 1
        groups_in_check = restrict(conf.target_groups * images_in_check // image_count + 2, 4, conf.target_groups)
        if images_in_check < 20: break
    check_limit_stage = len(check_limits) - 1
    out_log = PrettyTable()
    out_log.field_names = ["Iter", "J,S moved", "Join-split gain", "Im mv|chk|to_chk",
                           "grp chk|all", "total mv", "  Total distance  "]
    out_log.align["  Total distance  "] = "r"
    out_log.align["Join-split gain"] = "r"

    def print_status(end_line=False):
        out_log.clear_rows()
        out_log.add_row([itr + 1, moved_per_split_join, f"{join_split_gain:_}",
                         f"{moved_per_iter}|{checked_per_iter}|{images_in_check}",
                         f"{groups_in_check}|{conf.target_groups}",
                         f"{moved_images + moved_per_iter + moved_per_split_join}",
                         f"{int(group_internal_dist.sum()):_}"])
        print(out_log.get_string().splitlines()[-2], end='\n' if end_line else "\r")

    def get_group_score_from_list(im_list):
        group_im_hgs = im_hg_arr[im_list]
        group_hg_s = group_im_hgs.sum(axis=0)
        group_hg_m = np.floor_divide(group_hg_s, group_hg_s[-1])
        local_distances = np.square(group_im_hgs - group_hg_m).sum(axis=1)
        return group_hg_s, group_hg_m, local_distances

    def get_updated_group_hg(group_id, im_id, do_remove=False):
        im_list = group_lists[group_id].copy()
        add_hg = im_hg_arr[im_id]
        if do_remove:
            add_hg = -add_hg
            im_list.remove(im_id)
        else:
            im_list.append(im_id)
        group_im_hgs = im_hg_arr[im_list]
        group_hg_s = group_hg_sums[group_id] + add_hg
        group_hg_m = np.floor_divide(group_hg_s, group_hg_s[-1])
        local_distances = np.square(group_im_hgs - group_hg_m).sum(axis=1)
        return group_hg_s, group_hg_m, local_distances, im_list

    def find_best_groups(im_id, im_group_current):
        if len(group_lists[im_group_current]) <= 2:
            return []
        if groups_in_check < conf.target_groups:
            dist_to_groups = np.square(group_hg_means - im_hg_arr[im_id]).sum(axis=1)
            potential_groups = np.argpartition(dist_to_groups, groups_in_check)[:groups_in_check]
        elif check_limit_stage == 0:
            potential_groups = active_images[im_id].nonzero()[0]
        else:
            potential_groups = all_groups_list
        active_images[im_id, potential_groups] = False
        potential_groups = set(potential_groups)
        potential_groups.discard(im_group_current)
        return potential_groups

    def update_stats():
        for group_id in all_groups_list:
            im_list = group_lists[group_id]
            group_hg_sums[group_id], group_hg_means[group_id], local_dist = get_group_score_from_list(im_list)
            dist_to_group[im_list] = local_dist
            group_internal_dist[group_id] = local_dist.sum()

    def reactivate_groups(group_ids):
        for group_id in group_ids:
            active_images[group_lists[group_id]] = True
            active_images[:, group_id] = True
            split_cache.pop(group_id, None)

    def find_best_joins(leave_group,  best_split_gain):
        group_ids = [i for i in all_groups_list if i != leave_group]
        group_id_combinations = itertools.combinations(group_ids, 2)
        diff_db = dict()
        for group_1, group_2 in group_id_combinations:
            groups_dist = np.square(group_hg_means[group_1] - group_hg_means[group_2]).sum()
            diff_db[groups_dist] = group_1, group_2
        best_pairs = sorted(diff_db)[:groups_in_check * 20 + 25]
        best_diff, best_data = None, None
        for i, group_pair in enumerate(best_pairs):
            set_status(status_text + " Searching join.", progress_value=i, progress_max=len(best_pairs), muted=True)
            QApplication.processEvents()
            group_1, group_2 = diff_db[group_pair]
            im_list = group_lists[group_1] + group_lists[group_2]
            join_data = get_group_score_from_list(im_list)
            dist_diff = join_data[2].sum() - group_internal_dist[group_1] - group_internal_dist[group_2]
            if best_diff is None or best_diff > dist_diff:
                best_diff = dist_diff
                best_data = group_1, group_2, dist_diff, im_list, join_data
            print(f'Join {i}/{len(best_pairs)}, best join gain {best_diff}, best split '
                  f'gain {best_split_gain}, sum {best_split_gain + best_diff}', end="\r")
        return best_data

    def find_best_splits():
        def split_by_arg_list(im_list, im_distances, best_dist, best_split):
            for split_line in range(2, len(im_list) - 1):
                group_1_list_v, group_2_list_v = im_distances[:split_line], im_distances[split_line:]
                group_1_list = [im_list[i] for i in group_1_list_v]
                group_2_list = [im_list[i] for i in group_2_list_v]
                group_1_split = get_group_score_from_list(group_1_list)
                group_2_split = get_group_score_from_list(group_2_list)
                sum_dist = group_1_split[2].sum() + group_2_split[2].sum()
                if best_dist is None or best_dist > sum_dist:
                    best_dist = sum_dist
                    best_split = group_1_list, group_2_list, group_1_split, group_2_split
            return best_dist, best_split

        def find_best_split(group_id):
            im_list = group_lists[group_id]
            if len(im_list) < 4:
                return None

            best_dist, best_split = None, None
            im_distances = np.argsort(dist_to_group[im_list]).tolist()
            best_dist, best_split = split_by_arg_list(im_list, im_distances, best_dist, best_split)

            farthest_im_id = np.argmax(dist_to_group[im_list])
            group_im_hgs = im_hg_arr[im_list]
            local_distances = np.square(group_im_hgs - group_im_hgs[farthest_im_id]).sum(axis=1)
            im_distances = np.argsort(local_distances).tolist()
            best_dist, best_split = split_by_arg_list(im_list, im_distances, best_dist, best_split)

            return best_dist - group_internal_dist[group_id], best_split

        def get_best_split(group_id):
            best_split = split_cache.get(group_id, None)
            if best_split is None:
                if len(split_cache) > split_cache_size:
                    split_cache.pop(next(iter(split_cache)))
                best_split = find_best_split(group_id)
                split_cache[group_id] = best_split
            return best_split

        data.set_smooth_time(10)
        best_dist, best_split_group, split_data = None, None, None
        split_candidates = group_internal_dist.sort_values(ascending=False)[:groups_in_check].index
        for i, split_group in enumerate(split_candidates):
            set_status(status_text + " Searching split.", progress_value=i, progress_max=groups_in_check, muted=True)
            QApplication.processEvents()
            split_result = get_best_split(split_group)
            print(f'Split {i}/{len(split_candidates)}, best gain {best_dist}', end="\r")
            if split_result is None: continue
            if best_dist is None or best_dist > split_result[0]:
                best_dist, split_data = split_result
                best_split_group = split_group
        return best_split_group, best_dist, split_data

    def try_split_join():
        split_group, split_diff, split_data_pack = find_best_splits()
        split_1_list, split_2_list, split_1_data, split_2_data = split_data_pack
        join_group_1, join_group_2, join_diff, join_list, join_data = find_best_joins(split_group, split_diff)
        moved_files = len(group_lists[join_group_2]) + len(split_1_list)
        gain = split_diff + join_diff
        if gain < 0:
            three_groups = [join_group_1, join_group_2, split_group]
            image_to_group[join_list] = join_group_1
            image_to_group[split_1_list] = join_group_2
            group_lists[join_group_1] = join_list
            group_lists[join_group_2] = split_1_list
            group_lists[split_group] = split_2_list
            group_internal_dist[three_groups] = [join_data[2].sum(), split_1_data[2].sum(), split_2_data[2].sum()]
            dist_to_group[join_list] = join_data[2]
            dist_to_group[split_1_list] = split_1_data[2]
            dist_to_group[split_2_list] = split_2_data[2]
            group_hg_sums[join_group_1], group_hg_means[join_group_1] = join_data[:2]
            group_hg_sums[join_group_2], group_hg_means[join_group_2] = split_1_data[:2]
            group_hg_sums[split_group], group_hg_means[split_group] = split_2_data[:2]
            reactivate_groups(three_groups)
            return moved_files, gain
        data.set_smooth_time(10)
        return 0, gain

    update_stats()
    print(f"Starting groups optimization. Initial total distance {int(group_internal_dist.sum()):_}")
    for x in out_log.get_string().splitlines()[:3]:
        print(x)

    for itr in range(iterations):
        moved_per_iter, moved_per_split_join = 0, 0
        status_text = f"(5/5) Optimizing groups, iteration {itr + 1}."
        moved_per_split_join, join_split_gain = try_split_join()
        if moved_per_split_join:
            check_limit_stage_movement = min(check_limit_stage_movement + 1, 2)
            check_limit_stage = min(len(check_limits) - 1, check_limit_stage + check_limit_stage_movement)
        else:
            check_limit_stage_movement = 0
        images_in_check, groups_in_check = check_limits[check_limit_stage]
        if itr == 0: images_in_check = image_count
        if images_in_check < image_count:
            try_images = np.argpartition(dist_to_group, image_count - images_in_check)[-images_in_check:]
        else:
            try_images = range(image_count)
        for i, im_id in enumerate(try_images):
            set_status(status_text, muted=True, progress_value=i + 1, progress_max=images_in_check)
            QApplication.processEvents()
            im_group_old = image_to_group[im_id]
            potential_groups = find_best_groups(im_id, im_group_old)
            if not len(potential_groups): continue
            im_group_old_data = get_updated_group_hg(im_group_old, im_id, True)
            im_group_old_dist = im_group_old_data[2].sum()
            im_group_new_dist, im_group_new, im_group_new_data = None, None, None
            for im_group_try in potential_groups:
                im_group_new_data_try = get_updated_group_hg(im_group_try, im_id)
                if im_group_new_dist is None or im_group_new_dist > im_group_new_data_try[2].sum():
                    im_group_new_dist, im_group_new = im_group_new_data_try[2].sum(), im_group_try
                    im_group_new_data = im_group_new_data_try
            if im_group_old_dist + im_group_new_dist < group_internal_dist[[im_group_old, im_group_new]].sum():
                moved_per_iter += 1
                image_to_group[im_id] = im_group_new
                group_lists[im_group_old] = im_group_old_data[3]
                group_lists[im_group_new] = im_group_new_data[3]
                group_hg_sums[im_group_old], group_hg_means[im_group_old] = im_group_old_data[:2]
                group_hg_sums[im_group_new], group_hg_means[im_group_new] = im_group_new_data[:2]
                group_internal_dist[im_group_old] = im_group_old_data[2].sum()
                group_internal_dist[im_group_new] = im_group_new_data[2].sum()
                dist_to_group[im_group_old_data[3]] = im_group_old_data[2]
                dist_to_group[im_group_new_data[3]] = im_group_new_data[2]
                reactivate_groups([im_group_old, im_group_new])
            checked_per_iter = i + 1
            print_status()

        print_status(True)
        moved_images += moved_per_iter + moved_per_split_join
        data.set_smooth_time(10)
        if check_limit_stage:
            check_limit_stage -= 1
        elif not (moved_per_iter + moved_per_split_join):
            print(out_log.get_string().splitlines()[0])
            break

    if conf.final_sort == Constants.final_sort_by_color:
        sort_rule = lambda x: np.argmax(image_DB.hsv_hg[x].mean()[:256]), False
    elif conf.final_sort == Constants.final_sort_by_brightness_asc:
        sort_rule = lambda x: image_DB.means[x].mean(), False
    elif conf.final_sort == Constants.final_sort_by_brightness_desc:
        sort_rule = lambda x: image_DB.means[x].mean(), True
    elif conf.final_sort == Constants.final_sort_by_file_count_asc:
        sort_rule = lambda x: len(x), False
    elif conf.final_sort == Constants.final_sort_by_file_count_desc:
        sort_rule = lambda x: len(x), True
    else:
        sort_rule = None, False
    group_lists.sort(key=sort_rule[0], reverse=sort_rule[1])


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


def choose_destination_folder():
    base_folder = conf.start_folder
    destination_folder_index = -1
    digits = "[0123456789]"
    if fnmatch.fnmatch(conf.start_folder, "*_sorted_" + digits * 3):
        base_folder = conf.start_folder[:-11]
    elif fnmatch.fnmatch(conf.start_folder, "*_sorted"):
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

    data.new_folder = base_folder + "_sorted" + destination_folder_index_suffix
    data.new_vector_folder = base_folder + "_histograms" + destination_folder_index_suffix


def move_files_pd():
    local_run_index = data.run_index
    status_text = "Sorting done. " + ["Copying ", "Moving "][conf.move_files_not_copy] + "started"
    actual_groups = len(group_lists)
    set_status(status_text, 0, actual_groups)
    QApplication.processEvents()

    new_folder_digits = int(math.log(conf.target_groups, 10)) + 1
    action = shutil.move if conf.move_files_not_copy else shutil.copy
    try:
        os.makedirs(data.new_folder)
    except Exception as e:
        print("Could not create folder ", e)

    ungroupables = []
    for group_index, grp in enumerate(group_lists):
        if len(grp) >= conf.folder_size_min_files:
            dir_name = get_group_color_name(group_index)
            dir_name = data.new_folder + f"{group_index + 1:0{new_folder_digits}d} - ({len(grp):03d}){dir_name}"

            try:
                os.makedirs(dir_name)
            except Exception as e:
                print("Could not create folder ", e)
            for i in grp:
                im_path = image_DB.image_paths.iloc[i]
                try:
                    action(im_path, dir_name + "/")
                except Exception as e:
                    print("Could not complete file operation ", e)

                if local_run_index != data.run_index: return
                QApplication.processEvents()
        else:
            ungroupables += grp.values.tolist()
        set_status(status_text, group_index, actual_groups, muted=True)

    if len(ungroupables) > 0:
        files_count = len(ungroupables)
        dir_name = data.new_folder + f"_ungroupped - ({files_count:03d})"
        try:
            os.makedirs(dir_name)
        except Exception as e:
            print("Could not create folder", e)
        for i in ungroupables:
            im_path = image_DB.image_paths.iloc[i]
            try:
                action(im_path, dir_name)
            except Exception as e:
                print("Could not complete file operation ", e)
            if local_run_index != data.run_index: return
            QApplication.processEvents()
    if conf.move_files_not_copy:
        conf.start_folder = data.new_folder


def get_group_color_name(group_n):
    if data.HSV_color_codes is None:
        data.HSV_color_codes = np.vstack([mc.rgb_to_hsv(mc.to_rgb(i)) for i in mc.XKCD_COLORS.values()])
        data.HSV_color_names = [i[5:].replace("/", "-") for i in mc.XKCD_COLORS.keys()]

    mean_hsv = image_DB.loc[group_lists[group_n], "hsv_hg"].values.mean()
    group_color_one = np.zeros(3)
    for i in range(3):
        one_channel = mean_hsv[i * 256: (i + 1) * 256]
        median_value = np.percentile(one_channel, 70)
        filtered_channel = np.where(one_channel > median_value, one_channel, 0)
        group_color_one[i] = ndimage.center_of_mass(filtered_channel)[0] / 256

    color_diffs = np.square(group_color_one - data.HSV_color_codes).sum(axis=1)
    best_color_id = np.argmin(color_diffs)
    return " " + data.HSV_color_names[best_color_id]


# def export_image_vectors():
#     local_run_index = data.run_index
#
#     try:
#         os.makedirs(data.new_vector_folder)
#     except Exception as e:
#         print("Folder already exists", e)
#
#     save_db = [row[1] for row in group_db]
#     np.savetxt(data.new_vector_folder + "Groups.csv", save_db, fmt='%1.4f', delimiter=";")
#
#     for group_n in range(groups_count):
#         save_db = []
#         for image_index in group_db[group_n][0]:
#             save_db.append(
#                 [image_DB[image_index][0][0]] + image_DB[image_index][1] + [image_DB[image_index][2]])
#         np.savetxt(data.new_vector_folder + "Group_vectors_" + str(group_n + 1) + ".csv",
#                    save_db, delimiter=";", fmt='%s', newline='\n')
#         if local_run_index != data.run_index:
#             return


def get_dominant_image(work_pair):
    crop_compare = distance_db.crop_compare_category.iloc[work_pair]
    if crop_compare and crop_compare < 0:
        return True

    size_compare = distance_db.size_compare_category.iloc[work_pair]
    if size_compare and size_compare > 0:
        return True
    return False


def get_image_pair(work_pair):
    id_l = distance_db.im_1.iloc[work_pair]
    id_r = distance_db.im_2.iloc[work_pair]
    if get_dominant_image(work_pair):
        return id_r, id_l
    else:
        return id_l, id_r


def get_image_pair_crops(work_pair):
    crops = distance_db.crops.iloc[work_pair]
    if type(crops) is not list:
        return None, None

    if get_dominant_image(work_pair):
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
    image_delete_candidates = {}
    image_delete_final = set()
    size_comparison = 0
    list_filtered = -2
    original_distance_db = None

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.label_percent.mousePressEvent = self.central_label_click

        self.paint_sheet = PaintSheet(self)
        self.paint_sheet.setSizePolicy(QtWidgets.QSizePolicy(*[QtWidgets.QSizePolicy.Expanding] * 2))
        self.thumb_sheet_scene = ThumbSheet_Scene(self)
        self.thumb_sheet_scene.setMaximumHeight(155)
        self.thumb_sheet_scene.setMinimumHeight(155)
        self.thumb_sheet_scene.view.mark = self.mark_pair_visual
        self.thumb_sheet_scene.view.show_pair = self.show_pair

        self.paint_sheet.thumb_sheet = self.thumb_sheet_scene

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
        self.frame_img.layout().addWidget(self.thumb_sheet_scene)
        self.show_pair()
        self.animate_timer = QtCore.QTimer()
        self.animate_timer.timeout.connect(self.redraw_animation)
        self.animate_timer.setInterval(800)
        self.switch_suggested(True)

        self.label_name_r.mousePressEvent = self.right_label_click
        self.label_name_l.mousePressEvent = self.left_label_click
        self.thumb_sheet_scene.pair_count = len(distance_db)

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

        self.thumb_sheet_scene.clear_scene(len(distance_db))
        self.adjust_suggestions_block()
        self.reset_thumbs(True)
        self.goto_first_pair()

    def get_filtered_pair(self, distance_db_row):
        size_comparison = distance_db_row.size_compare_category
        crop_comparison = distance_db_row.crop_compare_category

        if self.list_filtered == -3:   return crop_comparison == 0 == size_comparison
        elif self.list_filtered == -4: return crop_comparison * size_comparison < 0
        elif self.list_filtered == -5: return crop_comparison != 0 and size_comparison == 0
        elif self.list_filtered == -6: return crop_comparison == 0 and size_comparison != 0
        elif self.list_filtered == -7: return crop_comparison * size_comparison > 0
        return False

    def adjust_suggestions_block(self):
        filter_settings = {
            -2: [2, 1, 1, 1, 1],
            -3: [0, 0, 0, 2, 1],
            -4: [2, 1, 1, 1, 1],
            -5: [0, 2, 0, 1, 1],
            -6: [0, 0, 2, 1, 1],
            -7: [0, 1, 2, 1, 1],
        }

        enabled_buttons = filter_settings[self.list_filtered]
        for button, option in zip(self.suggest_buttons.buttons(), enabled_buttons):
            button.setEnabled(option)
            if option > 1:
                button.click()

    def switch_suggested(self, button_id):
        self.suggest_mode = self.suggest_buttons.checkedId()
        self.update_all_badges()
        self.show_pair()

    def clear_marks(self):
        distance_db.del_im_1 = 0
        distance_db.del_im_2 = 0
        self.push_apply_marked.setEnabled(False)
        self.push_mark_clear.setEnabled(False)
        self.image_delete_candidates.clear()
        self.update_all_badges()
        self.update_central_lbl()

    def mark_suggested_pairs(self):
        for work_pair in range(self.current_pair + 1):
            im_1_mark, im_2_mark = self.get_delete_suggest(work_pair)
            self.mark_pair_in_db(work_pair, im_1_mark, im_2_mark, False)
        self.update_all_badges()
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

        self.update_all_badges()
        self.update_central_lbl()

    def resort_pairs(self):
        global distance_db
        set_status("(4/4) Resorting pairs...", 0, len(distance_db))
        local_run_index = data.run_index
        data.set_smooth_time(50)
        thumb_cache = {}
        thumb_cache_size = 1000
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
                if len(thumb_cache) > thumb_cache_size:
                    thumb_cache.pop(next(iter(thumb_cache)))
                img = load_image(image_DB.image_paths[im_id])
                img = img.convert("RGB").resize((480, 480), resample=Image.Resampling.BILINEAR)
                thumb_cache[im_id] = img
            return img

        def get_best_crop2(im_id_l, im_id_r, res_id, crops, best_dist=None):
            im_res = resolutions[res_id]

            def get_resized_image_thumb(im_id, width, height, res_id):
                img = resized_thumb_cache.get((im_id, width, height), None)
                if img is None:
                    if res_id < len(resolutions) - 1:
                        im_res2 = resolutions[res_id]
                        img = get_resized_image_thumb(im_id, im_res2, im_res2, res_id + 1)
                    else:
                        img = get_image_thumb(im_id)
                    img = img.resize((width, height), resample=Image.Resampling.BILINEAR)
                    resized_thumb_cache[(im_id, width, height)] = img
                return img

            def get_cropped_img(im_id, crop_tuple):
                resized_width = im_res + crop_tuple[0] + crop_tuple[2]
                resized_height = im_res + crop_tuple[1] + crop_tuple[3]
                resized_img = get_resized_image_thumb(im_id, resized_width, resized_height, res_id + 1)
                croping_tuple = crop_tuple[0], crop_tuple[1], crop_tuple[0] + im_res, crop_tuple[1] + im_res
                return resized_img.crop(croping_tuple)

            image_l_c = get_cropped_img(im_id_l, crops[:4])
            image_r_c = get_cropped_img(im_id_r, crops[4:])

            if best_dist is None:
                best_dist = la_norm(np.asarray(image_l_c).astype(int) - np.asarray(image_r_c))
            if sum(crops) > im_res * .7:
                return best_dist, crops

            def make_crops_row2(im_id, sub_crops, image_2_raw):
                crops_list = []
                for crop_part in range(4):
                    crop_tuple_plus = sub_crops.copy()
                    crop_tuple_plus[crop_part] += 1
                    diff = ImageChops.difference(get_cropped_img(im_id, crop_tuple_plus), image_2_raw)
                    crops_list.append(np.ravel(diff))
                return np.stack(crops_list)

            dist1 = la_norm(make_crops_row2(im_id_l, crops[:4], image_r_c), axis=1).tolist()
            dist2 = la_norm(make_crops_row2(im_id_r, crops[4:], image_l_c), axis=1).tolist()
            cropped_distances = dist1 + dist2

            best_crop_dist = min(cropped_distances)
            best_crop_id = cropped_distances.index(best_crop_dist)

            QApplication.processEvents()
            if best_crop_dist > best_dist:
                return best_dist, crops
            else:
                crops[best_crop_id] += 1
                return get_best_crop2(im_id_l, im_id_r, res_id, crops, best_crop_dist)

        def recalc_distance_with_crops(pair_record):
            if local_run_index != data.run_index:
                return
            id_l, id_r = pair_record[:2]

            crops = [0] * 8
            distance = None
            for res_id in range(len(resolutions)):
                crops = [i * 2 - (i > 0) for i in crops]
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

            resized_thumb_cache.clear()
            tasks_passed = pair_record.name + 1
            data.passed_percent = tasks_passed / tasks_count
            text_to_go = f"Resorting pairs... ({tasks_passed}/{tasks_count})."
            set_status(text_to_go, tasks_passed, muted=True)

            QApplication.processEvents()
            return pair_record

        def recalc_distance_no_crops(pair_record):
            if local_run_index != data.run_index:
                return
            id_l, id_r = pair_record[:2]

            # crops = [0] * 8
            # distance = None
            # for res_id in range(len(resolutions)):
            #     crops = [i * 2 - (i > 0) for i in crops]
            #     distance, crops = get_best_crop2(id_l, id_r, res_id, crops)

            # if sum(crops) == 0:
            #     crops = None
            image_l_c = get_image_thumb(id_l)
            image_r_c = get_image_thumb(id_r)

            distance = la_norm(np.asarray(image_l_c).astype(int) - np.asarray(image_r_c))

            pair_record['dist'] = int(distance)
            # pair_record['crops'] = [0] * 8
            # categories = {1: 1, 2: 1.1, 3: 3}
            # if crops:
            #     crop_size_l = (480 - crops[0] - crops[2]) * (480 - crops[1] - crops[3])
            #     crop_size_r = (480 - crops[4] - crops[6]) * (480 - crops[5] - crops[7])
            #     crop_size_compare = crop_size_r / crop_size_l
            #     for category, threshold in categories.items():
            #         if crop_size_compare > threshold:
            #             pair_record.crop_compare_category = - category
            #         if 1 / crop_size_compare > threshold:
            #             pair_record.crop_compare_category = category

            # resized_thumb_cache.clear()
            tasks_passed = pair_record.name + 1
            data.passed_percent = tasks_passed / tasks_count
            text_to_go = f"Resorting pairs... ({tasks_passed}/{tasks_count})."
            set_status(text_to_go, tasks_passed, muted=True)

            QApplication.processEvents()
            return pair_record

        if data.check_crop_levels:
            distance_db['crops'] = None
            distance_db = distance_db.apply(recalc_distance_with_crops, axis=1)
        else:
            distance_db = distance_db.apply(recalc_distance_no_crops, axis=1)

        if local_run_index != data.run_index:
            return

        distance_db.sort_values("dist", inplace=True)
        distance_db.reset_index(drop=True, inplace=True)

        self.push_show_size_and_crop.setEnabled(True)
        self.push_show_crop_only.setEnabled(True)
        self.push_show_mixed_only.setEnabled(True)

        thumb_cache.clear()
        data.set_smooth_time()
        set_status("Resort complete. Showing image pairs.", 0, 100)
        main_win.redraw_dialog()
        self.push_resort_pairs.setEnabled(False)
        QApplication.processEvents()
        self.current_pair = 0
        self.image_delete_candidates.clear()
        self.reset_thumbs(True)
        self.show()

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
        self.thumb_sheet_scene.clear_scene(len(distance_db))

        print(f"Added {len(self.image_delete_final) - old_image_delete_count} images to delete list, "
              f"totaling {len(self.image_delete_final)}")

        self.image_delete_candidates.clear()
        self.current_pair = 0
        self.toggle_filter_pairs()
        self.update_all_badges()
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
            self.thumb_sheet_scene.update_all_thumbs()
        self.thumb_mode = self.mode_buttons.checkedId()
        border_color = QtCore.Qt.GlobalColor.white if self.thumb_mode == -2 else QtCore.Qt.GlobalColor.black
        self.thumb_sheet_scene.box_pen.setColor(border_color)
        self.update_all_badges()
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

    def size_label_text(self, left_or_right: int, im_id=None):
        color_template = """QLabel {
                        border-width: 3px;
                        border-style: solid;
                        border-color: """
        red_gray_green = ("rgb(200, 70, 70);}", "rgb(70, 70, 70);}", "rgb(70, 200, 70);}")

        label = [self.label_size_l, self.label_size_r][left_or_right]
        if not len(distance_db) or im_id is None:
            label.setStyleSheet(color_template + red_gray_green[1])
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

        size_comparison = work_record.size_compare_category
        crop_comparison = work_record.crop_compare_category
        if get_dominant_image(self.current_pair) == (left_or_right == 0):
            size_comparison *= -1
            crop_comparison *= -1

        text += "\n" + size_labels[size_comparison]

        if crop_comparison:
            text += "\n" + crop_labels[crop_comparison]

        crop_list = get_image_pair_crops(self.current_pair)[left_or_right]
        if crop_list and sum(crop_list):
            text += f"\n To crop ({crop_list[0]}, {crop_list[1]}, {crop_list[2]}, {crop_list[3]})"
        label.setText(text)

        color_category = self.get_delete_suggest(self.current_pair, True)[left_or_right] + 1
        label.setStyleSheet(color_template + red_gray_green[color_category])

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
            thm_size = self.thumb_sheet_scene.thumb_size
            image_r = image_r.resize((thm_size, thm_size), Image.Resampling.BILINEAR)
            image_l = image_l.resize((thm_size, thm_size), Image.Resampling.BILINEAR)
        else:
            if image_l.height > image_r.height:
                image_r = image_r.resize(image_l.size, Image.Resampling.BILINEAR)
            elif image_l.height < image_r.height or image_l.width != image_r.width:
                image_l = image_l.resize(image_r.size, Image.Resampling.BILINEAR)

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
            self.paint_sheet.update()
            return

        self.current_pair += shift
        self.current_pair %= len(distance_db)
        self.thumb_sheet_scene.update_selection()
        self.update_all_badges()

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
        self.image_pixmaps[2] = pixmap_from_image(self.image_d)
        self.image_pixmaps[3] = self.image_pixmaps[2]
        if self.thumb_mode == -3:
            self.image_pixmaps[3] = pixmap_from_image(ImageOps.invert(self.image_d))

        self.paint_sheet.pos = None
        self.paint_sheet.img_zoom = 1
        self.paint_sheet.img_xy = QtCore.QPoint(0, 0)
        self.paint_sheet.img_xy = QtCore.QPoint(0, 0)
        self.paint_sheet.suggest_marks = self.get_delete_suggest(self.current_pair, True)

        if self.draw_diff_mode:
            self.redraw_animation()
        else:
            self.paint_sheet.pixmap_l = self.image_pixmaps[0]
            self.paint_sheet.pixmap_r = self.image_pixmaps[1]
            self.paint_sheet.update()

    def request_thumb(self, thumb_id):
        if thumb_id >= len(distance_db):
            return
        # print(f"Pair {thumb_id} stage *")
        id_l, id_r = get_image_pair(thumb_id)
        im_l = image_DB.image_paths[id_l]
        im_r = image_DB.image_paths[id_r]
        crop_l, crop_r = get_image_pair_crops(thumb_id)
        thm_size = self.thumb_sheet_scene.thumb_size - 12
        thumb_mode = self.thumb_mode
        thumb_colored = self.push_colored.isChecked()

        load_options = [im_l, im_r, crop_l, crop_r, thm_size, thumb_mode, thumb_colored, thumb_id]
        pool.apply_async(generate_thumb_pixmap, load_options, callback=recieve_loaded_thumb)

    def update_all_badges(self):
        self.thumb_sheet_scene.badges_version += 1

    def update_single_thumb_badges(self, work_pair):
        mark_left = distance_db.del_im_1.iloc[work_pair]
        mark_right = distance_db.del_im_2.iloc[work_pair]
        id_l = distance_db.im_1.iloc[work_pair]
        id_r = distance_db.im_2.iloc[work_pair]
        fatal_left = id_l in self.image_delete_candidates and mark_left != -1
        fatal_right = id_r in self.image_delete_candidates and mark_right != -1
        if get_dominant_image(work_pair):
            mark_left, mark_right = mark_right, mark_left
            fatal_left, fatal_right = fatal_right, fatal_left
        suggest_left, suggest_right = self.get_delete_suggest(work_pair, True)
        self.thumb_sheet_scene.update_badges(work_pair, [suggest_left, suggest_right,
                                                         fatal_left, fatal_right,
                                                         mark_left, mark_right])

    def get_delete_suggest(self, work_pair, check_flip=False):
        distance_db_row = distance_db.iloc[work_pair]
        size_comparison = distance_db_row.size_compare_category
        crop_comparison = distance_db_row.crop_compare_category
        name_comparison = distance_db_row.name_difference
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
        elif self.suggest_mode == -6:
            if name_comparison < 0:
                return delete_left
            if 0 < name_comparison:
                return delete_right
        return 0, 0

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
        self.font.setPointSize(14)

        self.redraw_timer.timeout.connect(self.redraw_dialog)
        self.redraw_timer.setInterval(50)
        self.drag_timer.timeout.connect(self.drag_timeout)
        self.drag_timer.setInterval(3000)

        self.select_folder_button.clicked.connect(self.select_directory)
        self.select_folder_button.resizeEvent = self.directory_changed
        self.select_folder_button.dragLeaveEvent = self.directory_changed
        self.select_folder_button.dragEnterEvent = self.directory_entered
        self.select_folder_button.dropEvent = self.directory_dropped
        self.directory_changed()

        self.select_folder_button_2.clicked.connect(self.select_directory_2)
        self.select_folder_button_2.resizeEvent = self.directory_changed_2
        self.select_folder_button_2.dragLeaveEvent = self.directory_changed_2
        self.select_folder_button_2.dragEnterEvent = self.directory_entered_2
        self.select_folder_button_2.dropEvent = self.directory_dropped_2
        self.directory_changed_2()

        self.slider_histogram_bands.valueChanged.connect(self.slider_histogram_bands_changed)
        self.slider_histogram_bands_changed()

        self.slider_image_split_steps.valueChanged.connect(self.slider_image_split_steps_changed)
        self.slider_image_split_steps_changed()

        self.btn_show_compare_wnd.clicked.connect(compare_wnd_toggle)

        self.slider_max_pairs.valueChanged.connect(self.slider_max_pairs_changed)
        self.slider_max_pairs_changed()

        self.slider_enforce_equal.valueChanged.connect(self.slider_equal_changed)
        self.slider_equal_changed()

        self.list_color_spaces.itemSelectionChanged.connect(self.color_spaces_reselected)
        self.color_spaces_reselected()
        # self.tabWidget.currentChanged.connect(self.duplicate_tab_switched)

        self.tab_unduplicate.hideEvent = lambda x: self.check_two_image_sets.setChecked(False)

        self.check_two_image_sets.stateChanged.connect(self.two_sets_toggle)

        list_color_spaces = [self.list_color_spaces_CMYK, self.list_color_spaces_HSV,
                             self.list_color_spaces_RGB, self.list_color_spaces_YCbCr]
        for c_list in list_color_spaces:
            [c_list.item(i).setSelected(True) for i in range(3)]
        [self.list_color_spaces.item(i).setSelected(True) for i in [0, 1]]

        self.btn_stop.clicked.connect(self.stop_button_pressed)
        self.btn_start.clicked.connect(self.start_button_pressed)
        self.progressBar.setVisible(False)
        self.select_folder_button_2.setVisible(False)
        # self.progressBar.valueChanged.connect(self.update)
        self.enable_elements()
        # self.duplicate_tab_switched()
        self.btn_show_compare_wnd.setVisible(False)

        if len(sys.argv) > 1:
            if os.path.isdir(sys.argv[1]):
                conf.start_folder = sys.argv[1]
                self.directory_changed()

        for obj in [self.groupBox_6, self.frame_7, self.frame_3, self.group_final]:
            obj.setEnabled(False)

    def start_button_pressed(self):
        # global move_files_not_copy
        # global show_histograms
        global export_histograms
        # global start_folder
        # global new_folder
        # global create_samples
        # global final_sort

        local_run_index = data.run_index

        conf.folder_constraint_type = self.combo_folder_constraints.currentIndex()
        conf.folder_constraint_value = self.spin_num_constraint.value()
        conf.search_subdirectories = self.check_subdirs_box.isChecked()
        conf.ignore_ext = self.check_ignore_ext.isChecked()
        conf.move_files_not_copy = self.radio_move.isChecked()
        # show_histograms = self.check_show_histograms.isChecked()
        conf.export_histograms = self.check_export_histograms.isChecked()
        conf.create_samples = self.check_create_samples.isChecked()
        # conf.enable_stage2_grouping = self.check_stage2_grouping.isChecked()
        # conf.stage1_grouping_type = self.combo_stage1_grouping.currentIndex()
        conf.search_duplicates = self.tab_unduplicate.isVisible()
        # conf.compare_by_angles = self.btn_angles.isChecked()
        conf.compare_resort = self.btn_compare_resort.isChecked()

        conf.final_sort = self.combo_final_sort.currentIndex() + Constants.final_sort_by_color
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
        if conf.search_duplicates:
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

        # self.duplicate_tab_switched()

    def two_sets_toggle(self):
        self.select_folder_button_2.setVisible(self.check_two_image_sets.isChecked())
        if not self.check_two_image_sets.isChecked():
            conf.start_folder_2 = ""

    # def duplicate_tab_switched(self):
    #     grouping = self.tab_grouping.isVisible()
    #     # self.group_move.setEnabled(grouping)
    #     # self.group_final.setEnabled(grouping)
    #     # self.groupBox_5.setEnabled(grouping)

    def slider_histogram_bands_changed(self):
        conf.hg_bands_count = self.slider_histogram_bands.value()
        self.lbl_histogram_bands.setText(str(conf.hg_bands_count))

    def slider_image_split_steps_changed(self):
        raw_splits = self.slider_image_split_steps.value()
        raw_split_steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30]
        conf.image_divisions = raw_split_steps[raw_splits]
        self.lbl_image_split_steps.setText(str(conf.image_divisions ** 2))

    def slider_equal_changed(self):
        conf.enforce_equal_folders = (self.slider_enforce_equal.value() / 100) ** 2
        self.lbl_slider_enforce_equal.setText("%d" % (self.slider_enforce_equal.value() / 3) + "%")

    def slider_max_pairs_changed(self):
        raw_slider = self.slider_max_pairs.value()
        conf.max_pairs = int(10 * 3 ** (0.06 * raw_slider))
        self.lbl_max_pairs.setText(f"{conf.max_pairs}")
        # if raw_slider == 200:
        #     conf.max_pairs = 0
        #     self.lbl_max_pairs.setText("all")

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

    def directory_entered_2(self, input_object):
        if input_object.mimeData().hasUrls():
            url_text = unquote(input_object.mimeData().text())
            firt_path = url_text[8:]
            firt_path_start = url_text[:8]

            if firt_path_start == "file:///":
                if os.path.isdir(firt_path):
                    self.select_folder_button.setText("Drop folder here")
                    input_object.accept()
                    return
        self.select_folder_button_2.setText("Only directory accepted")
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

    def directory_dropped_2(self, in_dir):
        url_text = unquote(in_dir.mimeData().text())
        firt_path = url_text[8:]
        firt_path_start = url_text[:8]
        if firt_path_start == "file:///":
            if os.path.isdir(firt_path):
                conf.start_folder_2 = firt_path
                self.directory_changed_2()

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

    def directory_changed_2(self, suppress_text=False):
        if os.path.exists(conf.start_folder_2):
            conf.start_folder_2 = os.path.normpath(conf.start_folder_2)
            if not suppress_text:
                set_status("Ready to start")

            start_folder_parts = [a + "\\" for a in conf.start_folder_2.split("\\")]
            start_folder_parts2 = [(a + " ") if a[-1] != "\\" else a for b in start_folder_parts for a in b.split(" ")]
            line_length = 0
            lines_count = max(round(len(conf.start_folder_2) / 35), 1)
            line_max = max(len(conf.start_folder_2) // lines_count, 20)
            button_text = ""

            for part_path in start_folder_parts2:
                extended_line_length = line_length + len(part_path)
                if extended_line_length > line_max:
                    button_text += "\n" + part_path
                    line_length = len(part_path)
                else:
                    button_text += part_path
                    line_length += len(part_path)

            self.select_folder_button_2.setText(button_text)
            self.btn_start.setEnabled(True)
        else:
            if not suppress_text:
                set_status("Please select folder")
            self.select_folder_button_2.setText("Select folder")
            self.btn_start.setEnabled(False)

    def select_directory(self):
        conf.start_folder = QFileDialog.getExistingDirectory(self, "Choose directory", conf.start_folder or "Y:/")
        self.directory_changed()

    def select_directory_2(self):
        conf.start_folder_2 = QFileDialog.getExistingDirectory(self, "Choose directory", conf.start_folder_2 or "Y:/")
        self.directory_changed_2()

    def redraw_dialog(self):
        if self.progressBar.maximum() != data.progress_max:
            self.progressBar.setRange(0, data.progress_max)
        self.progressBar.setValue(int(data.progress_value))
        if not data.progress_value:
            return
        text_to_go, text_end_time, text_to_go2, text_end_time2 = "", "", "", ""
        step_one_time, step_one_step = data.steps_buffer[0]
        last_time, last_step = data.steps_buffer[-1]
        if (data.last_iteration_progress_value != data.progress_value) and time() - last_time > 3:
            data.last_iteration_progress_value = data.progress_value
            data.steps_buffer.pop(0)
            data.steps_buffer.append((time(), data.progress_value))

        if step_one_time > 0 and (data.progress_value - step_one_step):
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
        # QApplication.processEvents()


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
