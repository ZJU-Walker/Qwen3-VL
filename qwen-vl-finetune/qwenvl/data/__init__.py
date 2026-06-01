import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

# --- Robot subtask (Trossen) tasks ------------------------------------------
# To add a NEW task: copy the entry below, point `annotation_path` at the new
# segment CSV, and register it in `data_dict`. The subtask label set is derived
# automatically from that CSV's `subtask` column -- no code changes elsewhere.
# All entries share `dataset_type: "robot_subtask"`, which routes them to
# RobotSubtaskDataset. Select one at train time with DATASETS=<key>.
#   - annotation_path: segment CSV (dataset,episode_id,start_frame,end_frame,subtask,...)
#   - data_path:       LeRobot data root holding the dataset folder(s) from the CSV
#   - camera:          which camera stream to read windows from
#   - prompt:          task instruction shown to the model (task-specific)
TROSSEN_BLOCK_MEM_0528 = {
    "dataset_type": "robot_subtask",
    "annotation_path": "/iris/projects/humanoid/trossen_data/scripts/labels/subtask_segments_mix_block_0528_auto.csv",
    "data_path": "/iris/projects/humanoid/trossen_data",
    "camera": "observation.images.cam_high",
    "prompt": (
        "Predict the subtask the robot should perform: put the green block to "
        "the plate, or put the yellow block to the plate."
    ),
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    # robot subtask tasks (add new ones here)
    "trossen_block_mem_0528": TROSSEN_BLOCK_MEM_0528,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
