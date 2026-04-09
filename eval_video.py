import os
import cv2
from tqdm import tqdm
import numpy as np
from py_sod_metrics import MAE, Emeasure, FmeasureHandler, Smeasure, WeightedFmeasure, IOUHandler, DICEHandler, FmeasureV2

BINARY_CLASSIFICATION_METRIC_MAPPING = {
    "fmeasure": {
        "handler": FmeasureHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True, with_binary=False, beta=0.3),
    },
    "iou": {
        "handler": IOUHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True, with_binary=False),
    },
    "dice": {
        "handler": DICEHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=False, with_binary=False),
    },
}

method='your_method_name' # change method name
for _data_name in ['CAMotion']:
    print("eval-dataset: {}".format(_data_name))
    mask_root = '/your_dataset_path/' # change path
    pred_root = '/your_prediction_path/'# change path
    mask_name_list = sorted(os.listdir(mask_root))
    FMS = FmeasureV2()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    FMS.add_handler(handler_name="fmeasure", metric_handler=BINARY_CLASSIFICATION_METRIC_MAPPING["fmeasure"]["handler"](**BINARY_CLASSIFICATION_METRIC_MAPPING["fmeasure"]["kwargs"]))
    FMS.add_handler(handler_name="iou", metric_handler=BINARY_CLASSIFICATION_METRIC_MAPPING["iou"]["handler"](**BINARY_CLASSIFICATION_METRIC_MAPPING["iou"]["kwargs"]))
    FMS.add_handler(handler_name="dice", metric_handler=BINARY_CLASSIFICATION_METRIC_MAPPING["dice"]["handler"](**BINARY_CLASSIFICATION_METRIC_MAPPING["dice"]["kwargs"]))
    subdirs = [os.path.join(mask_root, d) for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))]
    print(len(subdirs))
    for subdir in subdirs:
        pred_root_1 = os.path.join(pred_root, subdir.split('/')[-1]) 
        print(pred_root_1)
        mask_name_list = [f for f in os.listdir(os.path.join(subdir, "GT")) if f.endswith(('.png', '.jpg', '.jpeg'))]
        mask_name_list = sorted(mask_name_list)
        for mask_name in tqdm(mask_name_list, desc=f"Processing {subdir}", total=len(mask_name_list)):
            mask_path = os.path.join(subdir, "GT", mask_name)
            pred_path = os.path.join(pred_root_1, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)
            FMS.step(pred=pred, gt=mask)
        fm = FMS.get_result()["fmeasure"]
        iou = FMS.get_result()["iou"]
        dice = FMS.get_result()["dice"]
        wfm = WFM.get_result()["wfm"]
        sm = SM.get_result()["sm"]
        em = EM.get_result()["em"]
        mae = M.get_result()["mae"]
        
        frame = len(mask_name_list)
        results = {
            "Smeasure": np.mean(sm[-frame:]),
            "wFmeasure": np.mean(wfm[-frame:]),
            "MAE": np.mean(mae[-frame:]),
            "adpEm": np.mean(em["adp"][-frame:]),
            "meanEm": np.mean(em["curve"][-frame:], axis=0).mean(),
            "maxEm": np.mean(em["curve"][-frame:], axis=0).max(),
            "adpFm": np.mean(fm["adaptive"][-frame:]),
            "meanFm": np.mean(fm["dynamic"][-frame:], axis=0).mean(),
            "maxFm": np.mean(fm["dynamic"][-frame:], axis=0).max(),
            "meanDice": np.mean(dice["dynamic"][-frame:], axis=0).mean(),
            "maxDice": np.mean(dice["dynamic"][-frame:], axis=0).max(),
            "meanIoU": np.mean(iou["dynamic"][-frame:], axis=0).mean(),
            "maxIoU": np.mean(iou["dynamic"][-frame:], axis=0).max(),
        }
        print(results)
        with open("eval_results.txt", "a", encoding="utf-8") as file:
            file.write(subdir.split('/')[-1]+' '+str(results)+'\n')

    fm = FMS.get_results()["fmeasure"]
    iou = FMS.get_results()["iou"]
    dice = FMS.get_results()["dice"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]
    

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adaptive"],
        "meanFm": fm["dynamic"].mean(),
        "maxFm": fm["dynamic"].max(),
        "meanDice": dice["dynamic"].mean(),
        "maxDice": dice["dynamic"].max(),
        "meanIoU": iou["dynamic"].mean(),
        "maxIoU": iou["dynamic"].max(),
    }

    print(results)
    with open("eval_results.txt", "a", encoding="utf-8") as file:
        file.write(f"Overall {results}\n")

print("Eval finished!")