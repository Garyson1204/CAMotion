import os
import cv2
from tqdm import tqdm
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

method='your_method_name' 
for _data_name in ['CAMotion-TE']:
    print("eval-dataset: {}".format(_data_name))
    mask_root = '/your_dataset_path/{}/'.format("GT") # change path
    pred_root = '/your_result_path/'# change path
    mask_name_list = sorted(os.listdir(mask_root))
    FMS = FmeasureV2()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    FMS.add_handler(handler_name="fmeasure", metric_handler=BINARY_CLASSIFICATION_METRIC_MAPPING["fmeasure"]["handler"](**BINARY_CLASSIFICATION_METRIC_MAPPING["fmeasure"]["kwargs"]))
    FMS.add_handler(handler_name="iou", metric_handler=BINARY_CLASSIFICATION_METRIC_MAPPING["iou"]["handler"](**BINARY_CLASSIFICATION_METRIC_MAPPING["iou"]["kwargs"]))
    FMS.add_handler(handler_name="dice", metric_handler=BINARY_CLASSIFICATION_METRIC_MAPPING["dice"]["handler"](**BINARY_CLASSIFICATION_METRIC_MAPPING["dice"]["kwargs"]))
    
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        print(pred_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)
        FMS.step(pred=pred, gt=mask)

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
    file=open("eval_results.txt", "a")
    file.write('Overall '+str(results)+'\n')


print("Eval finished!")