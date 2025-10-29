import json

from regex import P

origin = {
    "VCMR-0.5-R1": 0,
    "VCMR-0.7-R1": 0,
    "VR-R10": 0,
    "VR-R100": 0,
    "SVMR-0.5-R1": 0,
    "SVMR-0.7-R1": 0
}

def load_json(path):
    d = json.load(open(path, "r"))
    da = {
    "VCMR-0.5-R1": 0,
    "VCMR-0.7-R1": 0,
    "VR-R10": 0,
    "VR-R100": 0,
    "SVMR-0.5-R1": 0,
    "SVMR-0.7-R1": 0
    }
    da["VCMR-0.5-R1"] = d["VCMR"]["0.5-r1"]
    da["VCMR-0.7-R1"] = d["VCMR"]["0.7-r1"]
    da["VR-R10"] = d["VR"]["r10"]
    da["VR-R100"] = d["VR"]["r100"]
    da["SVMR-0.5-R1"] = d["SVMR"]["0.5-r1"]
    da["SVMR-0.7-R1"] = d["SVMR"]["0.7-r1"]

    return da

original = load_json("/home/test/pengjin/data1/HashVcmr/method_tvr/results/tvr-video_sub_tef-1024-bit-2025_10_26_12_29_45/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")

withglobal = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-withglobal-2025_10_27_00_45_31/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")

wohra = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-wohra-2025_10_27_00_19_13/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")
woglobal = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-woglobal-2025_10_27_00_19_47/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")
wolocal = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-wolocal-2025_10_27_00_20_07/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")

def diff(origin, target):
    da = {
    "VCMR-0.5-R1": 0,
    "VCMR-0.7-R1": 0,
    "VR-R10": 0,
    "VR-R100": 0,
    "SVMR-0.5-R1": 0,
    "SVMR-0.7-R1": 0
    }
    for k in da:
        da[k] = target[k] - origin[k]

    return da

def printf(data):
    for k in data:
        # print(f"{k}: {data[k]:.2f}")
        print(f"& {data[k]:.2f}", end=" ")
    print()

# print("withglobal")
# printf(diff(original, withglobal))

# print("wohra")
# printf(diff(original, wohra))

# print("woglobal")
# printf(diff(original, woglobal))

# print("wolocal")
# printf(diff(original, wolocal))

only = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-onlyglobal-2025_10_27_12_00_14/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")
pointed = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-pointed-2025_10_27_15_21_00/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")
woconv = load_json("HashVcmr/method_tvr/results/tvr-video_sub_tef-woConv-2025_10_27_16_59_58/inference_tvr_val_None_predictions_SVMR_VCMR_VR_metrics.json")

print("onlyglobal")
printf(diff(original, only))
print("pointed")
printf(diff(original, pointed))
print("woconv")
printf(diff(original, woconv))