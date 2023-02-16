import os
import json
import numpy as np

def collect_results(path="logs/"):
    all_results = {}
    for fn in os.listdir(path):
        if fn.startswith("log."):
            
            train_proportion = float(fn.split('_')[-1]) * 100
            od = fn.split('od_')[1].split('_num')[0]
            model_name = f"{fn.split('json_')[1].split('_opt')[0]} using {train_proportion}% nodes in {od} dimensions"
            if model_name not in all_results:
                all_results[model_name] = {}
            train_data_name = fn.split(".")[2]
            if train_data_name not in all_results[model_name]:
                all_results[model_name][train_data_name] = {}

            for line in open(f"{path}/{fn}"):
                if "Average Final" in line:
                    test_data_name = line.split("(")[1].split(")")[0]
                    if test_data_name not in all_results[model_name][train_data_name]:
                        all_results[model_name][train_data_name][test_data_name] = {"same_domain":None, "train_ap": [], "valid_ap": [], "test_ap": [], "log_path": []}

                    same_domain = (line.split("(")[2].split(",")[0] == "same domain")
                    valid_ap = float(line.split("=")[1].split(",")[0].strip())
                    test_ap = float(line.split("=")[2].strip())
                    if all_results[model_name][train_data_name][test_data_name]["same_domain"] == None:
                        all_results[model_name][train_data_name][test_data_name]["same_domain"] = same_domain
                    if same_domain != all_results[model_name][train_data_name][test_data_name]["same_domain"]:
                        raise ValueError("same_domain not consistent!")
                    if same_domain:
                        all_results[model_name][train_data_name][test_data_name]["train_ap"].append(valid_ap)
                        all_results[model_name][train_data_name][test_data_name]["valid_ap"].append(test_ap)
                    else:
                        all_results[model_name][train_data_name][test_data_name]["valid_ap"].append(valid_ap)
                        all_results[model_name][train_data_name][test_data_name]["test_ap"].append(test_ap)
                    all_results[model_name][train_data_name][test_data_name]["log_path"].append(fn)
    for model_name in all_results:
        for train_data_name in all_results[model_name]:
            for test_data_name in all_results[model_name][train_data_name]:
                if all_results[model_name][train_data_name][test_data_name]["same_domain"]:
                    all_results[model_name][train_data_name][test_data_name]["train_ap"] = np.amax(all_results[model_name][train_data_name][test_data_name]["train_ap"])
                    max_valid_ind = np.argmax(all_results[model_name][train_data_name][test_data_name]["valid_ap"])
                    all_results[model_name][train_data_name][test_data_name]["valid_ap"] = all_results[model_name][train_data_name][test_data_name]["valid_ap"][max_valid_ind]
                    all_results[model_name][train_data_name][test_data_name]["log_path"] = all_results[model_name][train_data_name][test_data_name]["log_path"][max_valid_ind]

                else:
                    max_ind = np.argmax(all_results[model_name][train_data_name][test_data_name]["valid_ap"])
                    all_results[model_name][train_data_name][test_data_name]["valid_ap"] = all_results[model_name][train_data_name][test_data_name]["valid_ap"][max_ind]
                    all_results[model_name][train_data_name][test_data_name]["test_ap"] = all_results[model_name][train_data_name][test_data_name]["test_ap"][max_ind]
                    all_results[model_name][train_data_name][test_data_name]["log_path"] = all_results[model_name][train_data_name][test_data_name]["log_path"][max_ind]

    return all_results
            

all_results = collect_results("logs/")
open("results.json","w").write(json.dumps(all_results, indent="\t"))

for model_name in all_results:
    train_AP = []
    valid_AP_within_domain = []
    valid_AP_cross_domain = []
    test_AP_cross_domain = []
    for d in all_results[model_name].values():
        for dd in d.values():
            if dd["same_domain"]:
                train_AP.append(dd["train_ap"])
                valid_AP_within_domain.append(dd["valid_ap"])
            else:
                valid_AP_cross_domain.append(dd["valid_ap"])
                test_AP_cross_domain.append(dd["test_ap"])
    print(f"{model_name}, train AP: {np.mean(train_AP):.3f} (std: {np.std(train_AP):.3f})")
    print(f"{model_name}, domain valid AP: {np.mean(valid_AP_within_domain):.3f} (std: {np.std(valid_AP_within_domain):.3f})")
    print(f"{model_name}, cross domain valid AP: {np.mean(valid_AP_cross_domain):.3f} (std: {np.std(valid_AP_cross_domain):.3f})")
    print(f"{model_name}, cross domain test AP: {np.mean(test_AP_cross_domain):.3f} (std: {np.std(test_AP_cross_domain):.3f})")

