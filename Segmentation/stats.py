import os

folders = ["checkpoints_nshot_debug_VOS_new_500crop0.6new_0.7_twice_fixresize_testcro",
           "checkpoints_nshot_debug_VOS_new_500crop0.6new_0.7_once_noreverse",
    "checkpoints_nshot_debug_vos_crop0.6new_0.5_test_fixresize_testcrop",
           "checkpoints_nshot_debug_vos_crop0.6new_0.6_test_fixresize_testcrop",
"checkpoints_nshot_debug_vos_crop0.6new_0.7_test_fixresize_testcrop_singlenet",
    "checkpoints_nshot_debug_vos_crop0.6new_0.7_test_fixresize",
           "./checkpoints_nshot_debug_vos_crop0.6new_0.7_test",
           "checkpoints_nshot_debug_vos_crop0.6_0.7_test",
           "checkpoints_nshot_debug_vos_crop0.6_0.6_test",
           "checkpoints_nshot_debug_vos_crop0.6_0.5_test",
           "checkpoints_nshot_debug_vos_crop0.75_0.7_test",
           "checkpoints_nshot_debug_vos_crop0.75_0.6_test"]
cats = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox"]
# cats = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump", "paragliding-launch", "parkour"]


def cal_avg(folder):
    valss = []
    for cat in cats:
        filepath = os.path.join(folder, cat)
        logs = [x for x in os.listdir(filepath)if x.endswith('.txt')]
        filepath = os.path.join(filepath, logs[-1])
        # print(filepath)

        
        with open(filepath) as f:
            lines = f.readlines()

            vals = []
            for line in lines:
                # print(line)
                if line.startswith("Find new max validation acc: "):
                    vals.append(float(line[len("Find new max validation acc: "):].split(' ')[0]))
        val = max(vals) * 100 if len(vals) > 0 else 0.0
        # print(val)
        valss.append(val)
    print()
    print(sum(valss) / len(valss))

for folder in folders:
    print(folder)
    cal_avg(folder)
    # break
