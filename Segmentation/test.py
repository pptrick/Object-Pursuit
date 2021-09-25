from evaluation.seen_obj import test_seen_obj

if __name__ == "__main__":
    test_seen_obj(data_dir='/data/pancy/iThor/single_obj/FloorPlan2_ext/data_FloorPlan2_Bowl_3',
                  z_dir='./checkpoints_random_1_threshold_0.7/zs/', 
                  hypernet_path='./checkpoints_random_1_threshold_0.7/checkpoint/hypernet.pth',
                  backbone_path='./checkpoints_random_1_threshold_0.7/checkpoint/backbone.pth')