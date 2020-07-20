class DataConfig(object):
    data_path = "/home/ec2-user/SageMaker/data"
    data_name = "gray_diff2"
    img_height = 192
    img_width = 256
    num_channels = 4
    
class TrainConfig(DataConfig):
    model_name = "comma_prelu"
    batch_size = 32
    num_epoch = 10
    val_part = 33
    X_train_mean_path = "data/X_train_gray_diff2_mean.npy"
    
class TestConfig(TrainConfig):
    model_path = "checkpoints/final_model.hdf5"
    angle_train_mean = -0.004179079

class VisualizeConfig(object):
    pred_path = "submissions/final.csv"
    true_path = "data/CH2_final_evaluation.csv"
    img_path = "phase2_test/center/*.jpg"
