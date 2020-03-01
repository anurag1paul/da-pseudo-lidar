from utils.config import Config, configs

# data configs
configs.data.classes = ('Car')
configs.data.num_classes = len(configs.data.classes)

# evaluate configs
configs.evaluate = Config()
configs.evaluate.num_tests = 20
configs.evaluate.ground_truth_path = 'data/vkitti/raw_data/'
configs.evaluate.scene = 'Scene20'