warmup_epochs = int(train_config['warmup_epochs'])
warmup_factor = 1.0 / train_config['warmup_epochs']

# Learning rate scheduler with warmup
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) * warmup_factor
)
###################

theModel = MyDetectionModel(cfg="deneme.yaml")
theModel.load_pretrained_weights('yolov8m.pt')

#theModel.train()
predictor = DetectionPredictor()
x = predictor(source=the_image ,model=theModel)
x[0].save_txt("res1.txt",True)
x[1].save_txt("res1.txt",True)

###################