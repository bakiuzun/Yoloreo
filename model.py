import torch
from ultralytics.nn.tasks import *
from utils import parse_my_detection_model


class MyYolo(BaseModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""

        super().__init__()

        self.first_forward = False
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_my_detection_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        first_head = self.model[22]  # Detect()
        second_head = self.model[-1]  # Detect()
        if isinstance(first_head, (Detect, Segment, Pose)) and isinstance(second_head, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            first_head.inplace = self.inplace
            second_head.inplace = self.inplace

            #forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            #m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            first_head.stride = torch.tensor([ 8., 16., 32.])
            second_head.stride = torch.tensor([ 8., 16., 32.])
            self.stride = first_head.stride

            first_head.bias_init()  # only run once
            second_head.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

        self.head_1 = self.model[10:23]
        self.backbone = self.model[:10]
        self.head_2 = self.model[23:]
    """
    method used to build the stride
    explanation: the super class call one forward with a lambda x to calculate the strides of the network
    file: ultralytics/nn/task.py
    line: 246
    """
    def _build_stride(self, x, profile=False, visualize=False):
        self.first_forward = False
        x = super()._predict_once(x, profile, visualize)
        return x

    def _predict_once(self, x, profile=False, visualize=False):

        stereo = True
        ## refer to _build_stride comment
        ## will not enter here now debugging...
        if self.first_forward:
            return self._build_stride(x,profile,visualize)

        y, dt = [], []  # outputs

        y_1 = []
        y_2 = []
        x_1 = x[:,0]
        x_2 = x[:,1]
        
        if torch.isinf(x_2[0,0,0,0]):
            stereo = False
            x_2 = x_1.clone()
            
            
        x_1,y_1 = self._forward_backbone(x_1,y_1)
        x_2,y_2 = self._forward_backbone(x_2,y_2)

        attended_feature_2 =  self._cross_attention(x_1,x_2)
        attended_feature_1 =  self._cross_attention(x_2,x_1)

        x_1,y_1 = self._forward_head(attended_feature_1,y_1,head1=True)
        x_2,y_2 = self._forward_head(attended_feature_2,y_2,head1=False)

        data = {'x_1':x_1,'x_2':x_2,'stereo':stereo}
        """
        # CAT THE RESULT IN BATCH DIM
        x = []
        x.append(torch.cat((x_1[0],x_2[0]),dim=0))
        features = []  # Initialize the result list

        for i in range(len(x_1[1])):
            concatenated_element = torch.cat((x_1[1][i], x_2[1][i]), dim=0)
            features.append(concatenated_element)
        x.append(features)
        """
        return data


    def _forward_backbone(self,x,y):
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x,y

    def _forward_head(self,x,y,head1=True):
        if head1:
            for m in self.head_1:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return x,y

        else:
            for m in self.head_2:
                if m.f != -1:  # if not from previous layer
                    if isinstance(m.f, int):
                        x = y[m.f]
                    else:
                        result = []
                        for j in m.f:
                            # -13, refers to the first layer of the second head which is at index 25 in the .yaml file
                            # 25 - 13 = 12 = first layer of the first head  and as the list y contain values only for the backbone at this stage
                            # we cannot get at index 25 it would throw an out of bound error.
                            if j != -1:result.append(y[j-13] if j > 10 else y[j])
                            else:result.append(x)

                        x = result


                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return x,y



    """
    For each element in feature_1, we check how it interacts with each element in feature_2.
    This interaction is determined by the attention scores,
    The attention scores quantify the relationship or compatibility between each element in feature_1 and each element in feature_2
    """
    def _cross_attention(self,feature_1,feature_2):

         scores = torch.matmul(feature_1, feature_2.transpose(-2, -1))  # Dot product
         attention_weights = torch.nn.functional.softmax(scores, dim=-1)
         attended_feature_2 = torch.matmul(attention_weights, feature_2)

         return attended_feature_2

    def load_pretrained_weights(self,weights):
        #self.load(torch.load(weights))
        weights = torch.load(weights)

        ## LOAD HEAD 2 FROM HEAD 1 WEIGHTS
        head_2_dict = {}
        for name_1, param_1 in self.state_dict().items():
            name_1_number = int(name_1.split('.')[1])
            if name_1_number == 25:break
            if (name_1_number >=12 ):
                name_2_number = name_1_number + 13
                name_2 =  "model." + str(name_2_number) + "." + ".".join(name_1.split('.')[2:])
                head_2_dict[name_2] = param_1

        # the weights that we got from the parameter will contain yolo8 pre-trained on coco dataset
        # we copy the weight of the first head to the second head
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd.update(head_2_dict)
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load

        LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')
