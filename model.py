"""
author Uzun Baki
-- u.officialdeveloper@gmail.com
"""

import torch
from ultralytics.nn.tasks import *
from utils import parse_my_detection_model
import copy
import numpy as np

# torch.Size([16, 576, 20, 20]) backbone out shape

class Yoloreo(BaseModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False,weights=None):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""

        """
        YOLO init --> check yolo code
        """
        super().__init__()

        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_my_detection_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)


        first_head = self.model[22]  # Detect()
        second_head = self.model[-1]  # Detect()
        s = 256  # 2x min stride
        first_head.inplace = self.inplace
        second_head.inplace = self.inplace

        # strides by default for objection detection with YOLOV8
        first_head.stride = torch.tensor([ 8., 16., 32.])
        second_head.stride = torch.tensor([ 8., 16., 32.])
        self.stride = first_head.stride

        first_head.bias_init()  # only run once
        second_head.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

        ## OUR modification

        # for cross attention
        """
        self.linear_query_p3 = torch.nn.Conv1d(192, 192, 1)
        self.linear_key_p3 = torch.nn.Conv1d(192, 192, 1)
        self.linear_value_p3 = torch.nn.Conv1d(192, 192, 1)
        self.p3 = [self.linear_query_p3,self.linear_key_p3,self.linear_value_p3]

        self.linear_query_p4 = torch.nn.Conv1d(384, 384, 1)
        self.linear_key_p4 = torch.nn.Conv1d(384, 384, 1)
        self.linear_value_p4 = torch.nn.Conv1d(384, 384, 1)
        self.p4 = [self.linear_query_p4,self.linear_key_p4,self.linear_value_p4]
        """
        self.linear_query_p5 = torch.nn.Conv1d(576, 576, 1)
        self.linear_key_p5 = torch.nn.Conv1d(576, 576, 1)
        self.linear_value_p5 = torch.nn.Conv1d(576, 576, 1)
        self.p5 = [self.linear_query_p5,self.linear_key_p5,self.linear_value_p5]

        ## check yolov8.yaml for the indexation
        self.index = [5,7,10]
        self.backbone = self.model[:10]
        self.head_1 = self.model[10:23]
        self.head_2 = self.model[23:]


    def enable_all_gradients(self):
        for param in self.parameters():
            param.requires_grad = True


    def forward(self, x, *args, **kwargs):
        return self._predict_once(x)

    def _predict_once(self, x):

        # NOT USED NOW
        mono_res = None

        y_1 = []
        y_2 = []

        # x shape --> (batch_size,2,channel,height,widht)
        # check dataset.py
        x_1 = x[:,0]
        x_2 = x[:,1]

        x_1,y_1 = self._forward_backbone(x_1,y_1,)
        x_2,y_2 = self._forward_backbone(x_2,y_2)

        attended_feature_2 = self._cross_attention(x_1,x_2)
        attended_feature_1 = self._cross_attention(x_2,x_1)

        x_1,y_1 = self._forward_head(attended_feature_2,y_1,head1=True)
        x_2,y_2 = self._forward_head(attended_feature_1,y_2,head1=False)

        data = {'x_1':x_1,'x_2':x_2,"mono_res":mono_res}

        return data


    def _forward_backbone(self,x,y):
        # YOLO implementation
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x,y

    def _forward_head(self,x,y,head1=True):
        """
        forward process
        """
        if head1:
            # STANDART
            for m in self.head_1:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return x,y

        else:
            # head 2
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
                            # y index 0...9
                            # refer to the yolov8.yaml
                            if j != -1:
                                result.append(y[j-13] if j > 10 else y[j])
                            else:result.append(x)

                        x = result

                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return x,y




    def _cross_attention(self,feature_1,feature_2):
        """
        For each element in feature_1, we check how it interacts with each element in feature_2.
        This interaction is determined by the attention scores,
        The attention scores quantify the relationship or compatibility between each element in feature_1 and each element in feature_2
        """

        original_shape = feature_1.shape

        reshaped_feature_map_1 = feature_1.view(feature_1.shape[0], feature_1.shape[1], -1)
        reshaped_feature_map_2 = feature_2.view(feature_2.shape[0], feature_2.shape[1], -1)

        query = self.linear_query_p5(reshaped_feature_map_1)
        key = self.linear_key_p5(reshaped_feature_map_2)
        value = self.linear_value_p5(reshaped_feature_map_2)

        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.nn.functional.softmax(scores,dim=-1)
        attended_feature_2 = torch.matmul(attention_weights, value)

        attended_feature_2 = attended_feature_2.view(*original_shape)

        return attended_feature_2

    def load_pretrained_weights(self,weights):

        weights = torch.load(weights)

        ## LOAD HEAD 2 FROM HEAD 1 WEIGHTS
        head_2_dict = {}
        for name_1, param_1 in self.state_dict().items():
            name_1_number = int(name_1.split('.')[1])

            ## check yolov8.yaml to understand the numbers
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

