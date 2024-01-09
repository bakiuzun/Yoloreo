from ultralytics.models.yolo.detect.predict import DetectionPredictor
from dataset import CliffDataset
from torch.utils.data import DataLoader
import torch

from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils import ops

from ultralytics.engine.results import Results
import geopandas as gpd
from shapely.geometry import Polygon
from georef_bbox import GeorefBbox
from utils import save_image_with_bbox,get_georeferenced_pos


class YoloreoPredictor(DetectionPredictor):
    def __init__(self,cfg,csv_path,model,conf=0.50):

        super().__init__(cfg=cfg)

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = CliffDataset(path=csv_path)

        self.batch_size = 32
        self.dataloader = DataLoader(self.dataset,shuffle=False,batch_size=self.batch_size)

        self.args.conf = conf
        self.model = model
        self.model = self.model.float()
        self.model.eval()


        self.attr_heads = []
        self.attr_base_img_ids = []
        self.attr_patch_ids = []
        self.georef_poses = []

    def predict(self,save_res=False,create_shape_file=False):


        for idx,batch in enumerate(self.dataloader):

            self.preprocess(batch)

            # inference
            self.batch = batch

            out = self.model(batch['img'].to(self.device))
            self.result_head_1 = self.postprocess(out["x_1"], batch["img"][:,0], batch["img"][:,0])
            self.result_head_2 = self.postprocess(out["x_2"], batch["img"][:,1], batch["img"][:,1],head="head2")

            self.handle_result(save_res,create_shape_file,idx)

        if create_shape_file:
            self.create_shape_file()


    def handle_result(self,save_img_res,create_shape_file,idx):

        for i in range(len(self.result_head_1)):
            res_head_1 = self.result_head_1[i]
            res_head_2 = self.result_head_2[i]
            ## STEREO
            if res_head_1.path != res_head_2.path:
                if save_img_res:
                    save_image_with_bbox(res_head_1.orig_img,f"head1_{i+(self.batch_size+idx)}.png",res_head_1.boxes)
                    save_image_with_bbox(res_head_2.orig_img,f"head2_{i+(self.batch_size*idx)}.png",res_head_2.boxes)
            else:
                ## MONO MERGE RESULT
                ## THE IMAGE IS THE SAME
                if save_img_res:
                    save_image_with_bbox(res_head_1.orig_img,f"head1_head2_{i+(self.batch_size*idx)}.png",res_head_1.boxes,res_head_2.boxes)

                if create_shape_file:
                    self.fill_georef_poses(res_head_1.path,res_head_1.boxes,1)
                    self.fill_georef_poses(res_head_2.path,res_head_2.boxes,2)

    def fill_georef_poses(self,path,bbox,head_num):

        for pos in bbox.xyxy.cpu().numpy():
            x,y,base_img_id,patch_id = get_georeferenced_pos(path,int(pos[0]),int(pos[1]))
            x2,y2,_,_ = get_georeferenced_pos(path,int(pos[2]),int(pos[3]))

            self.attr_heads.append(head_num)
            self.attr_base_img_ids.append(base_img_id)
            self.attr_patch_ids.append(patch_id)
            self.georef_poses.append([x,y,x2,y2])

    def create_shape_file(self):

        print("LEN ",len(self.georef_poses))

        polygons = []
        bbox_x1 = []
        bbox_y1 = []
        bbox_x2 = []
        bbox_y2 = []
        for bbox in self.georef_poses:
            bbox_y1.append(bbox[1])
            bbox_x1.append(bbox[0])
            bbox_y2.append(bbox[2])
            bbox_x2.append(bbox[3])
            polygons.append(Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]),(bbox[2], bbox[3]), (bbox[0], bbox[3])]))

        if len(polygons) > 0:
            #data = {'geometry': polygons}
            data = {
            'geometry': polygons,
            'heads': self.attr_heads,
            'img_id': self.attr_base_img_ids,
            'patch_ids': self.attr_patch_ids,
            "bbox_x1": bbox_x1,
            "bbox_y1": bbox_y1,
            "bbox_x2": bbox_x2,
            "bbox_y2": bbox_y2,
            }
            gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')  # Change EPSG code as needed

            # Save the GeoDataFrame to a shapefile
            output_shapefile = 'bounding_boxes.shp'
            gdf.to_file(output_shapefile)




    def preprocess(self,batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

    def postprocess(self, preds, img, orig_imgs,head="head1"):


        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            #img_path = self.batch[0][i]

            ## tiny update comapred to the official yolov8
            img_path = self.batch["im_files_patch1"][i]
            if head == "head2":
                img_path = self.batch["im_files_patch2"][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
