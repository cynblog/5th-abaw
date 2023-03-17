import torch
from torchvision import transforms
from PIL import Image
import os, tqdm, cv2, pickle, threading
from glob import glob
from dataset import dataset

model_path = r"/amax/cvpr23_competition/pretrain_model/efficientnet_affectnet/enet_b2_8_best.pt"
model = torch.load(model_path)
model.classifier = torch.nn.Identity()
model.eval()

global transform
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def convert_feature(mov_dir):
    image_tensor_list = []
    mov_dir = sorted(mov_dir, key=lambda item : os.path.basename(item).rsplit(".")[0])
    mov_features = torch.tensor([])
    for each_pic in mov_dir:
    # for each_pic in os.listdir(mov_dir):
        # pic_path = os.path.join(mov_dir, each_pic)
        img = Image.open(each_pic).convert("RGB")
        img = transform(img).unsqueeze(0)
        mov_feature = model(img.to("cuda")).cpu()
        # image_tensor_list.append(img)
        mov_features = torch.cat([mov_features, mov_feature], dim=0)
    # print(mov_features.shape)
    # exit()
    return mov_features.cpu().detach().numpy()
    
# print(convert_feature("/amax/cvpr23_competition/challenge_4/val/jpg/24568"))

def get_full_face_path(video_name_list: list, jpg_path, face_path, resize_jpg_path, result_feature: dict):
    for video_name in tqdm.tqdm(video_name_list):
        video_jpg_path = os.path.join(jpg_path, video_name)
        video_face_path = os.path.join(face_path, video_name)
        video_jpg_list = glob(os.path.join(video_jpg_path, "*"))
        video_jpg_list = sorted(video_jpg_list)
        video_face_path_list = []
        break_flg = False
        for jpg in video_jpg_list:
            # jpg_name = os.path.basename(jpg)
            # face_jpg_path = os.path.join(video_face_path, jpg_name)
            # # 如果帧无法识别出人脸
            # if not os.path.exists(face_jpg_path):
            #     video_resize_jpg_path = os.path.join(resize_jpg_path, video_name)
            #     save_face_jpg_path = os.path.join(video_resize_jpg_path, jpg_name)
            #     # 判断这一帧是否resize过
            #     if not os.path.exists(save_face_jpg_path):
            #         # 将原始图片resize到224
            #         if not os.path.exists(video_resize_jpg_path):
            #             os.makedirs(video_resize_jpg_path)
            #         jpg_image = cv2.imread(jpg)
            #         cv2.imwrite(save_face_jpg_path, cv2.resize(jpg_image, (224, 224) ))
                
            #     video_face_path_list.append(save_face_jpg_path)
            #     break_flg = True
            # else:
            #     video_face_path_list.append(face_jpg_path)
            # if break_flg:
            #     continue 
            video_face_path_list.append(jpg)
        video_face_feature = convert_feature(video_face_path_list)
        result_feature.update({video_name: video_face_feature})
        
    
def main():
    dataset_type = "train"
    # jpg_path = rf"/amax/cvpr23_competition/challenge_4/{dataset_type}/jpg"
    jpg_path = rf'/data02/mxy/CVPR_ABAW/dataset/cropped_aligned'
    face_path = rf"/data02/cvpr23_competition/challenge_4/{dataset_type}/face_jpg"
    resize_jpg_path = rf"/data02/cvpr23_competition/challenge_4/{dataset_type}/resize_face_jpg"
    # feature_path = rf"/amax/cvpr23_competition/challenge_4/{dataset_type}_face_feature"
    feature_path = rf'/data02/mxy/CVPR_ABAW/dataset/feature/cropped_feature'
    
    result_feature = {}
    all_video_name_list = os.listdir(jpg_path)
    thread_num = 8
    result_feature = {}
    length = int (len(all_video_name_list) / thread_num)
    thread_list = [threading.Thread(target=get_full_face_path, \
        args=(all_video_name_list[length * i: length * (i + 1)], \
            jpg_path, face_path, resize_jpg_path, result_feature)) for i in range(thread_num)]
    thread_list.append(threading.Thread(target=get_full_face_path, \
        args=(all_video_name_list[length * thread_num:], \
            jpg_path, face_path, resize_jpg_path, result_feature)))

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    
    # get_full_face_path(all_video_name_list, jpg_path, face_path, resize_jpg_path, result_feature)       
    with open(feature_path + ".pkl", "wb") as f:
        pickle.dump(result_feature, f)


if __name__== "__main__":
    main()