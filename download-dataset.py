
from cmath import atanh
from io import StringIO
import json
import pandas as pd
import os
import boto3
from glob import glob
import subprocess
import shutil
import zipfile
import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='SagemakerGT用画像選定')  
parser.add_argument('-t','--target-path', help='Ground Labeling folder name')
parser.add_argument('-g','--job-name', help='Ground Truth job name')

args = parser.parse_args()

# 入力
target_path = args.target_path
job_name = args.job_name

#AWSの設定
bucket_name = "vege-label-job"
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
s3 = boto3.client('s3',region_name='ap-northeast-1')

def download_job_results(bucket_name, target_path):
    object_list = []
    response = s3.list_objects(Bucket=bucket_name, Prefix=target_path)
    if "Contents" in response:
        contents = response["Contents"]
        for content in contents:
            object_list.append(content["Key"])
            
    print("downloading images and job-outputs ...")
    
    for name in tqdm(object_list):
        dir_name = "/".join(name.split("/")[:-1])
        os.makedirs(dir_name, exist_ok=True)
        try:
            bucket.download_file(name, name)
        except:
            if not "iteration" in name:
                print("could not download {}".format(name))
    print("Successfully downloaded from {}!!".format(target_path))

def parse_gt_output(manifest_path, job_name):
    """
    Captures the json Ground Truth bounding box annotations into a pandas dataframe

    Input:
    manifest_path: S3 path to the annotation file
    job_name: name of the Ground Truth job

    Returns:
    df_bbox: pandas dataframe with bounding box coordinates
             for each item in every image
    """

#     filesys = s3fs.S3FileSystem()
#     with filesys.open(manifest_path) as fin:
    with open(manifest_path) as fin:
        annot_list = []
        for line in fin.readlines():
            record = json.loads(line)
            if job_name in record.keys():  # is it necessary?
                image_file_path = record["source-ref"]
                image_file_name = image_file_path.split("/")[-1]
                class_maps = record[f"{job_name}-metadata"]["class-map"]

                imsize_list = record[job_name]["image_size"]
                assert len(imsize_list) == 1
                image_width = imsize_list[0]["width"]
                image_height = imsize_list[0]["height"]

                for annot in record[job_name]["annotations"]:
                    left = annot["left"]
                    top = annot["top"]
                    height = annot["height"]
                    width = annot["width"]
                    class_name = class_maps[f'{annot["class_id"]}']

                    annot_list.append(
                        [
                            image_file_name,
                            class_name,
                            left,
                            top,
                            height,
                            width,
                            image_width,
                            image_height,
                        ]
                    )

        df_bbox = pd.DataFrame(
            annot_list,
            columns=[
                "img_file",
                "category",
                "box_left",
                "box_top",
                "box_height",
                "box_width",
                "img_width",
                "img_height",
            ],
        )

    return df_bbox

def save_df_to_s3(df_local, s3_bucket, destination):
    """
    Saves a pandas dataframe to S3

    Input:
    df_local: Dataframe to save
    s3_bucket: Bucket name
    destination: Prefix
    """

    csv_buffer = StringIO()
    s3_resource = boto3.resource("s3")

    df_local.to_csv(csv_buffer, index=False)
    s3_resource.Object(s3_bucket, destination).put(Body=csv_buffer.getvalue())


def main():
    """
    Performs the following tasks:
    1. Reads input from 'input.json'
    2. Parses the Ground Truth annotations and creates a dataframe
    3. Saves the dataframe to S3
    """

    with open("input.json") as fjson:
        input_dict = json.load(fjson)

    s3_bucket = input_dict["s3_bucket"]
    job_id = input_dict["job_id"]
    job_name = input_dict["ground_truth_job_name"]

    mani_path = f"s3://{s3_bucket}/{job_id}/ground_truth_annots/{job_name}/manifests/output/output.manifest"

    df_annot = parse_gt_output(mani_path, job_name)
    dest = f"{job_id}/ground_truth_annots/{job_name}/annot.csv"
    save_df_to_s3(df_annot, s3_bucket, dest)
    
def annot_yolo(df_ann, cats):
    """
    Prepares the annotation in YOLO format

    Input:
    annot_file: csv file containing Ground Truth annotations
    ordered_cats: List of object categories in proper order for model training

    Returns:
    df_ann: pandas dataframe with the following columns
            img_file int_category box_center_w box_center_h box_width box_height


    Note:
    YOLO data format: <object-class> <x_center> <y_center> <width> <height>
    """

#     df_ann = pd.read_csv(annot_file)

    df_ann["int_category"] = df_ann["category"].apply(lambda x: cats.index(x))
    df_ann["box_center_w"] = df_ann["box_left"] + df_ann["box_width"] / 2
    df_ann["box_center_h"] = df_ann["box_top"] + df_ann["box_height"] / 2

    # scale box dimensions by image dimensions
    df_ann["box_center_w"] = df_ann["box_center_w"] / df_ann["img_width"]
    df_ann["box_center_h"] = df_ann["box_center_h"] / df_ann["img_height"]
    df_ann["box_width"] = df_ann["box_width"] / df_ann["img_width"]
    df_ann["box_height"] = df_ann["box_height"] / df_ann["img_height"]

    return df_ann

def get_cats(json_file):
    """
    Makes a list of the category names in proper order

    Input:
    json_file: s3 path of the json file containing the category information

    Returns:
    cats: List of category names
    """

    with open(json_file) as fin:
        line = fin.readline()
        record = json.loads(line)
        labels = [item["label"] for item in record["labels"]]

    return labels




#s3からGTでアノテーション済みのジョブファイル一式ダウンロード
# dl_command = "aws s3 cp s3://vege-label-job/{}/ ./{}/. --recursive".format(target_path,target_path)
# os.system(dl_command)
download_job_results(bucket_name, target_path)


mani_path = "./{}/{}/manifests/output/output.manifest".format(target_path, job_name)
df_annot = parse_gt_output(mani_path, job_name)

categories = get_cats("./{}/{}/annotation-tool/data.json".format(target_path, job_name))
with open("./{}/classes.txt".format(target_path), "w") as f:
    f.write("\n".join(categories))
yolo_df = annot_yolo(df_annot, categories)

# txtに保存
prefix = "label"
unique_images = yolo_df["img_file"].unique()
output = f"./{target_path}/{prefix}"

os.makedirs(output, exist_ok=True)

# yolo_df からYOLO形式のtxtファイルを作成
for image_file in unique_images:
    df_single_img_annots = yolo_df.loc[yolo_df.img_file == image_file]
    annot_txt_file =   image_file.split(".")[0] + ".txt"
    destination = f"./{target_path}/{prefix}/{annot_txt_file}"
    
    csv_buffer = StringIO()
    df_single_img_annots.to_csv(
        csv_buffer,
        index=False,
        header=False,
        sep=" ",
        float_format="%.4f",
        columns=[
            "int_category",
            "box_center_w",
            "box_center_h",
            "box_width",
            "box_height",
        ],
    )
    
    with open(destination, "w") as f:
        f.write(csv_buffer.getvalue().replace("\r", ""))

for image_file in unique_images:
    df_single_img_annots = yolo_df.loc[yolo_df.img_file == image_file]
    annot_txt_file =   image_file.split(".")[0] + ".txt"
    destination = f"./{target_path}/{prefix}/{annot_txt_file}"
    
    csv_buffer = StringIO()
    df_single_img_annots.to_csv(
        csv_buffer,
        index=False,
        header=False,
        sep=" ",
        float_format="%.4f",
        columns=[
            "int_category",
            "box_center_w",
            "box_center_h",
            "box_width",
            "box_height",
        ],
    )
    
    with open(destination, "w") as f:
        f.write(csv_buffer.getvalue().replace("\r", ""))

#データセットの作成
img_list = glob("./{}/*.jpg".format(target_path))

namelist = []
for full_name in img_list:
    name, _ =  os.path.splitext(full_name.split("/")[-1])
    namelist.append(name)
random.seed(2020)
random.shuffle(namelist)

img_list_train = namelist[0:int(len(namelist)*0.7)]
img_list_valid =  namelist[int(len(namelist)*0.7):]

os.makedirs('./{}/dataset_{}/images/train'.format(target_path,target_path), exist_ok=True)
os.makedirs('./{}/dataset_{}/images/valid'.format(target_path,target_path), exist_ok=True)
os.makedirs('./{}/dataset_{}/labels/train'.format(target_path,target_path), exist_ok=True)
os.makedirs('./{}/dataset_{}/labels/valid'.format(target_path,target_path), exist_ok=True)

for i, j in enumerate(img_list_train):
    old_path = "./{}/{}.jpg".format(target_path,img_list_train[i])
    new_path = './{}/dataset_{}/images/train/{}.jpg'.format(target_path,target_path,img_list_train[i])
    shutil.copy(old_path, new_path)

for i, j in enumerate(img_list_valid):
    old_path = "./{}/{}.jpg".format(target_path,img_list_valid[i])
    new_path = './{}/dataset_{}/images/valid/{}.jpg'.format(target_path,target_path,img_list_valid[i])
    shutil.copy(old_path, new_path)

for i, j in enumerate(img_list_train):
    old_path = "./{}/{}/{}.txt".format(target_path,"label",img_list_train[i])
    new_path = './{}/dataset_{}/labels/train/{}.txt'.format(target_path,target_path,img_list_train[i])
    shutil.copy(old_path, new_path)

for i, j in enumerate(img_list_valid):
    old_path = "./{}/{}/{}.txt".format(target_path,"label",img_list_valid[i])
    new_path = './{}/dataset_{}/labels/valid/{}.txt'.format(target_path,target_path,img_list_valid[i])
    shutil.copy(old_path, new_path)

# daatsetのzip化
os.chdir(target_path)

zip_name = 'dataset_{}.zip'.format(target_path)
dir = 'dataset_{}'.format(target_path)
zp = zipfile.ZipFile(zip_name, 'w')

for dirname, subdirs, filenames in os.walk(dir):
    for fname in filenames:
        zp.write(os.path.join(dirname, fname))
zp.close()

bucket.upload_file(zip_name, "dataset/{}".format(zip_name))
