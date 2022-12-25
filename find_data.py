import Image_processing as ip

files = ip.image_path_AllFolder('C:/Users/Student/anaconda3/envs/yolov7/Lib/site-packages/keras')
for i in files:
    if i == 'multi_gpu_model':
        print(i)
else:
    print('File not found!')
    
