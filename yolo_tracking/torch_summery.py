from torchsummary import summary
import torch 

model_list = [
    '/Users/leo/Downloads/yolo_v5/yolov5/runs/train/exp3/weights/best.pt',
    '/Users/leo/Downloads/yolov5l/weights/best.pt',
    '/Users/leo/git_workspace/yolov7/weights/best.pt',
    '/Users/leo/Downloads/yolov7-X-custom20/best.pt',
    '/Users/leo/Downloads/yolov5m/weights/best.pt',
    '/Users/leo/Downloads/yolov5s/weights/best.pt'
    ]

model = torch.hub.load('ultralytics/yolov5', 'custom',
    path=f'{model_list[5]}', force_reload=True)

from thop import clever_format, profile
from nets.yolo4 import YoloBody

if __name__ == "__main__":
    input_shape     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    backbone        = 'cspdarknet'
    phi             = 'l'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(anchors_mask, num_classes, phi, backbone=backbone).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
   
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))




