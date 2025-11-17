import torch

def tissue_segmentation_models(model_name):
    # Credit, grandqc: https://www.nature.com/articles/s41467-024-54769-y  Trained on H&E Images  
    import torch    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    if model_name == 'grandqc':        
        import segmentation_models_pytorch as smp        
        weights_path = 'codes/grandqc_10mpp.pth'        
        
        model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b0', encoder_weights=None, weights_only=True, classes=2, activation=None)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        from torchvision import transforms
        eval_transforms = transforms.Compose([                       
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return model, eval_transforms 
    
    if model_name == 'pathprofiler':
        import cv2
        class CLAHE(object):
            # histogram equalisation
            def __init__(self):
                self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            def __call__(self, img):
                HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                HSV[:, :, 0] = self.clahe.apply(HSV[:, :, 0])
                img = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
                return img            
        
        import torch
        class UNet_down_block(torch.nn.Module):
            def __init__(self, input_channel, output_channel, down_size):
                super(UNet_down_block, self).__init__()
                self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
                self.bn1 = torch.nn.InstanceNorm2d(output_channel)
                self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
                self.bn2 = torch.nn.InstanceNorm2d(output_channel)
                self.max_pool = torch.nn.MaxPool2d(2, 2)
                self.relu = torch.nn.ReLU()
                self.down_size = down_size

            def forward(self, x):
                if self.down_size:
                    x = self.max_pool(x)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                return x
        
        class UNet_up_block(torch.nn.Module):
            def __init__(self, prev_channel, input_channel, output_channel):
                super(UNet_up_block, self).__init__()
                self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
                self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
                self.bn1 = torch.nn.InstanceNorm2d(output_channel)
                self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
                self.bn2 = torch.nn.InstanceNorm2d(output_channel)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout2d(p=0.2)

            def forward(self, prev_feature_map, x):
                x = self.up_sampling(x)
                x = torch.cat((x, self.dropout(prev_feature_map)), dim=1)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                return x
        
        class UNet(torch.nn.Module):
            def __init__(self):
                super(UNet, self).__init__()

                self.down_block1 = UNet_down_block(3, 16, False)
                self.down_block2 = UNet_down_block(16, 32, True)
                self.down_block3 = UNet_down_block(32, 64, True)
                self.down_block4 = UNet_down_block(64, 128, True)
                self.down_block5 = UNet_down_block(128, 256, True)
                self.down_block6 = UNet_down_block(256, 512, True)
                self.down_block7 = UNet_down_block(512, 1024, True)

                self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
                self.bn1 = torch.nn.InstanceNorm2d(1024)
                self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
                self.bn2 = torch.nn.InstanceNorm2d(1024)

                self.up_block1 = UNet_up_block(512, 1024, 512)
                self.up_block2 = UNet_up_block(256, 512, 256)
                self.up_block3 = UNet_up_block(128, 256, 128)
                self.up_block4 = UNet_up_block(64, 128, 64)
                self.up_block5 = UNet_up_block(32, 64, 32)
                self.up_block6 = UNet_up_block(16, 32, 16)

                self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
                self.last_bn = torch.nn.InstanceNorm2d(16)
                self.last_conv2 = torch.nn.Conv2d(16, 2, 1, padding=0)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.down_block1(x)
                x2 = self.down_block2(x1)
                x3 = self.down_block3(x2)
                x4 = self.down_block4(x3)
                x5 = self.down_block5(x4)
                x6 = self.down_block6(x5)
                x7 = self.down_block7(x6)
                x7 = self.relu(self.bn1(self.mid_conv1(x7)))
                x7 = self.relu(self.bn2(self.mid_conv2(x7)))
                x = self.up_block1(x6, x7)
                x = self.up_block2(x5, x)
                x = self.up_block3(x4, x)
                x = self.up_block4(x3, x)
                x = self.up_block5(x2, x)
                x = self.up_block6(x1, x)
                x = self.relu(self.last_bn(self.last_conv1(x)))
                x = self.last_conv2(x)
                return x    
        
        weights_path = 'codes/path_profiler.pth'        
        
        checkpoint = torch.load(weights_path, map_location=DEVICE)        
        
        state_dict = checkpoint['state_dict']
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove 'module.' prefix
            new_state_dict[name] = v

        model = UNet().to(DEVICE)
        model.load_state_dict(new_state_dict)
        model.eval()
        
        from torchvision import transforms
        eval_transforms = transforms.Compose([
            CLAHE(),  # apply histogram equalization in HSV space            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1] range
        ])
        return model, eval_transforms

def predict_nn_model_on_batch(model_name, images, batch_size=8):
    
    model, eval_transforms = tissue_segmentation_models(model_name)
    model.eval()

    device = next(model.parameters()).device

    predictions = []
    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size]
                       
            batch_tensors = [eval_transforms(img) for img in batch]            
            batch_tensors = torch.stack(batch_tensors) 
            batch_tensors = batch_tensors.to(device)

            preds = model(batch_tensors)  
            preds_np = preds.cpu().numpy()
            predictions.extend(list(preds_np))
    return predictions