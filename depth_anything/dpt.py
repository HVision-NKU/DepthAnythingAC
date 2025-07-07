import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhub.facebookresearch_dinov2_main.hubconf as dinov2

from depth_anything.util.blocksv2 import FeatureFusionBlock, _make_scratch
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import numpy as np

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = nn.Identity()
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
       
    def forward(self, out_features, patch_h, patch_w,need_fp=False,need_prior=False,teacher_features=None,alpha=0.8):
        depth_out={}
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
            
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
    
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        depth_out['out']=out
        
        return depth_out
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True, version='v1'):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.encoder = encoder
        self.version = version
        # in case the Internet connection is not stable, please load the DINOv2 locally
        # if localhub:
        #     self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=True)
        # else:
        #     self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        self.pretrained = dinov2.__dict__['dinov2_{:}14'.format(encoder)](pretrained=True)
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        
    def forward(self, x,need_fp=False,teacher_features=None,alpha=0.8,prior_mode='teacher'):
        depth_out={}
        h, w = x.shape[-2:]
        if self.version == 'v1':
            features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        else:

            features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        patch_h, patch_w = h // 14, w // 14
        
        depth_all = self.depth_head(features, patch_h, patch_w,need_fp,teacher_features,alpha)
        depth=depth_all['out']
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth).squeeze(1)
        depth_out['out']=depth

        return depth_out


class DepthAnything_AC(DPT_DINOv2):
    def __init__(self, config):
        super().__init__(**config)

    def from_pretrained(self, repo_id: str, encoder='vits'):
        from huggingface_hub import hf_hub_download
        
        filepath = hf_hub_download(repo_id=repo_id, filename=f"checkpoints/depth_anything_AC_{encoder}.pth")
        
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'version': 'v2'},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'version': 'v2'},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'version': 'v2'}
        }
        
        model = self(**model_configs[encoder])
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
            print("Using GPU for inference")
        else:
            print("Using CPU for inference")
        
        return model

    def get_intermediate_features(self, x):
        """
        Extract intermediate features from the model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            dict: Dictionary containing intermediate features including:
                - encoder_features: List of encoder feature maps
                - decoder_features: List of decoder feature maps  
                - decoder_features_path: List of decoder path features
                - cls_token: List of classification tokens
        """
        features = {
            'encoder_features': [],
            'decoder_features': [],
            'decoder_features_path': [],
            'cls_token': []
        }
        
        h, w = x.shape[-2:]
        patch_h, patch_w = h // 14, w // 14
      
        all_features = []
        for i in range(len(self.pretrained.blocks)):
            feat = self.pretrained.get_intermediate_layers(x, [i], return_class_token=True)[0]
            all_features.append(feat)
            if i in [2, 5, 8, 11]: 
                feat_map = feat[0]  
                B, N, C = feat_map.shape
                H = W = int(np.sqrt(N))
                features['encoder_features'].append(feat_map.reshape(B, H, W, C).permute(0, 3, 1, 2))
        
        out_features = []
        for layer_idx in self.intermediate_layer_idx[self.encoder]:
            out_features.append(all_features[layer_idx])
        out = []
        for i, feat in enumerate(out_features):
            if self.depth_head.use_clstoken:
                feat_map, cls_token = feat[0], feat[1]
                readout = cls_token.unsqueeze(1).expand_as(feat_map)
                feat_map = self.depth_head.readout_projects[i](torch.cat((feat_map, readout), -1))
                features['cls_token'].append(cls_token)
            else:
                feat_map = feat[0]
            feat_map = feat_map.permute(0, 2, 1).reshape((feat_map.shape[0], feat_map.shape[-1], patch_h, patch_w)).contiguous()
            
            feat_map = self.depth_head.projects[i](feat_map)
            feat_map = self.depth_head.resize_layers[i](feat_map)
            
            out.append(feat_map)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.depth_head.scratch.layer1_rn(layer_1)
        layer_2_rn = self.depth_head.scratch.layer2_rn(layer_2)
        layer_3_rn = self.depth_head.scratch.layer3_rn(layer_3)
        layer_4_rn = self.depth_head.scratch.layer4_rn(layer_4)
            
        features['decoder_features'].append(layer_1_rn)
        features['decoder_features'].append(layer_2_rn)
        features['decoder_features'].append(layer_3_rn)
        features['decoder_features'].append(layer_4_rn)

        path_4 = self.depth_head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.depth_head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.depth_head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.depth_head.scratch.refinenet1(path_2, layer_1_rn)

        features['decoder_features_path'].append(path_1)
        features['decoder_features_path'].append(path_2)
        features['decoder_features_path'].append(path_3)
        features['decoder_features_path'].append(path_4)
        
        return features
    