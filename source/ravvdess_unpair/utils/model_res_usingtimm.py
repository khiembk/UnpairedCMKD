import torch
import torch.nn as nn
import timm

def get_timm_model_name(arch_name):
    """
    Map tên từ args sang tên chuẩn của TIMM.
    Bạn có thể đổi 'tiny' thành 'small' hoặc 'base' ở đây nếu muốn model to hơn.
    """
    arch_name = arch_name.lower()
    if 'resnet18' in arch_name:
        return 'resnet18'
    elif 'resnet50' in arch_name:
        return 'resnet50'
    elif 'convnext' in arch_name:
        return 'convnext_tiny' # Hoặc 'convnext_small'
    elif 'swin' in arch_name:
        return 'swin_tiny_patch4_window7_224'
    else:
        raise ValueError(f"Architecture {arch_name} is not supported yet.")


class ImageNet(nn.Module):
    """
    Model xử lý Video/Image.
    Input: (B, T, C, H, W) -> Output: Logits, Features
    """
    def __init__(self, args, pretrained=False):
        super(ImageNet, self).__init__()
        self.args = args
        self.pretrained = pretrained
        
        # 1. Xác định tên model TIMM
        timm_model_name = get_timm_model_name(args.image_arch)
        print(f"=> Creating Video Backbone: {timm_model_name} | Pretrained: {self.pretrained}")

        # 2. Khởi tạo Backbone qua TIMM
        # num_classes=0: Chỉ lấy Feature Vector
        # global_pool='avg': Tự động Pooling bất chấp kích thước ảnh đầu vào
        self.backbone = timm.create_model(
            timm_model_name, 
            pretrained=self.pretrained, 
            num_classes=0, 
            global_pool='avg'
        )
        
        # Lấy kích thước feature vector (512, 768, 2048...) tự động
        self.feature_dim = self.backbone.num_features

        # 3. Projector / Classification Head
        self.head_video = nn.Linear(self.feature_dim, 6)

    def forward(self, x):
        # Input x: (B, T, C, H, W)
        
        # Check và permute nếu input bị ngược (B, C, T, H, W)
        if x.dim() == 5 and x.size(1) == 3: 
            x = x.permute(0, 2, 1, 3, 4)
            
        B, T, C, H, W = x.shape
        
        # Gộp Batch và Time để đưa vào Backbone 2D: (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)

        # Forward qua Backbone -> Ra luôn vector (B*T, Feature_Dim)
        features = self.backbone(x)

        # Tách lại Batch và Time: (B, T, Feature_Dim)
        features = features.view(B, T, -1)
        
        # Temporal Pooling: Trung bình cộng các frame
        final_features = torch.mean(features, dim=1) # (B, Feature_Dim)

        # Classification
        logits = self.head_video(final_features)
        
        return logits, final_features, []

    def forward_encoder(self, x):
        # Chỉ lấy feature, không qua logit head (dùng cho alignment loss)
        if x.dim() == 5 and x.size(1) == 3: 
            x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        features = self.backbone(x) # (B*T, Dim)
        features = features.view(B, T, -1)
        final_features = torch.mean(features, dim=1) # (B, Dim)
        
        return final_features, []
    
    def forward_head(self, feature_vector):
        return self.head_video(feature_vector)
    
    def fc(self, feature_vector):
        return self.head_video(feature_vector)


class AudioNet(nn.Module):
    """
    Model xử lý Audio (Spectrogram).
    Input: (B, 1, F, T) -> Output: Logits, Features
    """
    def __init__(self, args, pretrained=False):
        super(AudioNet, self).__init__()
        self.args = args
        self.pretrained = pretrained

        # 1. Xác định tên model TIMM
        timm_model_name = get_timm_model_name(args.audio_arch)
        print(f"=> Creating Audio Backbone: {timm_model_name} | Pretrained: {self.pretrained}")

        # 2. Khởi tạo Backbone qua TIMM
        # in_chans=1: Quan trọng để xử lý Spectrogram (ảnh đen trắng)
        self.backbone = timm.create_model(
            timm_model_name, 
            pretrained=self.pretrained, 
            in_chans=1, 
            num_classes=0, 
            global_pool='avg'
        )
        
        self.feature_dim = self.backbone.num_features

        # 3. Projector / Classification Head
        self.head_audio = nn.Linear(self.feature_dim, 6)

    def forward(self, x):
        # Input x: (B, 1, F, T) - Coi như ảnh 1 channel
        
        # Forward qua Backbone -> Ra luôn vector (B, Feature_Dim)
        features = self.backbone(x)

        # Classification
        logits = self.head_audio(features)
        
        return logits, features, []

    def forward_encoder(self, x):
        return self.backbone(x), []

    def forward_head(self, feature_vector):
        return self.head_audio(feature_vector)
    
    def fc(self, feature_vector):
        return self.head_audio(feature_vector)

