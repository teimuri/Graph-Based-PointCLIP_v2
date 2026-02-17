import torch
from clip import clip
import torch.nn as nn

from trainers.best_param import best_prompt_weight
from trainers.mv_utils_zs import Realistic_Projection
from trainers.graph_handler import aggergator_Graph
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # 1. Extract the batch size from the target tensor
    batch_size = target.size(0) 
    
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        # 2. Sum up the correct predictions
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        
        # 3. Divide by batch_size and multiply by 100 to get a percentage
        acc_percentage = correct_k.mul_(100.0 / batch_size) 
        
        res.append(float(acc_percentage.cpu().numpy()))
        
    return res
class Textual_Encoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        prompts = best_prompt_weight['{}_{}_test_prompts'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        text_feat = self.clip_model.encode_text(prompts)
        return text_feat


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())
    return model


@TRAINER_REGISTRY.register()
class PointCLIPV2_ZS(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        text_feat = textual_encoder()
        self.text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Realistic projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        
        self.view_weights = torch.Tensor(best_prompt_weight['{}_{}_test_weights'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]).cuda()
        self.gnn_aggregator = aggergator_Graph(self.channel).to(torch.float32).cuda()

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        # (Optional but recommended) Turn on gradients specifically for the GNN
        for param in self.gnn_aggregator.parameters():
            param.requires_grad = True

        # 3. OPTIMIZER: Tell Dassl to only train the GNN
        self.model = self.gnn_aggregator 
        
        # Change this line to explicitly pass the parameters:
        self.optim = build_optimizer(self.model.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # 4. Define the Loss Function (Cross-Entropy for classification)
        self.criterion = torch.nn.CrossEntropyLoss()


    def real_proj(self, pc, imsize=224):
        img = self.get_img(pc).cuda()
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img
    def commen_inference(self, pc):
        with torch.no_grad():
            # Realistic Projection
            images = self.real_proj(pc)            
            images = images.type(self.dtype)
            
            # Image features
            image_feat = self.visual_encoder(images)
            
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
            image_feat = image_feat.reshape(-1, self.channel).type(torch.float32) # Shape: [B * 10, C]
            batch_size = pc.shape[0]
        return image_feat,batch_size
    def model_inference(self, pc, label=None):
        image_feat,batch_size = self.commen_inference(pc)
        with torch.no_grad():
            # Realistic Projection

            # Pass through the GNN (Outputs shape: [Batch, Channel])
            aggr_feat = self.gnn_aggregator(image_feat, batch_size, self.num_views)

            # Normalize the final aggregated feature before comparing to text
            aggr_feat = aggr_feat / aggr_feat.norm(dim=-1, keepdim=True)


            self.feat_store.append(aggr_feat)
            self.label_store.append(label)

            # Logits calculation (Notice we no longer multiply by 10 since we pooled, not concatenated)
            logits = 100. * aggr_feat @ self.text_feat.to(aggr_feat.dtype).t()
        return logits

    def forward_backward(self, batch,training=None):
        # 1. Unpack the batch from the DataLoader
        pc = batch["img"].cuda()
        label = batch["label"].cuda()

        if training:
            # --- 3D AUGMENTATION START ---
            
            # # A. Random Rotation (Around the Y-axis / Up-axis)
            # theta = torch.rand(1).item() * 2 * 3.1415926  # Random angle
            # cos_t = torch.cos(torch.tensor(theta))
            # sin_t = torch.sin(torch.tensor(theta))
            # # Rotation matrix for Y-axis
            # rot_mat = torch.tensor([
            #     [cos_t, 0, sin_t],
            #     [0, 1, 0],
            #     [-sin_t, 0, cos_t]
            # ], device=pc.device)
            # pc = torch.matmul(pc, rot_mat)

            # B. Point Jittering (Adding small noise)
            # This helps the model stay robust to sensor noise (crucial for ScanObjectNN)
            noise = torch.randn_like(pc) * 0.04 
            pc = pc + noise

            # C. Random Scaling
            # Slightly change the size of the object
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            pc = pc * scale
            
            # --- 3D AUGMENTATION END ---

        image_feat,batch_size = self.commen_inference(pc)

        # 2. Project 3D points to 2D images
        # images = self.real_proj(pc).type(self.dtype)

        # 3. Extract CLIP features WITHOUT tracking gradients
        # with torch.no_grad():
        #     image_feat = self.visual_encoder(images)
        #     image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            
        #     # ADD THESE LINES to apply the view weights and cast to float32
        #     image_feat = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
        #     image_feat = image_feat.reshape(-1, self.channel).type(torch.float32)

        # 4. GNN Aggregation (Now receiving float32 weighted features)
        aggr_feat = self.gnn_aggregator(image_feat, batch_size, self.num_views)
        aggr_feat = aggr_feat / aggr_feat.norm(dim=-1, keepdim=True)
        
        # 5. Calculate Logits (Ensure text_feat is cast to float32 to match aggr_feat)
        logits = 100. * aggr_feat @ self.text_feat.detach().to(aggr_feat.dtype).t()
        # 6. Calculate Loss
        loss = self.criterion(logits, label)

        # 7. Backward Pass & Optimizer Step (Dassl handles the zero_grad() and step() here)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 8. Calculate accuracy for the training logger
        # (Assuming you imported the accuracy function from earlier)
        acc = accuracy(logits.detach(), label, topk=(1,))[0]

        # Return a dictionary so Dassl can print the loss/accuracy to your terminal
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        return loss_summary