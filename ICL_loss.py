import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(EnhancedLoss, self).__init__()
        self.margin = margin
        self.linear_layer = nn.Linear(1280, 1197)
        #self.linear_layer1 = nn.Linear(1280, 3)
        # 可学习的损失权重
        self.image_loss_weight = nn.Parameter(torch.tensor(2.0))
        self.text_loss_weight = nn.Parameter(torch.tensor(2.0))
        self.label_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        #self.image1_loss_weight = nn.Parameter(torch.tensor(1.0))
    def forward(self, image_output, text_output, text_target,logits,labels): #encoder_out, scores, targets,logits,labels
        #loss = criterion(imgs, scores, targets)
        #oss = criterion(encoder_out, scores, targets,logits,labels)
        #(encoder_out, scores, targets,logits,labels)
        image_mapped = self.linear_layer(image_output.reshape(-1, 1280))
        #image_mapped1 = self.linear_layer1(image_output.reshape(-1, 1280))
        selected_indices = torch.randperm(image_mapped.size(0))[:108].to("cuda:1")
        #selected_indices1 = torch.randperm(image_mapped1.size(0))[:108].to("cuda:")
        image_mapped = torch.index_select(image_mapped, 0, selected_indices)
        #图片与文本的损失
        image_loss = self.contrastive_loss(image_mapped, text_output, self.margin)
        # 文本之间的损失
        text_loss =  self.cross_entropy_loss(text_output, text_target)
        #标签之间的损失
        labels_loss =  self.cross_entropy_loss(logits, labels)
        #图片与标签的损失
        #image1_loss = self.contrastive_loss(image_mapped1, logits, self.margin)
        # 使用可学习的系数调整损失
        total_loss = self.image_loss_weight * image_loss + self.text_loss_weight * text_loss + self.label_loss_weight * labels_loss
        return total_loss

    def contrastive_loss(self, output, target, margin):
        #print(output.unsqueeze(1).shape)
        #print(target.unsqueeze(0).shape)
        similarity_matrix = F.cosine_similarity(output.unsqueeze(1), target.unsqueeze(0), dim=2)
        #positive_pairs_loss = torch.diagonal(similarity_matrix, offset=0, dim1=0, dim2=1)
        positive_pairs_loss = 1 - torch.diagonal(similarity_matrix, offset=0, dim1=0, dim2=1)
        negative_pairs_loss = torch.max(similarity_matrix - margin, torch.zeros_like(similarity_matrix))
        negative_pairs_loss = torch.sum(negative_pairs_loss, dim=1) - margin
        negative_pairs_loss = negative_pairs_loss[:len(positive_pairs_loss)]
        
        loss = torch.mean(positive_pairs_loss + negative_pairs_loss)
        return loss


