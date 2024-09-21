import time
from prepro import *
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from efficientnetb0 import EncoderCNN
from Transformer_enc_h_l_lf_cs_m_loss_copy40 import MemoryTransformer
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from utils import *
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np
from pycocoevalcap.spice.spice import Spice
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
# from ContrastiveMultiModalLoss_TEST import EnhancedLoss
import torch.nn as nn
from ICL import EnhancedLoss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import itertools

seed = 1234
seed_everything(seed)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_metrics(logits, labels):
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()  # 计算预测概率
  # 计算预测概率
    predss = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    Accuracy = accuracy_score(labels, predss)
    F1 = f1_score(labels, predss, average='weighted')
    Recall = recall_score(labels, predss, average='weighted')

    return Accuracy, F1, Recall, probs, labels


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='next101_b_model_LOSS')
parser.add_argument('--model_path', type=str, default='', help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='../../../data_8/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='../../../data_8/train2014', help='directory for resized images')
parser.add_argument('--image_dir_val', type=str, default='../../../data_8/val2014', help='directory for resized images')
parser.add_argument('--caption_path', type=str, default='../../../data_8/train_caption.json',
                    help='path for train annotation json file')
parser.add_argument('--caption_path_val', type=str, default='../../../data_8/val_caption.json',
                    help='path for val annotation json file')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=1024, help='dimension of word embedding vectors')
parser.add_argument('--nhead', type=int, default=8, help='the number of heads in the multiheadattention models')
parser.add_argument('--num_layers', type=int, default=4,
                    help='the number of sub-encoder-layers in the transformer model')

parser.add_argument('--attention_dim', type=int, default=464, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int, default=1024, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--epochs_since_improvement', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--encoder_lr', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--decoder_lr', default=0.00006, type=float,
                    help="initial learning rate")
parser.add_argument('--checkpoint', type=str, default=None, help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--accumulate_best_bleu4', type=float, default=0.)
parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='fine-tune encoder')

args = parser.parse_args()


# print(args)

def main(args):
    global accumulate_bleu4, epoch, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, vocab, bleu1, bleu2, bleu3, bleu3, bleu4, rouge_1, rouge_2, rouge_l, cider, spice, Accuracy, F1, Recall

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # vocab = CustomUpickler(open(args.vocab_path, 'rb')).load()
    vocab_size = len(vocab)
    num_layers1 = 3
    num_layers = 3
    d_model = 1280
    num_heads = 8
    dff = 2048
    dropout = 0.1
    num_classes = 9
    dff1 = d_model * 4
    if args.checkpoint is None:
        decoder = MemoryTransformer(num_layers1, num_layers, d_model, num_heads, dff, num_classes, dff1, vocab_size,
                                    dropout=dropout)
        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                              lr=args.decoder_lr)
        encoder = EncoderCNN()
        encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                              lr=args.encoder_lr) if args.fine_tune_encoder else None


    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        Accuracy = checkpoint['Accuracy']
        F1 = checkpoint['F1']
        Recall = checkpoint['Recall']
        bleu1 = checkpoint['bleu-1']
        bleu2 = checkpoint['bleu-2']
        bleu3 = checkpoint['bleu-3']
        bleu4 = checkpoint['bleu-4']
        accumulate_bleu4 = checkpoint['accumulate_bleu-4']
        rouge_1 = checkpoint['rouge["rouge-1"]']
        rouge_2 = checkpoint['rouge["rouge-2"]']
        rouge_l = checkpoint['rouge["rouge-l"]']
        cider = checkpoint['cider']
        spice = checkpoint['spice']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                  lr=args.encoder_lr)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # criterion = ContrastiveTextImageLoss().to(device)
    # =
    criterion = EnhancedLoss().to(device)

    data_transforms = {
        # data_transforms是一个字典，包含了两种数据变换（'train' 和 'valid'），分别用于训练和验证数据集。
        'train':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        # transforms.ToTensor() 和 transforms.Normalize() 用于将图像数据转换为张量并进行标准化。
    }
    # Build data loader
    train_loader = get_loader(args.image_dir, args.caption_path, vocab,
                              data_transforms['train'], args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.image_dir_val, args.caption_path_val, vocab,
                            data_transforms['valid'], args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    best_accumulate_bleu4 = 0.
    best_bleu1 = 0.
    best_bleu2 = 0.
    best_bleu3 = 0.
    best_bleu4 = 0.
    best_cider = 0.
    best_rouge = {'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                  'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    best_spice = 0.
    best_Accuracy = 0.
    best_F1 = 0.
    best_Recall = 0.

    for epoch in range(args.start_epoch, args.epochs):
        '''if args.epochs_since_improvement == 20:
            break'''
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_accumulate_bleu4, recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, recent_cider, recent_rouge, recent_spice, recent_Accuracy, recent_F1, recent_Recall= validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion)

        is_best = recent_accumulate_bleu4 > best_accumulate_bleu4
        best_accumulate_bleu4 = max(recent_accumulate_bleu4, best_accumulate_bleu4)
        best_bleu1 = max(recent_bleu1, best_bleu1)
        best_bleu2 = max(recent_bleu2, best_bleu2)
        best_bleu3 = max(recent_bleu3, best_bleu3)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        best_rouge['rouge-1']['f'] = max(recent_rouge['rouge-1']['f'], best_rouge['rouge-1']['f'])
        best_rouge['rouge-2']['f'] = max(recent_rouge['rouge-2']['f'], best_rouge['rouge-2']['f'])
        best_rouge['rouge-l']['f'] = max(recent_rouge['rouge-l']['f'], best_rouge['rouge-l']['f'])
        best_accumulate_bleu4 = max(recent_accumulate_bleu4, best_accumulate_bleu4)
        best_cider = max(recent_cider, best_cider)
        best_spice = max(recent_spice, best_spice)
        best_Accuracy = max(recent_Accuracy, best_Accuracy)
        best_F1 = max(recent_F1, best_F1)
        best_Recall = max(recent_Recall, best_Recall)

        if recent_bleu1 > best_bleu1:
            best_bleu1 = recent_bleu1
        print(f'Best BLEU-1: {best_bleu1}')

        if recent_bleu2 > best_bleu2:
            best_bleu2 = recent_bleu2
        print(f'Best BLEU-2: {best_bleu2}')

        if recent_bleu3 > best_bleu3:
            best_bleu3 = recent_bleu3
        print(f'Best BLEU-3: {best_bleu3}')

        if recent_bleu4 > best_bleu4:
            best_bleu4 = recent_bleu4
        print(f'Best BLEU-4: {best_bleu4}')

        # 更新最佳 Rouge 和 CIDEr 指标
        if recent_rouge['rouge-1']['f'] > best_rouge['rouge-1']['f']:
            best_rouge['rouge-1'] = recent_rouge['rouge-1']
        print(f'Best rouge_1: {best_rouge["rouge-1"]}')
        if recent_rouge['rouge-2']['f'] > best_rouge['rouge-2']['f']:
            best_rouge['rouge-2'] = recent_rouge['rouge-2']
        print(f'Best rouge_2: {best_rouge["rouge-2"]}')
        if recent_rouge['rouge-l']['f'] > best_rouge['rouge-l']['f']:
            best_rouge['rouge-l'] = recent_rouge['rouge-l']
        print(f'Best rouge_l: {best_rouge["rouge-l"]}')
        if recent_cider > best_cider:
            best_cider = recent_cider
        print(f'Best CIDEr: {best_cider}')
        if recent_accumulate_bleu4 > best_accumulate_bleu4:
            best_accumulate_bleu4 = recent_accumulate_bleu4
        print(f'Best accumulate bleu4: {best_accumulate_bleu4}')
        if recent_spice > best_spice:
            best_spice = recent_spice
        print(f'Best spice score: {best_spice}')

        if recent_Accuracy > best_Accuracy:
            best_Accuracy = recent_Accuracy
        print(f'Best Accuracy score: {best_Accuracy}')
        if recent_F1 > best_F1:
            best_F1 = recent_F1
        print(f'Best F1 score: {best_F1}')
        if recent_Recall > best_Recall:
            best_Recall = recent_Recall
        print(f'Best Recall score: {best_Recall}')

        # 计算总指标Sm
        Sm = 1 / 7 * best_bleu4 + 2 / 7 * (best_rouge['rouge-l']['f'] + best_spice + best_cider)
        print(f'Sm: {Sm}')

        if not is_best:
            args.epochs_since_improvement += 1
            print("\nEpoch since last improvement: %d\n" % (args.epochs_since_improvement,))
        else:
            args.epochs_since_improvement = 0

    save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, best_accumulate_bleu4, is_best, best_bleu1, best_bleu2, best_bleu3,
                    best_bleu4, best_rouge, best_cider, best_spice, best_Accuracy, best_F1, best_Recall)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Use tqdm to display progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, (imgs, caps, caplens, labels) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, logits, encoder_out = decoder(imgs, caps, caplens)
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = caps[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = scores.data.to(device)
            targets = targets.data.to(device)
            logits = logits.to(device)
            labels = labels.to(device)
            # print("imgs: ", imgs.shape)
            # print("scores: ", scores.shape)
            # print("targets: ", targets.shape)
            loss = criterion(encoder_out, scores, targets, logits, labels)

            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, args.grad_clip)

            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # = accuracy(logits, labels, 1)
            # top5 = accuracy(logits, labels, 5)
            losses.update(loss.item(), sum(decode_lengths))
            # top1accs.update(top1, sum(decode_lengths))
            # top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Update tqdm progress bar
            t.set_postfix(loss=losses.avg)
            t.update()
    # Average metrics

    print('Epoch: [{0}]\t'
          'Loss {loss.avg:.4f}'.format(epoch, loss=losses,
                                       ))


import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import itertools

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: accumulate-BLEU-4 score
    """
    decoder.eval()
    total_Accuracy = 0.0
    total_F1 = 0.0
    total_Recall = 0.0
    all_probs = []
    all_labels = []
    num_classes = 9
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}

    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    references = []
    hypotheses = []

    for i, (imgs, caps, caplens, labels) in enumerate(val_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, logits, encoder_out = decoder(imgs, caps, caplens)
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = caps[:, 1:]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = scores.data.to(device)
        targets = targets.data.to(device)
        logits = logits.to(device)
        labels = labels.to(device)

        Accuracy, F1, Recall, probs, label_np = compute_metrics(logits, labels)
        total_Accuracy += Accuracy
        total_F1 += F1
        total_Recall += Recall
        loss = criterion(encoder_out, scores, targets, logits, labels)
        losses.update(loss.item(), sum(decode_lengths))

        batch_time.update(time.time() - start)
        start = time.time()

        class_preds = torch.max(logits, dim=1)[1].cpu().numpy()
        preds_flat = class_preds.flatten()
        labels_flat = label_np.flatten()

        for c in range(num_classes):
            per_class_correct[c] += np.sum((preds_flat == c) & (labels_flat == c))
            per_class_total[c] += np.sum(labels_flat == c)

        _, caption_preds = torch.max(scores_copy, dim=2)
        caption_preds = caption_preds.tolist()

        all_probs.append(probs)
        all_labels.append(label_np)

        for j in range(caps_sorted.shape[0]):
            img_caps = caps_sorted[j].tolist()
            img_captions = [w for w in img_caps if w not in {vocab.__call__('<start>'), vocab.__call__('<pad>')}]
            references.append([img_captions])

        temp_preds = [caption_preds[j][:decode_lengths[j]] for j in range(len(caption_preds))]
        hypotheses.extend(temp_preds)

        if i % args.log_step == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                  loss=losses))

        assert len(references) == len(hypotheses)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    n_classes = all_probs.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = roc_auc_score(all_labels == i, all_probs[:, i])

    save_dir = './9lei/221'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use a color map to generate a sufficient number of unique colors
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 distinct colors

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curve_epoch_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(n_classes), [roc_auc[i] for i in range(n_classes)], tick_label=[f'Class {i}' for i in range(n_classes)])
    plt.xlabel('Classes')
    plt.ylabel('AUC')
    plt.title('AUC Histogram')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')
    plt.savefig(f'{save_dir}/auc_histogram_epoch_{epoch}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    accuracy_per_class = {i: per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0 for i in per_class_total}
    bars = plt.bar(accuracy_per_class.keys(), accuracy_per_class.values())
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')
    plt.savefig(f'{save_dir}/accuracy_per_class_epoch_{epoch}.png')
    plt.close()

    Accuracy = total_Accuracy / len(val_loader)
    F1 = total_F1 / len(val_loader)
    Recall = total_Recall / len(val_loader)

    accumulate_bleu4 = corpus_bleu(references, hypotheses)
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    references_text = []
    for ref in references:
        ref_text = " ".join([" ".join([vocab.idx2word[word_idx] for word_idx in cap]) for cap in ref])
        references_text.append(ref_text)

    hypotheses_text = []
    for hyp in hypotheses:
        hyp_text = " ".join([vocab.idx2word[word_idx] for word_idx in hyp])
        hypotheses_text.append(hyp_text)

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses_text, references_text, avg=True)

    refs_str = []
    for ref_list in references:
        ref_str = [' '.join([vocab.idx2word[word_idx] for word_idx in cap]) for cap in ref_list]
        refs_str.append(ref_str)

    hypotheses_str = [' '.join([vocab.idx2word[word_idx] for word_idx in hyp]) for hyp in hypotheses]
    gts = {idx: ref_list for idx, ref_list in enumerate(refs_str)}
    res = {idx: [hyp] for idx, hyp in enumerate(hypotheses_str)}

    cider = Cider()
    cider_score, _ = cider.compute_score(gts=gts, res=res)

    spice = Spice()
    spice_score, _ = spice.compute_score(gts, res)

    return accumulate_bleu4, bleu1, bleu2, bleu3, bleu4, cider_score, rouge_scores, spice_score, Accuracy, F1, Recall

if __name__ == '__main__':
    main(args)
