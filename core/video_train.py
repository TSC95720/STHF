import torch
from torch.cuda import amp


def process_vedio(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1


def foward_video(iter, base, meter, scaler, epoch):
    for _ in range(base.steps):
        input1, input2, input3, input4, label1, label2 = iter.next_one()
        base.model_optimizer.zero_grad()

        rgb_imgs, rgb_imgs_i, ir_imgs, ir_imgs_i = input1, input2, input3, input4
        rgb_pids, ir_pids = label1, label2

        rgb_imgs, rgb_imgs_i, ir_imgs, ir_imgs_i = rgb_imgs.to(base.device), rgb_imgs_i.to(base.device), ir_imgs.to(base.device), ir_imgs_i.to(base.device)
        rgb_pids, ir_pids = rgb_pids.to(base.device).long(), ir_pids.to(base.device).long()

        rgb_imgs = process_vedio(rgb_imgs)
        rgb_imgs_i = process_vedio(rgb_imgs_i)
        ir_imgs = process_vedio(ir_imgs)
        ir_imgs_i = process_vedio(ir_imgs_i)

        with amp.autocast(enabled=True):
            features, cls_score, features_i, cls_scores_i = base.model(x1=rgb_imgs, x2=ir_imgs, i1=rgb_imgs_i,
                                                                       i2=ir_imgs_i)
            pids = torch.cat([rgb_pids, ir_pids], dim=0)

            # Caculating loss
            features = features.float()
            features_i = features_i.float()
            cls_score = cls_score.float()
            cls_scores_i = cls_scores_i.float()

            ide_loss = base.pid_creiteron(cls_score, pids)
            triplet_loss = base.tri_creiteron(features.squeeze(), pids)

            ide_loss_i = base.pid_creiteron(cls_scores_i, pids)
            triplet_loss_i = base.tri_creiteron(features_i.squeeze(), pids)

            total_loss = ide_loss + triplet_loss + ide_loss_i + triplet_loss_i

        scaler.scale(total_loss).backward()
        scaler.step(base.model_optimizer)
        scaler.update()

        meter.update({'pid_loss': ide_loss.data,
                      'tri_loss': triplet_loss.data,
                      'pid_loss_i': ide_loss_i.data,
                      'tri_loss_i': triplet_loss_i.data,
                      })
    return meter
