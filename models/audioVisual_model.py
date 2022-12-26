import torch
import torch.nn.functional as F
from torch.autograd import Variable


class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        self.lip_net, self.face_net, self.unet, self.FRNet = nets[0], nets[1], nets[2], nets[3]

    def _get_mask(self, pred_spec, mix_spec):
        mask_real = (pred_spec[:, 0, :, :] * mix_spec[:, 0, :, :] + pred_spec[:, 1, :, :] * mix_spec[:, 1, :, :]) / \
                    (mix_spec[:, 0, :, :] * mix_spec[:, 0, :, :] + mix_spec[:, 1, :, :] * mix_spec[:, 1, :, :] + 1e-30)
        mask_imag = (pred_spec[:, 1, :, :] * mix_spec[:, 0, :, :] - pred_spec[:, 0, :, :] * mix_spec[:, 1, :, :]) / \
                    (mix_spec[:, 0, :, :] * mix_spec[:, 0, :, :] + mix_spec[:, 1, :, :] * mix_spec[:, 1, :, :] + 1e-30)
        mask = torch.cat((mask_real.unsqueeze(1), mask_imag.unsqueeze(1)), 1)
        mask.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)  # B, 2, 257, 256
        return mask

    def _step1_sep(self, audio_spec, visual_feature, activation, scalar):
        pred_mask = scalar * self.unet(audio_spec, visual_feature, activation)  # (B, 2, 257, 256)
        return pred_mask

    # input mask
    def _step2_refine(self, pred_masks, visual_features, num_speakers):
        # preds: (B, 2, 257, 256), visual_features: (B, 640, 64);  输入mask,输出mask
        pred_masks_clone = pred_masks.clone()  # (B, 2, 256, 256)
        cumsum = 0
        for n in num_speakers:  # 2, 2, 3, 4, 5, 2, 2, 3, 4, 5
            pred_masks_clone[cumsum:cumsum+n] = self.FRNet(pred_masks[cumsum:cumsum+n], visual_features[cumsum:cumsum+n])  # (B, 2, 256, 256)
            cumsum += n
        return pred_masks_clone

    def _step2_refine_test(self, pred_masks, visual_features):
        # preds: (total_seg, num_speakers, 2, 257, 256), visual_features: (total_seg, num_speakers, 640, 64)
        pred_masks_clone = pred_masks.clone()
        for n in range(len(pred_masks)):
            pred_masks_clone_tmp = self.FRNet(pred_masks[n], visual_features[n])
            pred_masks_clone[n] = pred_masks_clone_tmp
        return pred_masks_clone

    def _get_spec(self, mask_prediction, audio_spec):
        # mask_prediction: (B, 2, 256, 256),  audio_spec: (B, 2, 257, 256)
        pred_spec_real = audio_spec[:, 0, :-1, :] * mask_prediction[:, 0, :, :] - audio_spec[:, 1, :-1, :] * mask_prediction[:, 1, :, :]
        pred_spec_imag = audio_spec[:, 1, :-1, :] * mask_prediction[:, 0, :, :] + audio_spec[:, 0, :-1, :] * mask_prediction[:, 1, :, :]
        pred_spec = torch.stack((pred_spec_real, pred_spec_imag), dim=1)  # (B, 2, 256, 256)
        return pred_spec

    def _get_spec_add_dim(self, mask_prediction, audio_spec):
        # mask_prediction: (B, 2, 256, 256)
        pred_spec_real = audio_spec[:, 0, :-1, :] * mask_prediction[:, 0, :, :] - audio_spec[:, 1, :-1, :] * mask_prediction[:, 1, :, :]
        pred_spec_imag = audio_spec[:, 1, :-1, :] * mask_prediction[:, 0, :, :] + audio_spec[:, 0, :-1, :] * mask_prediction[:, 1, :, :]
        pred_spec_real = torch.cat((pred_spec_real, audio_spec[:, 0, -1:, :]), dim=1)
        pred_spec_imag = torch.cat((pred_spec_imag, audio_spec[:, 1, -1:, :]), dim=1)
        pred_spec = torch.stack((pred_spec_real, pred_spec_imag), dim=1)  # (B, 2, 257, 256)
        return pred_spec

    def _spec_add_dim(self, pred_spec, audio_spec):
        # pred_psec: (B, 2, 256, 256),  audio_spec: (B, 2, 257, 256)
        result = torch.cat((pred_spec, audio_spec[:, :, -1:, :]), dim=2)
        return result

    def _get_spec_full(self, mask_prediction, audio_spec):
        # mask_prediction: (B, 2, 257, 256),  audio_spec: (B, 2, 257, 256)
        pred_spec_real = audio_spec[:, 0, :, :] * mask_prediction[:, 0, :, :] - audio_spec[:, 1, :, :] * mask_prediction[:, 1, :, :]
        pred_spec_imag = audio_spec[:, 1, :, :] * mask_prediction[:, 0, :, :] + audio_spec[:, 0, :, :] * mask_prediction[:, 1, :, :]
        pred_spec = torch.stack((pred_spec_real, pred_spec_imag), dim=1)  # (B, 2, 257, 256)
        return pred_spec

    def forward_train(self, input):
        # print(f"rank:{dist.get_rank()}, begin forward", flush=True)
        audio_specs = input['audio_specs']  # B, 2, 257, 256
        audio_spec_mix = input['audio_spec_mix']  # B, 2, 257, 256
        mouthrois = input['mouthrois']  # B, 1, 64, 88, 88
        frames = input['frames']  # B, 3, 224, 224

        # calculate ground-truth masks
        gt_mask_real = (audio_specs[:, 0, :, :] * audio_spec_mix[:, 0, :, :] + audio_specs[:, 1, :, :] * audio_spec_mix[:, 1, :, :]) / \
                       (audio_spec_mix[:, 0, :, :] * audio_spec_mix[:, 0, :, :] + audio_spec_mix[:, 1, :, :] * audio_spec_mix[:, 1, :, :] + 1e-30)
        gt_mask_imag = (audio_specs[:, 1, :, :] * audio_spec_mix[:, 0, :, :] - audio_specs[:, 0, :, :] * audio_spec_mix[:, 1, :, :]) / \
                       (audio_spec_mix[:, 0, :, :] * audio_spec_mix[:, 0, :, :] + audio_spec_mix[:, 1, :, :] * audio_spec_mix[:, 1, :, :] + 1e-30)
        gt_masks = torch.cat((gt_mask_real.unsqueeze(1), gt_mask_imag.unsqueeze(1)), 1)
        gt_masks.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)  # B, 2, 257, 256

        if self.opt.compression_type == 'hyperbolic':
            K = self.opt.hyperbolic_compression_K
            C = self.opt.hyperbolic_compression_C
            gt_masks = K * (1 - torch.exp(-C * gt_masks)) / (1 + torch.exp(-C * gt_masks))
        elif self.opt.compression_type == 'sigmoidal':
            a = self.opt.sigmoidal_compression_a
            b = self.opt.sigmoidal_compression_b
            gt_masks = 1 / (1 + torch.exp(-a * gt_masks + b))

        # pass through visual stream and extract lip features
        lip_features = self.lip_net(Variable(mouthrois, requires_grad=False), self.opt.num_frames)  # (B, 512, 1, 64)
        # pass through visual stream and extract face features
        if self.opt.number_of_face_frames == 1:
            face_features = self.face_net(Variable(frames, requires_grad=False))
        else:
            face_features = self.face_net.forward_multiframe(Variable(frames, requires_grad=False))
        if self.opt.l2_feature_normalization:
            face_features = F.normalize(face_features, p=2, dim=1)
        # what type of visual feature to use
        face_features = face_features.repeat(1, 1, 1, lip_features.shape[-1])  # (B, 128, 1, 64)
        if self.opt.visual_feature_type == 'both':
            visual_features = torch.cat((face_features, lip_features), dim=1)
        elif self.opt.visual_feature_type == 'lip':
            visual_features = lip_features
        else:  # 'face':
            visual_features = face_features

        # audio-visual feature fusion through UNet and predict mask
        if self.opt.compression_type == 'hyperbolic':
            scalar = self.opt.hyperbolic_compression_K
            activation = 'Tanh'
        elif self.opt.compression_type == 'none':
            scalar = self.opt.mask_clip_threshold
            activation = 'Tanh'
        else:  # 'sigmoidal':
            scalar = 1
            activation = 'Sigmoid'

        if self.opt.weighted_mask_loss:
            weight = torch.log1p(torch.norm(audio_spec_mix, p=2, dim=1)).unsqueeze(1).repeat(1, 2, 1, 1)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = 1

        # refine module: input mask, output mask
        mask_predictions_pre = self._step1_sep(audio_spec_mix, visual_features, activation, scalar)  # (B, 2, 257, 256)
        pred_specs_pre = self._get_spec_full(mask_predictions_pre, audio_spec_mix)  # (B, 2, 257, 256)
        mask_predictions_aft = self._step2_refine(mask_predictions_pre, visual_features.squeeze(2), input['num_speakers'])  # (B, 2, 257, 256)
        pred_specs_aft = self._get_spec_full(mask_predictions_aft, audio_spec_mix)  # (B, 2, 257, 256)
        # pred_specs_aft = self._step2_refine(pred_specs_pre, visual_features.squeeze(2), input['num_speakers'], audio_spec_mix)  # (B, 2, 257, 256)

        pred_specs_pre = pred_specs_pre.transpose(1, 2).transpose(2, 3)  # (B, 257, 256, 2)
        pred_specs_aft = pred_specs_aft.transpose(1, 2).transpose(2, 3)  # (B, 257, 256, 2)

        output = {}
        output['mask_predictions_pre'] = mask_predictions_pre  # (B, 2, 256, 256)
        output['mask_predictions_aft'] = mask_predictions_aft  # (B, 2, 256, 256)
        output['gt_masks'] = gt_masks  # (B, 2, 257, 256)
        output['weight'] = weight
        output['audio_specs'] = audio_specs.transpose(1, 2).transpose(2, 3)  # (B, 257, 256, 2)
        output['pred_specs_pre'] = pred_specs_pre  # (B, 257, 256, 2)
        output['pred_specs_aft'] = pred_specs_aft  # (B, 257, 256, 2)
        output['num_speakers'] = input['num_speakers']
        output['indexes'] = input['indexes']

        # print(f"rank:{dist.get_rank()} finish forward", flush=True)

        return output

    def forward_test(self, input):
        audio_spec_mix = input['audio_spec_mix']  # B, 2, 257, 256
        mouthrois = input['mouthrois']  # B, 1, 64, 88, 88
        frames = input['frames']  # B, 3, 224, 224

        # pass through visual stream and extract lip features
        lip_features = self.lip_net(Variable(mouthrois, requires_grad=False), self.opt.num_frames)  # (B, 512, 1, 64)
        # pass through visual stream and extract face features
        if self.opt.number_of_face_frames == 1:
            face_features = self.face_net(Variable(frames, requires_grad=False))
        else:
            face_features = self.face_net.forward_multiframe(Variable(frames, requires_grad=False))
        if self.opt.l2_feature_normalization:
            face_features = F.normalize(face_features, p=2, dim=1)
        # what type of visual feature to use
        face_features = face_features.repeat(1, 1, 1, lip_features.shape[-1])  # (B, 128, 1, 64)
        if self.opt.visual_feature_type == 'both':
            visual_features = torch.cat((face_features, lip_features), dim=1)
        elif self.opt.visual_feature_type == 'lip':
            visual_features = lip_features
        else:  # 'face':
            visual_features = face_features

        # audio-visual feature fusion through UNet and predict mask
        if self.opt.compression_type == 'hyperbolic':
            scalar = self.opt.hyperbolic_compression_K
            activation = 'Tanh'
        elif self.opt.compression_type == 'none':
            scalar = self.opt.mask_clip_threshold
            activation = 'Tanh'
        else:  # 'sigmoidal':
            scalar = 1
            activation = 'Sigmoid'

        # refine module: input mask, output mask
        mask_predictions_pre = self._step1_sep(audio_spec_mix, visual_features, activation, scalar)  # (B, 2, 257, 256)

        num_speakers = input['num_speakers']
        mask_predictions_pre = mask_predictions_pre.reshape(-1, num_speakers, 2, 257, 256)  # total_seg, num_speakers, 2, 257, 256
        visual_features = visual_features.reshape(-1, num_speakers, 640, 1, 64).squeeze(3)  # total_seg, num_speakers, 640, 64

        mask_predictions_aft = self._step2_refine_test(mask_predictions_pre, visual_features)  # (total_seg, num_speakers, 2, 257, 256)
        total_seg, num_speakers = mask_predictions_aft.shape[:2]
        mask_predictions_aft = mask_predictions_aft.reshape(total_seg * num_speakers, 2, 257, 256)
        mask_predictions_pre = mask_predictions_pre.reshape(total_seg * num_speakers, 2, 257, 256)
        pred_specs_pre = self._get_spec_full(mask_predictions_pre, audio_spec_mix)  # (B, 2, 257, 256)
        pred_specs_aft = self._get_spec_full(mask_predictions_aft, audio_spec_mix)  # (B, 2, 257, 256)
        pred_specs_pre = pred_specs_pre.permute(0, 2, 3, 1)  # B, 257, 256, 2
        pred_specs_aft = pred_specs_aft.permute(0, 2, 3, 1)  # (B, 257, 256, 2)

        output = {}
        output['pred_specs_pre'] = pred_specs_pre  # B, 257, 256, 2
        output['pred_specs_aft'] = pred_specs_aft  # (B, 257, 256, 2)
        return output

    def forward(self, input):
        if self.opt.mode == 'test':
            return self.forward_test(input)
        else:
            return self.forward_train(input)
