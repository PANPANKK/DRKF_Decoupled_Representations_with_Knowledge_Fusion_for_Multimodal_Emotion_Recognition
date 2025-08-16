import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from models.subnets.regression_head import RegressionHead 

class AudioModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.classifier = FcEncoder(config.hidden_size,[config.hidden_size,
        #                                                 config.final_dropout,
        #                                                 config.num_labels])
        self.init_weights()
    def forward(self,input_values,attention_mask=None):
        outputs = self.wav2vec2(input_values,attention_mask=attention_mask)
        hidden_states0 = outputs[0]
        hidden_states1 = torch.mean(hidden_states0, dim=1)
        # logits = self.classifier(hidden_states1)
        # print('audio_encoder',hidden_states0.shape, hidden_states1.shape, attention_mask.shape)
        if not self.is_emotion:
            return hidden_states0, hidden_states1, self.generate_feature_attention_mask(hidden_states0)
        else:           # just to get the emotion logits
            return self.classifier(hidden_states1)

    def generate_feature_attention_mask(self, hidden_states):
        # 检查特征张量中的零填充部分
        # 假设特征张量中的零填充部分在最后一个维度上为全零
        # 这里我们检查每个序列的第一个维度是否有非零值
        feature_attention_mask = (hidden_states.sum(dim=-1) != 0).long()
        return feature_attention_mask
    
    
class AudioEncoder(nn.Module):
    r"""Speech emotion classifier."""
    def __init__(self, processor=None,audio_model:AudioModel=None, sample_rate=16000, model_path=None, is_emotion=False):
        super(AudioEncoder,self).__init__()
        if model_path is None:
            self.processor = processor
            self.audio_model = audio_model
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.audio_model = AudioModel.from_pretrained(model_path)
        self.sample_rate=sample_rate
        self.audio_model.is_emotion=is_emotion

        if is_emotion:
            # 冻结情感编码器的参数
            for param in self.parameters():
                param.requires_grad = False

    def forward(self,audio_inputs):
        # inputs = self.processor(audio_inputs, sampling_rate=self.sample_rate,return_attention_mask=True, padding=True)#, return_tensors='pt')
        inputs = self.processor(audio_inputs, sampling_rate=self.sample_rate,return_attention_mask=True, padding=True, return_tensors='pt')
        input_values = inputs['input_values']#[0]  # 输入的音频特征 (input_ids)
        attention_mask = inputs['attention_mask']#[0]  # attention mask
        # print(input_values)
        # print(attention_mask)
        # return self.audio_model(input_values,attention_mask)#feat,feat_cls
        return self.audio_model(input_values.cuda(),attention_mask.cuda())#feat,feat_cls
