# %%
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pickle
import math
from pathlib import Path
from transformers import BertTokenizer,BertModel,Wav2Vec2ForCTC, Wav2Vec2Processor
import sys
# sys.path.append("..")
from utils import *
import librosa
import os
from .custom_wav2vec import CustomWav2Vec2ForCTC

current_root=get_current_root(__file__)

dataset_path=Path(rf'/path/to/m3ed')

#raw data info
filenames=set(file_list(dataset_path/'audio'))
depreciated_file_list=set(read_text_list(dataset_path/'depreciated_file_list.txt'))
annotations_dict=read_json_to_dict(dataset_path/'annotation.json')
# choose_gpu()
device=get_device()

#提取各个电视剧的标注信息
dia_names=[key for key in annotations_dict.keys()]
annotations=[DotDict(annotation) for opera,annotation in annotations_dict.items()]
label_dict=read_json_to_dict(dataset_path/'splitInfo/emotions.json')
def get_data_list():
    # 用于根据annotation划分数据集
    train_dia=set(read_text_list(dataset_path/'splitInfo/movie_list_train.txt'))
    test_dia=set(read_text_list(dataset_path/'splitInfo/movie_list_test.txt'))
    val_dia=set(read_text_list(dataset_path/'splitInfo/movie_list_val.txt'))
    data_list={
        'train':[],
        'test':[],
        'val':[]
        }
    
    for annotation in annotations:
        for dia_name,dia in zip(dia_names,annotation.values()):
            #各个电视剧的dia
            dia=DotDict(dia).Dialog

            #根据剧名划分集合
            stage=''
            if dia_name in train_dia:
                stage='train'
            elif dia_name in test_dia:
                stage='test'
            elif dia_name in val_dia:
                stage='val'
            else:
                continue 

            #utt
            for filename,anno_info in dia.items():
                anno_info=DotDict(anno_info)
                filename=f'{anno_info.Speaker}_{filename}.wav'
                if filename not in filenames or filename in depreciated_file_list:continue

                file_path=dataset_path/f'audio/{filename}'
                # speech_array, sampling_rate = librosa.load(file_path, sr=16000)
                
                emo_labels=anno_info.EmoAnnotation.final_mul_emo.split(',')
                multi_hot=[0 for i in range(len(label_dict))]
                for emo_label in emo_labels:
                    multi_hot[label_dict[emo_label]]=1.0

                data_list[stage].append(DotDict({
                    'filename':filename,
                    'file_path':file_path,
                    'speaker':anno_info.Speaker,
                    'start':anno_info.StartTime,
                    'end':anno_info.EndTime,
                    'text':anno_info.Text,
                    'label':np.array(multi_hot)
                }))
    return data_list


class M3ED_Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list=data_list
        self.audio,self.text, self.labels = self._get_data()

    def _get_data(self):
        # file——path是待读取的文件路径
        # audio =[audio_process_func(librosa.load(path=item.file_path,sr=16000)[0],16000)[0] for item in self.data_list]#[(input,cls,dimension)]
        audio =[item.file_path for item in self.data_list]
        text = [item.text for item in self.data_list]
        labels = [item.label for item in self.data_list]      
        return audio, text, labels
    
    def _get_audio_embedding(self,index):
        # try:
        return audio_process_func(librosa.load(path=self.audio[index],sr=16000)[0],16000)[0].squeeze(0)  # 去除 batch 维度
        # except:
        #     err_list.append(self.audio[index])
        #     print(f"dataloader _get_audio_embedding wav2vec error: {self.audio[index]}")
        #     return np.array(0)
        #     # exit()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text=self.text[index]
        audio=self._get_audio_embedding(index)#librosa.load(path=self.audio[index],sr=16000)[0]
        label=self.labels[index]
        

        return DotDict({
            'text': text,
            'audio': audio,
            'label': label,
            'file_path':self.audio[index]
        })
    
    def _get_label_input(self):
        labels_embedding = np.arange(len(label_dict))
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

pretrained_path=Path(rf'/path/to/model_dir')
# RoBERTa 配置
roberta_tokenizer = BertTokenizer.from_pretrained(pretrained_path/'hfl/chinese-roberta-wwm-ext-large')
roberta_model = BertModel.from_pretrained(pretrained_path/'hfl/chinese-roberta-wwm-ext-large').to(device)

# wav2vec配置
# wav2vec_model = Wav2Vec2ForCTC.from_pretrained(pretrained_path/"jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn").to(device)
wav2vec_model = CustomWav2Vec2ForCTC.from_pretrained(pretrained_path/"jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn").to(device)
# print(wav2vec_model)
wav2vec_processor = Wav2Vec2Processor.from_pretrained(pretrained_path/"jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
wav2vec_model.eval()

#以batch为单位
def collate_fn(batch):

    def _get_text_embedding(text_inputs):

        # # 动态获取 max_length
        tokenized_texts = [roberta_tokenizer.encode(text, add_special_tokens=False) for text in text_inputs]
        max_length = max(len(tokens) for tokens in tokenized_texts) + 2  # 加上特殊标记的长度

        # 获取文本的词嵌入
        text_inputs = roberta_tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True,max_length=max_length).to(device)
        # text_inputs = roberta_tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = roberta_model(**text_inputs)
         # 获取词嵌入
        text_embeddings = outputs.last_hidden_state

        # 获取注意力掩码
        attention_mask = text_inputs["attention_mask"]

        return text_embeddings, attention_mask
    
    def _get_audio_embedding(audio_inputs):
        # 使用 Wav2Vec2Processor 处理音频
        audio_inputs = wav2vec_processor(
            audio_inputs,  # 音频文件列表
            return_tensors="pt",  # 返回 PyTorch tensors
            padding=True,  # 填充到最大长度
            truncation=True,  # 截断到最大长度
            sampling_rate=16000,
            max_length=max_seq_audio_len  # 最大长度限制，例如 10 秒(10*sr)，默认最长音频的
        )

        # 输出的输入特征和注意力掩码
        input_values = audio_inputs["input_values"]
        attention_mask = audio_inputs["attention_mask"]

        # 传递给 Wav2Vec2 模型以获取特征向量
        with torch.no_grad():
            model_outputs = wav2vec_model(input_values=input_values, attention_mask=attention_mask).to(device)

        # 获取特征向量
        features = model_outputs.last_hidden_state

        # 返回特征向量和注意力掩码
        return features, attention_mask


    # # 获取一个批次中每个样本的序列长度
    max_seq_audio_len= max( [sample.audio.shape[0] for sample in batch])
    # 用于存放填充后的audio_seq_input
    audio_seq_input_paddeds = []
    # 用于存放填充后的audio_seq_input的mask向量
    audio_attention_masks = []
    #填充处理
    for sample in batch:
        audio_input=sample.audio#audio_process_func(sample.audio,16000)[0].squeeze(0)  # 去除 batch 维度
        if len(audio_input.shape)==1:
            audio_input=audio_input.unsqueeze(0)
            print(f'error info:{sample.file_path}')
        audio_seq_len =audio_input.shape[0]
        audio_pad_len = max_seq_audio_len - audio_seq_len

        audio_seq_input_padded= torch.cat([audio_input, torch.zeros(audio_pad_len, audio_input.shape[1]).to(device)], dim=0)
        audio_seq_input_paddeds.append(audio_seq_input_padded)
        audio_attention_mask = torch.cat([torch.ones(audio_seq_len).to(device), torch.zeros(audio_pad_len).to(device)], dim=0)
        audio_attention_masks.append(audio_attention_mask)
    #变成torch batch张量
    audio_seq_input_paddeds = torch.stack(audio_seq_input_paddeds).to(device)
    audio_attention_masks = torch.stack(audio_attention_masks).to(device)

    #audio_seq_input_paddeds,audio_attention_masks=_get_audio_embedding([sample.audio for sample in batch])
     

    label=torch.tensor(np.array([sample.label for sample in batch])).to(device)
    # audio,audio_masks=_get_audio_embedding([sample.audio for sample in batch])
    text,text_masks=_get_text_embedding([sample.text for sample in batch])


    # text=torch.stack([sample.text for sample in batch])
    # text_mask=torch.tensor(np.stack([_get_text_mask(sample.text) for sample in batch]))
    return text,text_masks,\
        audio_seq_input_paddeds,audio_attention_masks,\
            label
    return DotDict({
        'text': text,
        'text_mask':text_masks,
        'audio': audio_seq_input_paddeds,
        'audio_attention_mask': audio_attention_masks,
        'label': label,
    })

### 将整个wav2vec计算过程封装，其中processor的作用是对语音进行编码，类似于BERT中的tokenizer.
def audio_process_func(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    y = wav2vec_processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)
    ### 这里，如果embeddings为true，则返回的y其实是池化后的hidden_states，否则是logits
    with torch.no_grad():
        y = wav2vec_model(y)
    # print(f"y[0].shape:{y[0].shape}")
    # print(f'{y["encoder_output"].shape}')
    return y["encoder_output"]
    return y[0],y[1],y[2]


#---------------------------test----------------------------------

if __name__=='__main__':
    data_list=get_data_list()
    train_set=M3ED_Dataset(data_list['train']+data_list['test']+data_list['val'])
    train_dataloader = DataLoader(
        train_set,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        # collate_fn=collate_fn
    )
    
    for step,batch in enumerate(train_dataloader):
        print(f"step:{step}\n{batch}")
        # break
    
    # list2txt(err_list,'depreciated_file_list.txt')

