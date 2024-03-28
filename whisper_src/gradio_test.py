"""
音频太长会被截断。
"""
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import gradio as gr
from gradio.data_classes import FileData
import numpy as np

class AudioTranscriber:
    def __init__(self, model_path):
        """
        初始化音频转录器。
        
        参数:
            model_path: 模型和处理器的本地路径。
        """
        print("开始加载语音模型")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(model_path, language="chinese", task="transcribe")

        # 确保使用CUDA，如果可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print("语音模型加载成功🏅🏅🏅")

    def transcribe(self, audio_file_path):
        """
        转录指定音频文件的内容。
        
        参数:
            audio_file_path: 音频文件的路径。
            
        返回:
            音频内容的文本转录。
        """
        payload = FileData(path=audio_file_path)
        # print(payload.path) # record_1213_002.wav

        audio_handle = gr.Audio()
        # sr即sampling rate(采样率), y即audio data as numpy array。
        sr, y = audio_handle.preprocess(payload)
        # (16000, array([  0,   0,   0, ..., 198, 333, 373], dtype=int16))
        y = y.astype(np.float32)
        # 归一化音频数据，使其振幅位于[-1, 1]
        max_abs_y = np.max(np.abs(y))
        # 避免除以0的问题
        if max_abs_y > 0:  # 如果最大绝对值大于0，避免执行除以0的操作
            y /= max_abs_y
        # 预处理音频
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_features.to(self.device)

        # 执行推断
        with torch.no_grad():
            generated_ids = self.model.generate(input_values)

        # 使用generated_ids而不是直接从logits获取predicted_ids
        transcription = self.processor.decode(
            generated_ids[0], 
            skip_special_tokens=True    # 使用`skip_special_tokens=True`保证输出干净。
            )

        return transcription

# 使用示例
model_path = "large-v3"  # 模型的本地路径
# 这里使用的是HF开源的"openai/whisper-large-v3"
# 笔者使用的 NVIDIA A100-PCIE-40GB "openai/whisper-large-v3" 运行时占用的显存位 7269MiB / 40960MiB。
transcriber = AudioTranscriber(model_path)

# 音频文件路径
audio_file_path = "record_1213_002.wav"
print("识别结果:", transcriber.transcribe(audio_file_path))