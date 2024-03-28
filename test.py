from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

class AudioTranscriber:
    def __init__(self, model_path):
        """
        初始化音频转录器。
        
        参数:
            model_path: 模型和处理器的本地路径。
        """
        print("开始加载语音模型")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(model_path)

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
        # 预处理音频
        input_values = self.processor(audio_file_path, return_tensors="pt").input_values.to(self.device)

        # 执行推断
        with torch.no_grad():
            logits = self.model(input_values).logits

        # 后处理以获取识别结果
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        return transcription

# 使用示例
model_path = "./large-v3"  # 模型的本地路径
# 这里使用的是HF开源的"openai/whisper-large-v3"
# 笔者使用的 NVIDIA A100-PCIE-40GB "openai/whisper-large-v3" 运行时占用的显存位 7269MiB / 40960MiB。
transcriber = AudioTranscriber(model_path)

# 假设你有一个音频文件路径
audio_file_path = "path_to_your_audio.wav"
print("识别结果:", transcriber.transcribe(audio_file_path))
