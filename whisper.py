from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
import torch
from config import Config

class ASR_Model:
    def __init__(self):
        config = Config()
        processor = AutoProcessor.from_pretrained(config.model_path, torch_dtype=torch.float16, local_files_only=True, language="chinese")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                                            config.model_path,
                                            attn_implementation="flash_attention_2", torch_dtype=torch.float16, local_files_only=True
                                        )
        model.cuda()
        generate_kwargs = {"task": "transcribe", "num_beams": 1}
        generate_kwargs["language"] = "chinese"
        self.infer_pipe = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      max_new_tokens=128,
                      chunk_length_s=30,
                      batch_size=2,
                      torch_dtype=torch.float16,
                      generate_kwargs=generate_kwargs,
                      device="cuda")

    def forward(self, audio_np):

        result = self.infer_pipe(audio_np, return_timestamps=True)
        text = result['text']
        return text
