from fastapi import FastAPI, Request, Response
import numpy as np
from io import BytesIO
from whisper import ASR_Model
from vad import VAD
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

app = FastAPI()
model = ASR_Model()
logger.remove()
logger.add("speech_to_llm.log", rotation="1 GB", backtrace=True, diagnose=True, format="{time} {level} {message}")
# 添加中间件
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.head("/transcribe")
async def head_transcribe(request: Request):
    # 获取客户端IP地址
    client_host = request.client.host
    # 尝试获取User-Agent头部，如果不存在则返回"Unknown"
    user_agent = request.headers.get('user-agent', 'Unknown')
    # 记录信息
    logger.info(f"HEAD request to /transcribe from {client_host} with User-Agent: {user_agent}")
    # 针对head方法的请求,返回为空
    return Response(content=None, status_code=200)

@app.post("/transcribe")
async def transcribe_audio(request: Request):
    logger.info(f"\n开始进行语音处理")
    SAMPLE_RATE = 16000
    frame_duration = 1.0
    continuous_no_speech_threshold = 0.8
    prefix_retention_length=0.8
    min_audio_length = 1.0
    max_audio_length = 30.0
    vad_threshold = 0.5 
    history_audio_buffer = RingBuffer(1)
    #初始化流切片
    stream_slicer = StreamSlicer(frame_duration=frame_duration,
                                 continuous_no_speech_threshold=continuous_no_speech_threshold,
                                 min_audio_length=min_audio_length,
                                 max_audio_length=max_audio_length,
                                 prefix_retention_length=prefix_retention_length,
                                 vad_threshold=vad_threshold,
                                 sampling_rate=SAMPLE_RATE) 

    async def generate_results():
        empty_chunks_count = 0  # 用于记录空音频块的数量
        max_allowed_empty_chunks = 3  # 允许的最大空音频块数量，可以根据实际情况调整
        async for chunk in request.stream():
            # print(chunk)
            audio_stream = BytesIO(chunk)
            audio = np.frombuffer(audio_stream.getvalue(), np.int16).flatten() / 32768.0
            if audio.size == 0:
                break
            #     empty_chunks_count += 1
            # # 当连续接收到过多空音频块时，可能表示结束或出现问题
            #     if empty_chunks_count > max_allowed_empty_chunks:
            #         logger.info("接收到过多的空音频块，结束处理。")
            #         # print("接收到过多的空音频块，结束处理。")
            #         break
            #     continue  # 忽略当前的空音频块，继续处理下一个块

        # 如果到达这里，说明当前音频块非空，重置空音频块计数器
            empty_chunks_count = 0
            stream_slicer.put(audio)
            if stream_slicer.should_slice():
                # 解码音频
                sliced_audio = stream_slicer.slice()
                history_audio_buffer.append(sliced_audio)
                result = model.forward(np.concatenate(history_audio_buffer.get_all()))
                logger.info(f"\n解码的结果为：{result}")
                # print(result)

                yield f"data:{result}\n\n"
            logger.info(f"\n")

    return StreamingResponse(generate_results(), media_type="text/event-stream")

class RingBuffer:

    def __init__(self, size):
        self.size = size
        self.data = []
        self.full = False
        self.cur = 0

    def append(self, x):
        if self.size <= 0:
            return
        if self.full:
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.size
        else:
            self.data.append(x)
            if len(self.data) == self.size:
                self.full = True

    def get_all(self):
        """ Get all elements in chronological order from oldest to newest. """
        all_data = []
        for i in range(len(self.data)):
            idx = (i + self.cur) % self.size
            all_data.append(self.data[idx])
        return all_data

    def has_repetition(self):
        prev = None
        for elem in self.data:
            if elem == prev:
                return True
            prev = elem
        return False

    def clear(self):
        self.data = []
        self.full = False
        self.cur = 0

class StreamSlicer:
    def __init__(self, frame_duration, continuous_no_speech_threshold, min_audio_length,
                 max_audio_length, prefix_retention_length, vad_threshold, sampling_rate):
        self.vad = VAD()  # Assuming VAD is correctly implemented
        self.frame_duration = frame_duration  # Duration of each audio frame in seconds
        self.sampling_rate = sampling_rate
        self.frame_samples = int(sampling_rate * frame_duration)  # Number of samples per frame

        # Convert time-based parameters to frame counts
        self.continuous_no_speech_threshold_frames = int(continuous_no_speech_threshold / frame_duration)
        self.min_audio_length_frames = int(min_audio_length / frame_duration)
        self.max_audio_length_frames = int(max_audio_length / frame_duration)
        self.prefix_retention_frames = int(prefix_retention_length / frame_duration)

        self.vad_threshold = vad_threshold
        self.audio_buffer = []
        self.prefix_audio_buffer = []
        self.speech_frames = 0
        self.no_speech_frames = 0
        self.continuous_no_speech_frames = 0

    def put(self, audio_frame):
        is_speech = self.vad.is_speech(audio_frame, self.vad_threshold, self.sampling_rate)
        if is_speech:
            self.speech_frames += 1
            self.continuous_no_speech_frames = 0
        else:
            self.no_speech_frames += 1
            self.continuous_no_speech_frames += 1

        self.audio_buffer.append(audio_frame)

        # Check for slicing conditions
        if self.should_slice():
            self.slice()

    def should_slice(self):
        total_frames = len(self.audio_buffer)
        return (
            total_frames >= self.min_audio_length_frames or
            total_frames >= self.max_audio_length_frames or
            self.continuous_no_speech_frames >= self.continuous_no_speech_threshold_frames
        )

    def slice(self):
        # 合并prefix_audio_buffer和audio_buffer以形成完整的音频片段
        concatenated_buffer = self.prefix_audio_buffer + self.audio_buffer
        concatenated_audio = np.concatenate(concatenated_buffer)
        
        # 更新prefix_audio_buffer为最后几帧，用于下一段音频的开头
        if len(self.audio_buffer) > self.prefix_retention_frames:
            self.prefix_audio_buffer = self.audio_buffer[-self.prefix_retention_frames:]
        else:
            self.prefix_audio_buffer = self.audio_buffer
        
        # 重置audio_buffer和相关计数器
        self.audio_buffer = []
        self.speech_frames = 0
        self.no_speech_frames = 0
        self.continuous_no_speech_frames = 0
        
        # 返回切分后的音频段，此处可以根据需要返回更多信息，例如时间戳等
        return concatenated_audio

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11143)

