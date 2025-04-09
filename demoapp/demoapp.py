import gradio as gr
from fastrtc import (
    WebRTC,
    AlgoOptions,
    ReplyOnPause, AdditionalOutputs,
    audio_to_bytes,
)
import re
import numpy as np
import soundfile as sf
from langchain.chat_models import init_chat_model
from source.whisperclient import WhisperClient
from source.orpheusclient import OrpheusClient


# Create the Whisper client.
whisper_client = WhisperClient(base_url="http://localhost:8000")

# Create the LLM.   
chat_model_parameters = {}
chat_model_parameters["max_tokens"] = 8192
chat_model_parameters["temperature"] = 1.0
chat_model_parameters["model_provider"] = "ollama"
chat_model_parameters["model"] = "gemma3:27b"
llm = init_chat_model(**chat_model_parameters)

# Create the Orpheus client.
orpheus_client = OrpheusClient(api_url="http://localhost:5005", voice="leo")


class ChatApplication:
    """
    This class is responsible for the chat application.
    """

    def __init__(self, configuration=None):

        # Store the configuration, then initialize the agent, and build the interface.
        self.__configuration = configuration
        self.__build_interface()


    def __build_interface(self):

        with gr.Blocks() as self.demo:

            # Define the parameters for the audio stream.
            algo_options = AlgoOptions(
                audio_chunk_duration=0.6,
                started_talking_threshold=0.2,
                speech_threshold=0.1, # 0.1 is default
            )

            # Add the chat window.     
            chatbot = gr.Chatbot(type="messages", label="Chatbot")

            # Add the audio stream.
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
            )
            audio.stream(
                fn=ReplyOnPause(self.response_on_pause, algo_options=algo_options),
                inputs=[audio, chatbot],
                outputs=[audio],
                time_limit=3600,
            )
            audio.on_additional_outputs(
                lambda x: (x), # 
                outputs=[chatbot],
                queue=True, 
                show_progress="hidden"
            )
                    
                   
    def response_on_pause(
            self,
            audio: tuple[int, np.ndarray],
            chatbot
        ):
        """
        This function is called when the user pauses the audio stream.
        It transcribes the audio using Whisper, generates a response using the LLM,
        and synthesizes the response using Orpheus.
        """

        # Get the sample rate and audio data.
        sample_rate = audio[0]

        # Transcribe the audio using Whisper.
        audio_data = audio_to_bytes(audio)
        audio_file_path = "audio.wav"
        with open(audio_file_path, "wb") as file:
            file.write(audio_data)
        text = whisper_client.transcribe(
            file=audio_file_path,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        ).text  

        # Add the user message to the chatbot.
        chatbot += [gr.ChatMessage(role="user", content=text)]
        yield (sample_rate, np.zeros(1, dtype=np.float32)), AdditionalOutputs(chatbot)

        # Emotion tags
        emotion_tags = {
            "<laugh>": "laughter",
            "<sigh>": "sigh",
            "<chuckle>": "chuckle",
            "<cough>": "cough",
            "<sniffle>": "sniffle",
            "<groan>": "groan",
            "<yawn>": "yawn",
            "<gasp>": "gasp",
        }

        # Write the sysem prompt.
        system_prompt = "You are a helpful assistant. You give short, concise answers. You do not use any emojis."
        system_prompt += " You can use the following tags to indicate emotions: "
        system_prompt += ", ".join(emotion_tags.keys())
        system_prompt += "."

        # Use the transcribed text to generate a response.
        llm_messages = [{"role": "system", "content": system_prompt}]
        for message in chatbot:
            if isinstance(message, dict):
                llm_messages.append({"role": message["role"], "content": message["content"]})
            else:
                llm_messages.append({"role": message.role, "content": message.content})
        reply = llm.invoke(llm_messages).content

        # Add the reply to the chatbot.
        reply_sanitized = reply
        for tag, _ in emotion_tags.items():
            reply_sanitized = reply_sanitized.replace(tag, "")
        reply_sanitized = reply_sanitized.strip()
        chatbot += [gr.ChatMessage(role="assistant", content=reply_sanitized)]
        yield (sample_rate, np.zeros(1, dtype=np.float32)), AdditionalOutputs(chatbot)

        # Split reply into sentences. Split on periods, exclamation marks, and question marks.
        sentences = re.split(r'(?<=[.!?]) +', reply)

        # Synthesize the sentences using Orpheus.
        for i, sentence in enumerate(sentences):
            output_file = "output.wav"
            output_file = orpheus_client.synthesize(sentence, output_file)
            audio, sample_rate = sf.read(output_file)
            audio = audio.astype(np.float32)
            yield (sample_rate, audio), AdditionalOutputs(chatbot)


# Start the application.
application = ChatApplication()
application.demo.launch()
