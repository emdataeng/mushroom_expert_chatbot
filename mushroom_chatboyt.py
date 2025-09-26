from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Using the `Qwen2_5_VLForConditionalGeneration` class to enable multimodal generation
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",  # automatically uses right precision based on model
    device_map="auto"  # automatically uses right device e.g. GPU if available
)

# Using the `AutoProcessor` class to enable multimodal generation
processor = AutoProcessor.from_pretrained(MODEL_NAME)
   
#Returns timetaken of the interaction in string format DD-MM-YYYY_HH-MM-SS, the time taken must be in seconds if less than a minute, in minutes if less than an hour, in hours if less than a day and in days otherwise.
def get_timestamp(time_taken):    
    if time_taken < 60:
        time_taken_str = f"{time_taken} seconds"
    elif time_taken < 3600:
        time_taken_str = f"{time_taken // 60} minutes"
    elif time_taken < 86400:
        time_taken_str = f"{time_taken // 3600} hours"
    else:
        time_taken_str = f"{time_taken // 86400} days"
    return time_taken_str

#this function records in a json log file the user prompt and the expert reply, as well as the generation parameters, the time taken to generate the reply and the model name. Each entry is created by appending to the file a a new key in the json object whose name is the timestamp of the interaction in format HH-MM-SS, the time taken variable must be in seconds if less than a minute, in minutes if less than an hour, in hours if less than a day and in days otherwise. The name of the log file is the current ime in format DD-MM-YYYY followed by _log.json. If the log file already exists, the new entry is appended to the existing file. If it does not exist, a new file is created.
import json
from datetime import datetime
import os 
def log_interaction(image_file_name, user_prompt, expert_reply, temp, topK, topP, time_taken, model_name, expert_reply_json="No Image"):
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")
    
    # Create log file name
    log_file_name = f"a1_{date_str}_log.json"
    
    # Create log entry
    log_entry = {
        "image_file_name": image_file_name,
        "user_prompt": user_prompt,
        "expert_reply_json": expert_reply_json,
        "expert_reply": expert_reply,
        "generation_parameters": {
            "temperature": temp,
            "topK": topK,
            "topP": topP
        },
        "time_taken": get_timestamp(time_taken),
        "model_name": model_name
    }
    
    # Load existing log file or create new one
    if os.path.exists(log_file_name):
        with open(log_file_name, "r") as f:
            log_data = json.load(f)
    else:
        log_data = {}
    
    # Append new entry
    log_data[time_str] = log_entry
    
    # Save updated log file
    with open(log_file_name, "w") as f:
        json.dump(log_data, f, indent=4)
    

    #this function is for logging for troubleshooting and debugging, it creates a log file named troubleshooting_log.txt and appends the message to it with a timestamp
def log_troubleshooting(message):
    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
    
    # Create log file name
    log_file_name = "a1_troubleshooting_log.txt"
    
    # Create log entry
    log_entry = f"[{timestamp}] {message}\n"
    
    # Append to log file
    with open(log_file_name, "a") as f:
        f.write(log_entry)

#System prompt    
from PIL import Image
import time
IMAGE_PATH = "./data"
image_file_name = "mushroom_copper_spike.jpg"
im = Image.open(f"{IMAGE_PATH}/{image_file_name}") #.convert("RGB")??

from transformers import GenerationConfig
from qwen_vl_utils import process_vision_info

json_response_example = """{
    "common_name": "Inkcap",
    "genus": "Coprinus",
    "confidence": 0.5,
    "visible": ["cap", "hymenium", "stipe"],
    "color": "orange",
    "edible": true
}"""

system_prompt_str = f"""
You are a mushroom expert chatbot. Your role: answer queries strictly about mushrooms using mycological terms, succinct and data-driven.

Rules:
- If you don’t know, say "I don't know." Never invent facts.
- Always include scientific and common names when known.
- Keep answers concise and evidence-focused.

Image handling & JSON (required):
- If the user sends an image with a picture of a mushroom, first output exactly one valid JSON object with these fields:
  - common_name (string or list)
  - genus (string or list)
  - family (string or list)
  - confidence (float, 0–1)
  - visible (list, choose from: ["cap","hymenium","stipe"])
  - color (string)
  - edible (bool)
  - notes (optional short string for important facts)
  Example: {json_response_example}
- JSON must be the first output and strictly parsable (no extra text inside the JSON).
- Else if the image contains a scan of a mushroom book page:
  - Extract and summarize key mushroom details in the JSON.
  - If the page has multiple mushrooms, focus on the most prominent one.
  - If the passage does not refer to a specific mushroom, extract the most relevant general info in the JSON creating a generic entry.
  - Take into account that the pages may not be in English.
- If the image is not mushroom-related, respond: "The image does not contain mushrooms."

After the JSON:
- You must always write at least one explanatory paragraph. Do not end your output right after the JSON.
- If the user asked a question when sending the image, answer it.
- If no question, produce a one-paragraph summary of the image based on the JSON.
  - If the JSON was a scan of a book page and it was not in English, translate the summary to English.

Session state rules:
- Persist the most recent image JSON for the duration of this chat.
- For any follow-up question (e.g., "Are these poisonous?", "What about the stem?"), use the stored JSON as the factual basis unless:
  - the user uploads a new image (replace stored JSON), or
  - the user explicitly asks you to forget or replace the stored JSON.
- If confidence < 0.35, state that identification is uncertain and avoid definitive claims.

Formatting:
- JSON first (parsable), then a single paragraph answer/summary.
- Tone: professional, concise, mycological.
"""

import re
def extract_json_and_explanation(output_text: str):
    """
    Extract the JSON-like block and any explanation text from the model output.
    """
    json_match = re.search(r"```json(.*?)```", output_text, re.DOTALL)

    if json_match:
        json_part = json_match.group(1).strip()
        explanation = output_text[json_match.end():].strip()
    else:
        json_part = None
        explanation = output_text.strip()

    return json_part, explanation



def call_model(user_prompt, image, temp, topK, topP, keep_in_mind=None) -> str:
    """
    Wrapper function to call the LLM.
    """
    # Conversation always starts with system prompt
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt_str}
            ],
        }
    ]
    
    # Add persistent memory into system context
    if keep_in_mind:
        conversation.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": f"Keep in mind: {json.dumps(keep_in_mind)}"}
                ],
            }
        )
    
    # Add user message (with or without image)
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})

    conversation.append({"role": "user", "content": user_content})
    
    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=500,
        temperature=temp,
        top_k=topK,
        do_sample=True,
        top_p=topP
    )
    
    # Prepare model inputs
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    
    generated_ids = model.generate(**inputs, generation_config=gen_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    
    return output_text

#Keep in mind with memoization to persist memory across calls to this function.
import time

def chatbot_interface(user_prompt, image=None, temp=0.7, topK=0, topP=0.5, image_filename=None):
    # Initialize memory (only first time) and when image is not None, in this way it keeps in mind only the last image
    if not hasattr(chatbot_interface, "keep_in_mind") or image is not None:
        chatbot_interface.keep_in_mind = {}

    # Run generation
    start_time = time.time()

    raw_output_text = call_model(user_prompt, image, temp, topK, topP, chatbot_interface.keep_in_mind)
    
    end_time = time.time() 
    time_taken = end_time - start_time
    
    # Log interaction    
    expert_reply_json,  expert_reply_explanation = extract_json_and_explanation(raw_output_text)
    
    if expert_reply_json:
        try:
            # Update persistent memory
            chatbot_interface.keep_in_mind = json.loads(expert_reply_json)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON image summary from model output.")
            expert_reply_json = "Warning: Failed to parse JSON image summary from model output."
            chatbot_interface.keep_in_mind = {}
    elif not image and expert_reply_json is None: expert_reply_json = "No image uploaded, so no explanation provided."
            
    log_troubleshooting(f"chatbot_interface - end of function")
    
    log_interaction(
        f"uploaded_image: {image_filename}" if image else "text_only",
        user_prompt,
        expert_reply_explanation,
        temp, topK, topP,
        time_taken,
        MODEL_NAME,
        expert_reply_json
    )
    
    # Return only the explanation (user-facing)
    return expert_reply_explanation


#Mushroom expert response function
import gradio as gr
import random
from PIL import Image

def response(image, question, history, image_filename):
    #model parameters
    temp=0.7
    topK=0
    topP=0.5

    if question and "hello" in question.lower():
        return "Hello! I am a mushroom expert. Ask me anything about mushrooms."
    elif question and "bye" in question.lower():
        return "Goodbye! Have a great day."
    elif image is not None: #and question.strip() != ""
        reply = chatbot_interface(question, image, temp=temp, topK=topK, topP=topP, image_filename=image_filename)    
        return reply
        #return f"I see you uploaded an image and asked: '{question}'. My guess: It's some kind of mushroom 🍄."
    elif image is None and question.strip() != "":
        reply = chatbot_interface(question, image, temp=temp, topK=topK, topP=topP)    
        return reply
    else:
        return random.choice([
            "I am not sure about that. Can you ask me something else?",
            "Could you reformulate your question? I am not sure I understand.",
            "I don't understand, can you ask someone else?",
            "What a stellar question! I am not sure about the answer though.",
            "You know what? I am not cut out for this. I am going to take a break."
        ])

# Prevent multiple servers in Jupyter
if "demo" in locals() and demo.is_running:
    demo.close()

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## 🍄 Your Personal Mushroom Expert 🍄‍🟫")
    gr.Markdown("Ask me anything about mushrooms! Upload a picture and type your question.")
    
    chatbot = gr.Chatbot()      
    
    with gr.Row():
        #image = gr.Image(type="pil", label="Upload Mushroom Image")
        image = gr.Image(type="filepath", label="Upload Mushroom Image")        
        text = gr.Textbox(label="Your Question", placeholder="Ask me anything about mushrooms...")
    btn = gr.Button("Ask the Expert")
    
    # Define the function to handle chat interactions, image it’s just a string path to the file
    def chat_fn(image, text, history):
        filename = None
        pil_img = None

        if image:
            filename = os.path.basename(image) #keep filename for logging
            pil_img = Image.open(image)  # convert path to PIL image

        reply = response(pil_img, text, history, filename)
        if text:
            prev_prompt = text
        elif pil_img is not None:
            prev_prompt = f"[Image only: {filename}]"
        else:
            prev_prompt = ""
        history = history + [(prev_prompt, reply)]
        return history, None, None  # clear inputs
    
    btn.click(chat_fn, [image, text, chatbot], [chatbot, image, text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

