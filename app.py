from fastapi import FastAPI,Request ## send of req
from pydantic import BaseModel ## format of data
from transformers import T5ForConditionalGeneration,T5Tokenizer # pretained model 
import torch # the connection device
import re # data cleanig
from fastapi.templating import Jinja2Templates # show of ui
from fastapi.responses import HTMLResponse # fastapi req of end point of html
from fastapi.staticfiles import StaticFiles # fastapi response of summmary

# initialize our fastapi app
app = FastAPI(title = "Text Summarizer App",description = " Text Summarization using T5",Version = "1.0")

# Load Model And Tokenizer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "_save_summary_model")

model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# DEVICE
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# template
templates = Jinja2Templates(directory=".")

# Input Schema for dialogue (schema -> format of data)
class DialogueInput(BaseModel):
    dialogue: str

# clean data function
def clean_data(text):
    text = re.sub(r"\r\n"," ",text) #line
    text = re.sub(r"\s+"," ",text) # spaces
    text = re.sub(r"<.*?>"," ",text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

# summary function
def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary

# Api Endpints

# GET (/) => send request server to client
# post (/summarize/) => send request client to server
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    dialogue = dialogue_input.dialogue
    summary = summarize_dialogue(dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})