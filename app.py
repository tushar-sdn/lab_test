from typing import List, Optional, Annotated,Union
import os
from fastapi.templating import Jinja2Templates
import re
import string
import contractions
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from llama_index.core import SimpleDirectoryReader
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

templates = Jinja2Templates(directory="templates")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')



app = FastAPI()
# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend's origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "/home/ambikeshsingh/Lab_test_poc/Lab_Test_AI/doc"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key="*************************************")

def get_gemini_response(file_data, input_prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_prompt, file_data])
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content: {e}")

# Define the method to clean text
def clean_text(text):
    try:
        text = text.replace('\n', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning text: {e}")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/file_upload")
async def analysis(
    text: Optional[str] = Form(None),
    files: Annotated[Union[UploadFile, None], File()] = None):
    try:
      
        joint_text = ""

        # Ensure files is always a list
        if files is not None:
            if not isinstance(files, list):
                files = [files]

            for file in files:
                if file.filename:  # Ensure the file has a name (is not empty)
                    file_location = os.path.join(UPLOAD_DIR, file.filename)
                    with open(file_location, "wb") as f:
                        f.write(await file.read())

            reader = SimpleDirectoryReader(input_dir=UPLOAD_DIR)
            documents = reader.load_data()
            clean_documents = clean_text(' '.join([doc.text for doc in documents]))
            joint_text += clean_documents

        if text:
            joint_text += ' ' + clean_text(text)
            print(text,"################")


        if not joint_text.strip():
            raise HTTPException(status_code=400, detail="No content to process")

        input_prompt = f"""Input the text below and identify the most likely lab tests needed based on the content.

{joint_text}
"""
        

        response = get_gemini_response(joint_text, input_prompt)
        # clean_response = response.replace("*", "")  # Replace all asterisks with empty string

        # Define the regular expression to match special characters
        special_char_pattern = r"[^\w\s]"  # Matches characters except letters (a-z, A-Z), numbers (0-9), and whitespace

        clean_response = re.sub(special_char_pattern, "", response)
        
        print(clean_response)
        
        directory_path = "/home/ambikeshsingh/Lab_test_poc/Lab_Test_AI/doc"


    
        print("Deleting files in directory:", directory_path)
        # List all files in the directory
        files = os.listdir(directory_path)

        # Iterate through each file and remove it
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File '{file_name}' has been successfully removed.")
   
        return JSONResponse(content={"response": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
