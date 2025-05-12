import pytesseract
from PIL import Image
import os 

def process_image(file_path):
    if file_path is None or not os.path.exists(file_path):
        return "No image file found."

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            return "No text detected in the image."
        #return ollama_llm("Summarize this extracted text from image", text)
        return text
    except Exception as e:
        return f"Error processing image: {str(e)}"
    
if __name__ == '__main__':
  print(process_image("./images/air-pollution.jpeg"))