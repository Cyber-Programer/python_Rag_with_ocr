from PyPDF2 import PdfReader
import os

# --- Configuration ---
# Your CV PDF file path
PDF_FILE_PATH = 'md_sifat_islam_cv.pdf' 
OUTPUT_TEXT_PATH = 'cv_extracted_text.txt'

def extract_text_from_pdf(pdf_path,output_path):
    # check if the pdf file exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file {pdf_path} dose nto exist.")
        return
    
    all_text = ""
    try:
        # Create a PDF reader object
        reader = PdfReader(pdf_path)
        for i,page in enumerate(reader.pages):
            print(f"Extracting text from page: {i+1}")
            text = page.extract_text()
            all_text += f"Page : {i+1}"+ "\n" + text + "\n\n\n" + "------Page Break------" + "\n\n\n"

        # if output path directory does not exist, create it
        output_dir = os.path.dirname(output_path)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the extracted text to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
        print(f"\n✅ Extraction complete! Text saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- USAGE ---
if os.path.exists(PDF_FILE_PATH):
    extract_text_from_pdf(PDF_FILE_PATH, OUTPUT_TEXT_PATH)
else:
    print(f"Error: PDF file not found at {PDF_FILE_PATH}. Please check the path.")
