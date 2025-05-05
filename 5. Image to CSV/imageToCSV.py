import cv2, csv, re, pytesseract
from PIL import Image

# Path to the Tesseract executable (update this if necessary)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Example for Linux, adjust as needed

# Function to preprocess the image for better OCR
def preprocess_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to improve OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply thresholding to binarize the image
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise the image
    img = cv2.fastNlMeansDenoising(img, h=30)
    
    # Convert back to PIL Image for pytesseract
    img = Image.fromarray(img)
    return img

# Function to extract text from an image
def extract_text_from_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Configure Tesseract to recognize both English and Bengali
    custom_config = r'--oem 3 --psm 6 -l eng+ben'
    
    # Extract text from the image
    extracted_text = pytesseract.image_to_string(img, config=custom_config)
    return extracted_text

# Function to clean and split text into words/phrases for CSV
def process_text_to_csv(text, output_csv_path):
    # Split the text into lines and clean it
    lines = text.split('\n')
    processed_rows = []
    
    for line in lines:
        # Remove extra spaces and clean the line
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            # Split the line into words/phrases based on spaces and commas
            words = re.split(r'[,\s]+', line)
            # Filter out empty strings
            words = [word for word in words if word]
            processed_rows.append(words)
    
    # Write to CSV with proper UTF-8 encoding
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        for row in processed_rows:
            writer.writerow(row)

# Main function to run the program
def main():
    # Path to the input image
    image_path = '/home/rhythm/Desktop/Final/5. Image to CSV/sample_image.jpeg'
    # Path to the output CSV file
    output_csv_path = '/home/rhythm/Desktop/Final/5. Image to CSV/extracted_text.csv'
    
    # Extract text from the image
    text = extract_text_from_image(image_path)
    
    # Process the text and save it to CSV
    process_text_to_csv(text, output_csv_path)
    print(f"Text extracted and saved to {output_csv_path}")

if __name__ == "__main__":
    main()