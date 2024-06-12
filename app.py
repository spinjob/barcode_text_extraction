import os
import cv2
import pytesseract
import streamlit as st
from pyzbar.pyzbar import decode
import pandas as pd
from tempfile import NamedTemporaryFile, TemporaryDirectory
from io import StringIO

def BarcodeReader(image_path, output_folder, min_confidence=80):
    img = cv2.imread(image_path)
    detectedBarcodes = decode(img)

    min_width = 10  # Minimum width threshold for barcode region
    min_height = 10  # Minimum height threshold for barcode region
    padding = 40  # Further increased padding around the barcode for cropping

    barcode_data = None
    cropped_image_path = None
    delimited_text = ""
    barcode_region_flipped = None

    if detectedBarcodes:
        for barcode in detectedBarcodes:
            (x, y, w, h) = barcode.rect
            
            if w > min_width and h > min_height:
                # Draw rectangle with padding
                cv2.rectangle(img, (x - padding, y - padding), (x + w + padding, y + h + padding), (255, 0, 0), 2)
                
                if barcode.data != "":
                    barcode_data = barcode.data.decode('utf-8')
                    barcode_type = barcode.type
                    st.write(f"Data: {barcode_data}, Type: {barcode_type}")
                
                # Crop the barcode region with padding
                barcode_region = img[max(0, y - padding):min(y + h + padding, img.shape[0]), 
                                     max(0, x - padding):min(x + w + padding, img.shape[1])]
                
                # Flip the cropped barcode region upside down
                barcode_region_flipped = cv2.flip(barcode_region, -1)

                # Save the cropped and flipped barcode image
                cropped_image_path = os.path.join(output_folder, os.path.basename(image_path).rsplit('.', 1)[0] + "_cropped." + os.path.basename(image_path).rsplit('.', 1)[1])
                cv2.imwrite(cropped_image_path, barcode_region_flipped)
                st.write("Cropped image saved as:", cropped_image_path)

                break  # Assuming only one barcode per image
    else:
        # Use OCR to find the largest text bounding box
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        max_area = 0
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf > min_confidence:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                area = w * h
                if area > max_area:
                    max_area = area
                    best_x, best_y, best_w, best_h = x, y, w, h

        if max_area > 0:
            x, y, w, h = best_x, best_y, best_w, best_h
            # Draw rectangle with padding
            cv2.rectangle(img, (x - padding, y - padding), (x + w + padding, y + h + padding), (255, 0, 0), 2)
            # Crop the region with padding
            barcode_region = img[max(0, y - padding):min(y + h + padding, img.shape[0]), 
                                 max(0, x - padding):min(x + w + padding, img.shape[1])]
            
            # Flip the cropped region upside down
            barcode_region_flipped = cv2.flip(barcode_region, -1)

            # Save the cropped and flipped image
            cropped_image_path = os.path.join(output_folder, os.path.basename(image_path).rsplit('.', 1)[0] + "_cropped_ocr." + os.path.basename(image_path).rsplit('.', 1)[1])
            cv2.imwrite(cropped_image_path, barcode_region_flipped)
            st.write("Cropped OCR image saved as:", cropped_image_path)

    # Perform OCR on the cropped image with detailed output
    try:
        data = pytesseract.image_to_data(barcode_region_flipped, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            text = data['text'][i]
            conf = int(data['conf'][i])
            if conf > min_confidence and text.strip():
                delimited_text += f"{text}|"
        delimited_text = delimited_text.strip('|')
        st.write("Delimited Text:", delimited_text)
    except pytesseract.TesseractNotFoundError:
        st.write("Tesseract is not installed or it's not in your PATH.")

    # Save the image with the highlighted rectangle
    output_image_path = os.path.join(output_folder, os.path.basename(image_path).rsplit('.', 1)[0] + "_output." + os.path.basename(image_path).rsplit('.', 1)[1])
    cv2.imwrite(output_image_path, img)
    st.write("Output image saved as:", output_image_path)

    return os.path.basename(image_path), barcode_data, delimited_text, cropped_image_path

def process_images_to_csv(image_paths, output_folder, min_confidence=80):
    # List to hold rows for the CSV
    rows = [["image", "barcode", "text"]]

    for image_path in image_paths:
        image_name, barcode_data, delimited_text, cropped_image_path = BarcodeReader(image_path, output_folder, min_confidence)
        rows.append([image_name, barcode_data if barcode_data else "Barcode not found", delimited_text])

    # Write rows to the CSV file
    df = pd.DataFrame(rows[1:], columns=rows[0])
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return df, csv_buffer.getvalue()

# Streamlit UI
st.title("Barcode and Product Code Extractor")
st.write("Upload images to generate a CSV file with extracted barcode and text data.")
# Add a link to the Google Sheet template
st.markdown("""
<p style='font-size: 14px;'> <a href="https://docs.google.com/spreadsheets/d/10G-1cvhLBmcMP8TjM99qbszRZAZoBtA2heoZ99nZagc/edit?usp=sharing" target="_blank">Google Sheet Template for the generated CSV</a></p>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    with TemporaryDirectory() as upload_dir, TemporaryDirectory() as processed_dir:
        image_paths = []
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(temp_file_path)

        df, csv_content = process_images_to_csv(image_paths, processed_dir)

        st.write("Processing complete. Download the results:")
        st.download_button("Download CSV", data=csv_content, file_name="output.csv")

        st.write("CSV Table:")
        st.dataframe(df)

        st.write("Cropped Barcode Images:")
        for image_path in image_paths:
            cropped_image_path = os.path.join(processed_dir, os.path.basename(image_path).rsplit('.', 1)[0] + "_cropped." + os.path.basename(image_path).rsplit('.', 1)[1])
            if os.path.exists(cropped_image_path):
                st.image(cropped_image_path, caption="Cropped Barcode Image")
