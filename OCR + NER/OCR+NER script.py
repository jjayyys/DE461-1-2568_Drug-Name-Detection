import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from rapidfuzz import process, fuzz, distance  # Fuzzy Matching and Performance Metrics
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# 1. MODEL & IMAGE PATHS
# ------------------------------------------
# Path of best.pt (e.g., 'runs/detect/train/weights/best.pt')
# Use paths relative to this script so it works regardless of current working dir
BASE_DIR = Path(__file__).resolve().parent
print(f"Base Directory: {BASE_DIR}")
# Default model path (adjusted to where models live in this workspace)
MODEL_PATH = str((BASE_DIR.parent / "models" / "YOLOv9t" / "yolov9t_best_model.pt").resolve())

# Default image path (test_images lives next to this script)
IMAGE_PATH = str((BASE_DIR / "test_images" / "PARACETAMOL.jpg").resolve())

# 2. GROUND TRUTH (for Performance Metrics)
# ------------------------------------------
# ใส่ข้อความที่ถูกต้องของภาพนั้นๆ เพื่อใช้วัดค่า CER/WER (ถ้าไม่ใส่ จะข้ามการคำนวณ Metrics)
# เช่นถ้าภาพคือยา Paracetamol ให้ใส่ "PARACETAMOL"
GROUND_TRUTH = "PARACETAMOL"  # [ACTION REQUIRED] or None

# 3. DATABASE (for NER / Spell Check)
# ------------------------------------------
# Drug Name (Dictionary) for Post-processing
KNOWN_DRUG_NAMES = [
    "PARACETAMOL", "AMLODIPINE", "IBUPROFEN", "CREMAFFIN", 
    "LIPITOR", "NEXIUM", "PLAVIX", "PROPRANOLOL",
    "RISPERDAL", "TYLENOL", "VALPROIC ACID", "ZANTAC", "CYPROHEPTADINE",
    "PROGESTERONE", "ITRACONAZOLE", "CHOLECALCIFEROL"
]

# ==========================================
# 1. SYSTEM SETUP
# ==========================================
# Load EasyOCR (Load once, use many times)
print("Loading OCR Engine...")
reader = easyocr.Reader(['en'], gpu=True) 

# ==========================================
# 2. METRICS CALCULATION (Evaluation)
# ==========================================

def calculate_metrics(prediction, ground_truth):
    """
    Calculate Performance Metrics (OCR Evaluation):
    1. CER (Character Error Rate): ความผิดพลาดระดับตัวอักษร
    2. WER (Word Error Rate): ความผิดพลาดระดับคำ
    3. Exact Match: ตรงกันเป๊ะหรือไม่
    4. F1 Score (Character Level): ค่าความแม่นยำเฉลี่ย
    """
    if not ground_truth:
        return None

    pred_clean = prediction.upper().strip()
    gt_clean = ground_truth.upper().strip()

    # --- 1. Character Error Rate (CER) ---
    # ใช้ Levenshtein distance หารด้วยความยาวของ ground truth
    char_dist = distance.Levenshtein.distance(pred_clean, gt_clean)
    cer = char_dist / max(len(gt_clean), 1)

    # --- 2. Word Error Rate (WER) ---
    pred_words = pred_clean.split()
    gt_words = gt_clean.split()
    # ใช้ Levenshtein แบบ list (token level) จำลองการหา WER
    # (Rapidfuzz หรือไลบรารีทั่วไปมักใช้ Logic เดียวกันสำหรับ sequence)
    word_dist = distance.Levenshtein.distance(pred_words, gt_words)
    wer = word_dist / max(len(gt_words), 1)

    # --- 3. Exact Match Rate (EMR) ---
    exact_match = 1.0 if pred_clean == gt_clean else 0.0

    # --- 4. F1 Score (Character Level Approximation) ---
    # คำนวณแบบง่ายจาก LCS (Longest Common Subsequence) หรือ Similarity
    # Rapidfuzz ratio คืนค่า 0-100 ซึ่งคล้ายคลึงกับ F1 ในเชิง Similarity
    similarity = fuzz.ratio(pred_clean, gt_clean) / 100.0
    
    return {
        "CER": cer,
        "WER": wer,
        "Exact_Match": exact_match,
        "Similarity_Score": similarity  # ใช้แทน F1/Accuracy ในเชิง Similarity
    }

# ==========================================
# 3. PREPROCESSING & NER LOGIC
# ==========================================

def preprocess_for_ocr(img_crop):
    """
    Best Practice Preprocessing for OCR:
    Grayscale -> Upscale -> Denoise -> Binarization
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Upscaling (สำคัญสำหรับตัวอักษรเล็กบนขวดยา)
    scale_factor = 2
    upscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 3. Denoising (ลดจุดรบกวนภาพถ่าย)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 4. Adaptive Thresholding (Binarization)
    # แปลงเป็นขาว-ดำ โดยคำนวณค่า threshold แยกตามพื้นที่ (ดีกว่าแบบ Global threshold)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

def ner_post_processing(ocr_text, db_list, threshold=60):
    """
    NER Correction using Fuzzy Logic:
    แก้ไขคำผิดจาก OCR โดยเทียบกับฐานข้อมูลยา
    """
    if not ocr_text:
        return ocr_text, 0, False

    clean_text = ocr_text.upper().strip()
    
    # หาคำที่ใกล้เคียงที่สุดใน DB
    result = process.extractOne(clean_text, db_list, scorer=fuzz.token_sort_ratio)
    
    if result:
        best_match, score, _ = result
        # ถ้าความมั่นใจเกิน Threshold ให้แทนที่ด้วยคำที่ถูกต้องทันที
        if score >= threshold:
            return best_match, score, True # True = มีการแก้ไข
            
    return ocr_text, 0, False

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    if not MODEL_PATH or not IMAGE_PATH:
        print("❌ Error: กรุณาระบุ MODEL_PATH และ IMAGE_PATH ในส่วน Configuration ด้านบนก่อนรัน")
        return

    # --- Step 1: Load YOLO Model ---
    print(f"Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # --- Step 2: Load Image ---
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ Error: ไม่สามารถอ่านไฟล์รูปภาพได้ ตรวจสอบ Path อีกครั้ง")
        return
    
    original_h, original_w = image.shape[:2]
    
    # --- Step 3: Inference (Detection) ---
    results = model(image)[0]
    output_image = image.copy()
    
    print(f"\n🔍 Found {len(results.boxes)} detections.")
    
    for i, box in enumerate(results.boxes):
        # 3.1 Get Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # 3.2 Padding (Best Practice)
        # ขยายกรอบเล็กน้อยเพื่อให้ OCR เห็นขอบตัวหนังสือครบถ้วน
        pad = 10
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(original_w, x2 + pad)
        y2_p = min(original_h, y2 + pad)
        
        cropped_img = image[y1_p:y2_p, x1_p:x2_p]
        
        # 3.3 Image Preprocessing
        processed_img = preprocess_for_ocr(cropped_img)
        
        # 3.4 OCR Extraction
        ocr_result = reader.readtext(processed_img, detail=0, paragraph=True)
        raw_text = " ".join(ocr_result)
        
        # 3.5 NER / Correction
        final_text, ner_score, is_corrected = ner_post_processing(raw_text, KNOWN_DRUG_NAMES)
        
        # 3.6 Metrics Evaluation (ถ้ามี Ground Truth)
        metrics_display = ""
        if GROUND_TRUTH:
            metrics = calculate_metrics(final_text, GROUND_TRUTH)
            metrics_display = (f" | CER: {metrics['CER']:.2f}, WER: {metrics['WER']:.2f}, "
                               f"Sim: {metrics['Similarity_Score']:.2f}")
        
        # --- Display Results in Console ---
        print(f"\n--- Detection #{i+1} (Conf: {conf:.2f}) ---")
        print(f"   Original OCR : '{raw_text}'")
        if is_corrected:
            print(f"   NER Corrected: '{final_text}' (Confidence: {ner_score}%)")
        else:
            print(f"   Final Result : '{final_text}'")
            
        if metrics_display:
            print(f"   [Metrics]    : {metrics_display}")
            
        # --- Visualization on Image ---
        # วาดกรอบ
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # เขียนข้อความ (ถ้าแก้ NER ได้ให้โชว์คำที่แก้)
        label_text = f"{final_text}"
        cv2.putText(output_image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Save Result ---
    output_filename = f"{GROUND_TRUTH}_result.jpg"
    cv2.imwrite(output_filename, output_image)
    print(f"\n✅ Processing Complete. Result saved as '{output_filename}'")

if __name__ == "__main__":
    main()