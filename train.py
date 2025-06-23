# --- ADDED: Suppress TensorFlow informational messages ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# --- CONFIGURATION ---
KNOWN_FACES_DIR = 'dataset'
EMBEDDINGS_FILE = 'face_embeddings_final.pkl'
REQUIRED_INPUT_SIZE = (160, 160)

# --- Models ---
print("Initializing models...")
EMBEDDER = FaceNet()
DETECTOR = MTCNN()
print("✓ Models initialized.")

def extract_face(image, required_size=REQUIRED_INPUT_SIZE):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = DETECTOR.detect_faces(image_rgb)
    if not results:
        return None, None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None, None
    face_image = cv2.resize(face, required_size)
    return face_image, (x1, y1, width, height)

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, axis=0)
    return EMBEDDER.embeddings(samples)[0]

def save_embeddings(embeddings, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Embeddings saved to {filepath}")
    except Exception as e:
        print(f"✗ Error saving embeddings: {e}")

def compute_embeddings(directory):
    known_embeddings = {}
    if not os.path.exists(directory):
        print(f"✗ Directory {directory} does not exist!")
        return known_embeddings
    
    print(f"Processing dataset directory: {directory}")
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            embeddings_for_person = []
            print(f"Processing person: {person_name}")
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(person_dir, filename)
                    image = cv2.imread(path)
                    if image is not None:
                        face, _ = extract_face(image)
                        if face is not None:
                            embedding = get_embedding(face)
                            embeddings_for_person.append(embedding)
                            print(f"  ✓ Processed {filename}")
                        else:
                            print(f"  ✗ No face found in {filename}")
                    else:
                        print(f"  ✗ Could not read {filename}")
            
            if embeddings_for_person:
                # Average all embeddings for this person
                known_embeddings[person_name] = np.mean(embeddings_for_person, axis=0)
                print(f"✓ Created embedding for {person_name} from {len(embeddings_for_person)} images")
            else:
                print(f"✗ No valid embeddings found for {person_name}")
    
    return known_embeddings

def main():
    print("🔄 GENERATING FACE EMBEDDINGS...")
    
    # Compute embeddings from dataset
    embeddings = compute_embeddings(KNOWN_FACES_DIR)
    
    if embeddings:
        # Save embeddings to file
        save_embeddings(embeddings, EMBEDDINGS_FILE)
        print(f"\n✓ Successfully created embeddings for {len(embeddings)} people:")
        for name in embeddings.keys():
            print(f"  - {name}")
    else:
        print("✗ No embeddings were generated!")

if __name__ == '__main__':
    main()