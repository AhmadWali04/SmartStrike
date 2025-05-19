import cv2
import os
import time

def collect_images():
    
    BASE_DIR = os.getenv('SMARTSTRIKE_BASE', '.')
    MOVES_DIR = os.getenv('MOVES_DIR', os.path.join(BASE_DIR, 'Moves'))

   
    moves = ["jab", "cross", "left_hook", "right_hook", "uppercut", "kick"]

    
    os.makedirs(MOVES_DIR, exist_ok=True)
    for move in moves:
        move_folder = os.path.join(MOVES_DIR, move)
        os.makedirs(move_folder, exist_ok=True)

   
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return False

    print("Starting image collection. Press 'q' to quit at any time.")

 
    for move in moves:
        print(f"\nGet ready to perform '{move}' strikes. Capturing 100 images...")
        time.sleep(3)  
        count = 0
        while count < 100:
            ret, frame = cap.read()  
            if not ret:
                print("Error: Failed to capture frame. Exiting.")
                break  
            cv2.putText(frame, f"{move}: {count+1}/100", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Collecting - {move}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Image collection aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False
            img_path = os.path.join(MOVES_DIR, move, f"{move}_{count}.png")
            cv2.imwrite(img_path, frame)
            count += 1
        if not ret:
            break  
        print(f"Collected {count} images for '{move}'.")
        time.sleep(2)  

    cap.release()
    cv2.destroyAllWindows()
    print("Image collection completed.")
    return True

if __name__ == "__main__":
    collect_images()
