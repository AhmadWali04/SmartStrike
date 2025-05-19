from collectImages import collect_images
from create_dataset import create_dataset
from train_classifier import train_model
from inference_classifier import run_inference

def main():
    mode = input("Enter 'T' for Training mode or 'W' for Watch mode: ").strip().lower()
    if mode.startswith('t'):
        print("** Training Mode Selected **")
        if collect_images() is False:
            print("Image collection aborted. Training process terminated.")
            return
        create_dataset()
        train_model()
        print("Training completed. You can now run Watch mode to use the model.")
    elif mode.startswith('w'):
        print("** Watch Mode Selected ** (Real-time Strike Classification)")
        run_inference()
    else:
        print("Invalid selection. Please run again and choose 'T' or 'W'.")

if __name__ == "__main__":
    main()
