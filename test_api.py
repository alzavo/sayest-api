import requests
import json
import os

# Configuration
API_URL = "http://localhost:8000/predict"
AUDIO_FILE_PATH = "/home/aleksei/dev/sayest/audio/L1/e7cd-68c0-b5df-35b0_aia_take1.wav"
PHONEMES = "a i j a"
WORD = "aia"


def test_prediction():
    # 1. Check if file exists
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Error: File not found at {AUDIO_FILE_PATH}")
        return

    # 2. Prepare the payload
    # 'data' contains form fields
    payload = {"phonemes": PHONEMES, "word": WORD}

    # 'files' contains the binary audio data
    # format: 'fieldname': ('filename', open_file_handle, 'content_type')
    files = {
        "audio": (
            os.path.basename(AUDIO_FILE_PATH),
            open(AUDIO_FILE_PATH, "rb"),
            "audio/wav",
        )
    }

    try:
        print(f"Sending request to {API_URL}...")

        # 3. Send POST request
        response = requests.post(API_URL, data=payload, files=files)

        # 4. Handle Response
        if response.status_code == 200:
            print("\n✅ Success!")
            result = response.json()
            # Pretty print the JSON
            print(json.dumps(result, indent=4, ensure_ascii=False))
        else:
            print(f"\n❌ Failed with status code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to the server. Is it running?")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        # Close the file handle
        files["audio"][1].close()


if __name__ == "__main__":
    test_prediction()
