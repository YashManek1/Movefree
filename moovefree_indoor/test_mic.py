import speech_recognition as sr

def list_microphones():
    mics = sr.Microphone.list_microphone_names()
    print("\n🎤 AVAILABLE MICROPHONES:")
    for i, mic_name in enumerate(mics):
        print(f"Index {i}: {mic_name}")
    print("\n⚠️ Note the Index number of your actual microphone.\n")

if __name__ == "__main__":
    list_microphones()