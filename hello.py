from cam import Camera

def main():
    print("Hello from justcompose!")
    cam = Camera(device=0, capture_mode="landmarks_coords") # device=r"C:\Users\lusca\Universidade\CV\TPs\TPFinal\JustCompose\assets\two_open_hands_example448x800.jpeg"
    cam.capture()

if __name__ == "__main__":
    main()
