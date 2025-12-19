## como eu vou poder definir gestos?

Mediapipe reconhece-os nativamente

## Mas como ele funciona?

Criei uma classe para reconhecer gestos via Mediapipe, protótipo

```python
class HandSpeller():
    """
    The Gesture Recognizer
    """
    _MODEL_PATH = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/gesture_recognizer.task"
    
    def __init__(self, model = _MODEL_PATH, running_mode=vision.RunningMode.LIVE_STREAM):
        # São atributos padrões do Mediapipe, entao declaro-os como attr
        self.base_options = python.BaseOptions(model_asset_path=model)
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options, running_mode=running_mode)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
    
    def process_image(self, image):
        # Vou pegar um frame do OpenCV, convertê-lo para RGB e processá-lo MediaPipe 
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert BGR (OpenCV) -> RGB (MediaPipe Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Get the gestures
        # Mesma lógica das landmarks
        result = self.recognizer.recognize(mp_image)
        
        if not result.gestures or not result.hand_landmarks:
            return image, None, None
        
        
        top_gesture = result.gestures[0][0]  # [hand][ranking]
        gesture_name = top_gesture.category_name
        gesture_score = top_gesture.score
        # Exemplo: All gestures: [Category(index=-1, score=0.883987307548523, display_name='', category_name='Pointing_Up')] 

        print(f"Gesture: {gesture_name}, Score: {gesture_score:.2f}\n")
        for i in range(len(result.gestures)):
            print(f"All gestures: {result.gestures[i]}\n")
```
Saída esperada (detalhe ao console):

![Ronaldo Fazendo o Número 1](image-4.png)


## Print do GestureRecognizerResult

o HandSpeller retorna uma lista com as mãos, que são o seguinte objeto: 
```python
GestureRecognizerResult(
    # É uma lista de listas para suportar múltiplas mãos com múltiplos gestos.
    gestures=[
        [
            Category(
                index=-1, 
                score=0.6321459412574768, 
                display_name='', 
                category_name='Thumb_Up'  # <- O nome do gesto reconhecido
            )
        ]
    ],
    
    # Lista de classificações de 'handedness' (Mão Direita/Esquerda)
    handedness=[
        [
            Category(
                index=0, 
                score=0.7721295356750488, 
                display_name='Right', 
                category_name='Right'  # <- Classificação da mão (Direita)
            )
        ]
    ],
    
    # Landmarks da mão em coordenadas normalizadas (0.0 a 1.0)
    # Lista de listas, onde a lista interna representa uma mão detectada.
    hand_landmarks=[
        [
            # Landmark 0 (Pulso - Root)
            NormalizedLandmark(x=0.1922, y=0.7747, z=-6.52e-08, visibility=0.0, presence=0.0), 
            
            # Landmarks do DEDÃO (Thumb: 1, 2, 3, 4)
            NormalizedLandmark(x=0.1906, y=0.7037, z=-0.0146, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.1708, y=0.6097, z=-0.0229, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.1589, y=0.5426, z=-0.0296, visibility=0.0, presence=0.0),
            NormalizedLandmark(x=0.1540, y=0.4900, z=-0.0347, visibility=0.0, presence=0.0),
            
            # Landmarks dos OUTROS DEDOS (Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20)
            NormalizedLandmark(x=0.1277, y=0.6540, z=-0.0233, visibility=0.0, presence=0.0), # 5 - Base do Indicador
            NormalizedLandmark(x=0.0830, y=0.6506, z=-0.0420, visibility=0.0, presence=0.0), # 6
            NormalizedLandmark(x=0.1060, y=0.6654, z=-0.0508, visibility=0.0, presence=0.0), # 7
            NormalizedLandmark(x=0.1255, y=0.6682, z=-0.0543, visibility=0.0, presence=0.0), # 8 - Ponta do Indicador
            # ... (e assim por diante para os 21 landmarks)
            NormalizedLandmark(x=0.1221, y=0.7031, z=-0.0242, visibility=0.0, presence=0.0), # 9 - Base do Dedo Médio
            NormalizedLandmark(x=0.0784, y=0.7058, z=-0.0397, visibility=0.0, presence=0.0), # 10
            NormalizedLandmark(x=0.1039, y=0.7144, z=-0.0415, visibility=0.0, presence=0.0), # 11
            NormalizedLandmark(x=0.1230, y=0.7163, z=-0.0415, visibility=0.0, presence=0.0), # 12 - Ponta do Dedo Médio
            NormalizedLandmark(x=0.1235, y=0.7530, z=-0.0274, visibility=0.0, presence=0.0), # 13 - Base do Dedo Anelar
            NormalizedLandmark(x=0.0845, y=0.7548, z=-0.0425, visibility=0.0, presence=0.0), # 14
            NormalizedLandmark(x=0.1054, y=0.7572, z=-0.0368, visibility=0.0, presence=0.0), # 15
            NormalizedLandmark(x=0.1236, y=0.7575, z=-0.0310, visibility=0.0, presence=0.0), # 16 - Ponta do Dedo Anelar
            NormalizedLandmark(x=0.1290, y=0.7992, z=-0.0318, visibility=0.0, presence=0.0), # 17 - Base do Dedo Mínimo
            NormalizedLandmark(x=0.0951, y=0.8009, z=-0.0416, visibility=0.0, presence=0.0), # 18
            NormalizedLandmark(x=0.1090, y=0.7995, z=-0.0364, visibility=0.0, presence=0.0), # 19
            NormalizedLandmark(x=0.1244, y=0.7999, z=-0.0310, visibility=0.0, presence=0.0)  # 20 - Ponta do Dedo Mínimo
        ]
    ],
    
    # Landmarks da mão em coordenadas de mundo (metros)
    # Estas coordenadas são 3D e dão a posição real da mão no espaço.
    hand_world_landmarks=[
        [
            # Landmark 0 (Pulso) - Note que os valores Z são muito maiores aqui (profundidade).
            Landmark(x=0.0549, y=0.0365, z=0.0555, visibility=0.0, presence=0.0),
            
            # Landmark 1 (Dedo Polegar)
            Landmark(x=0.0454, y=-0.0008, z=0.0332, visibility=0.0, presence=0.0),
            # ... (os 21 landmarks de mundo seguem a mesma ordem de NormalizedLandmark)
            Landmark(x=-0.0037, y=0.0354, z=0.0011, visibility=0.0, presence=0.0) # 20 - Ponta do Dedo Mínimo
        ]
    ]
)
```

Exemplo com dois gestos:
![alt text](image-6.png)

```python
GestureRecognizerResult( # Atributos dele
    # ==========================================================
    # GESTOS DETECTADOS (Gestures)
    # Lista de gestos, onde o primeiro elemento é a Mão 1 e o segundo é a Mão 2.
    # ==========================================================
    gestures=[
        [
            # Mão 1: Polegar para Cima (Thumb_Up)
            Category(
                index=-1, 
                score=0.632, 
                display_name='', 
                category_name='Thumb_Up'
            )
        ],
        [
            # Mão 2: Apontando para Cima (Pointing_Up)
            Category(
                index=-1, 
                score=0.757, 
                display_name='', 
                category_name='Pointing_Up'
            )
        ]
    ],
    
    # ==========================================================
    # CLASSIFICAÇÃO DA MÃO (Handedness)
    # A ordem corresponde à ordem dos gestos e landmarks.
    # ==========================================================
    handedness=[
        [
            # Mão 1: Classificada como Mão Direita
            Category(
                index=0, 
                score=0.772, 
                display_name='Right', 
                category_name='Right'
            )
        ],
        [
            # Mão 2: Classificada como Mão Esquerda
            Category(
                index=1, 
                score=0.995,  # Alta confiança na classificação da mão
                display_name='Left', 
                category_name='Left'
            )
        ]
    ],
    
    # ==========================================================
    # LANDMARKS DA MÃO (Coordenadas Normalizadas 0.0-1.0)
    # Contém duas listas de 21 Landmarks cada (uma para cada mão).
    # ==========================================================
    hand_landmarks=[
        # ------------------------------------------------------
        # HAND_LANDMARKS[0]: Mão Direita (Thumb_Up)
        # ------------------------------------------------------
        [
            NormalizedLandmark(x=0.1922, y=0.7747, z=-6.52e-08, ...), # 0 - Pulso
            NormalizedLandmark(x=0.1906, y=0.7037, z=-0.0146, ...),  # 1 - Dedo 1 (Polegar)
            # ... mais 19 landmarks da Mão Direita (omiti o resto para brevidade)
            NormalizedLandmark(x=0.1244, y=0.7999, z=-0.0310, ...) # 20 - Ponta do Dedo 5 (Mínimo)
        ], 
        # ------------------------------------------------------
        # HAND_LANDMARKS[1]: Mão Esquerda (Pointing_Up)
        # ------------------------------------------------------
        [
            NormalizedLandmark(x=0.9066, y=0.8752, z=-1.85e-07, ...), # 0 - Pulso
            NormalizedLandmark(x=0.8796, y=0.8213, z=-0.0205, ...),  # 1 - Dedo 1 (Polegar)
            # ... mais 19 landmarks da Mão Esquerda
            NormalizedLandmark(x=0.9354, y=0.8300, z=-0.0299, ...) # 20 - Ponta do Dedo 5 (Mínimo)
        ]
    ],
    
    # ==========================================================
    # LANDMARKS DE MUNDO (Coordenadas em Metros)
    # Contém duas listas de 21 Landmarks 3D.
    # ==========================================================
    hand_world_landmarks=[
        # ------------------------------------------------------
        # HAND_WORLD_LANDMARKS[0]: Mão Direita
        # ------------------------------------------------------
        [
            Landmark(x=0.0549, y=0.0365, z=0.0555, ...), # 0 - Pulso (em metros)
            # ... (20 landmarks restantes da Mão Direita)
            Landmark(x=-0.0037, y=0.0354, z=0.0011, ...)
        ], 
        # ------------------------------------------------------
        # HAND_WORLD_LANDMARKS[1]: Mão Esquerda
        # ------------------------------------------------------
        [
            Landmark(x=-0.0131, y=0.0641, z=0.0693, ...), # 0 - Pulso (em metros)
            # ... (20 landmarks restantes da Mão Esquerda)
            Landmark(x=0.0011, y=0.0487, z=0.0031, ...)
        ]
    ]
)
```

Se eu quiser saber o gesto da mão esquerda e as coordenadas do index finger da mão direita:
```python
print(GestureRecognizerResult.gestures[1][0].category_name) # Nos gestos -> Mão esquerda -> Attr do nome
print(GestureRecognizerResult.hand_landmarks[0][6:9]) # Nas landmarks normalizadas -> Mão Direita -> Landmarks do index finger -> Print de suas coords completas
```

Maniopulando mais ele
```python
pprint.pprint(f"{results.gestures}", indent=4)
# ("[[Category(index=-1, score=0.6321459412574768, display_name='', "
# "category_name='Thumb_Up')], [Category(index=-1, score=0.7573762536048889, "
# "display_name='', category_name='Pointing_Up')]]")

pprint.pprint(f"{results.gestures[0]}", indent=4) 
# ("[Category(index=-1, score=0.6321459412574768, display_name='', category_name='Thumb_Up')]")
print(type(results.gestures[0])),
# <class 'list'>
```

Se eu quiser percorrer cada mão do frame eu possof azer isso:
```python
base_options = python.BaseOptions(model_asset_path="C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=running_mode, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert BGR (OpenCV) -> RGB (MediaPipe Image)
#dentro do while de captura de frame:
while cv.isOpened():
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    w, h = 400 # O tamanho do frame
    
    # Get the gestures
    results = recognizer.recognize(mp_image)
    # Each hand will be represented as within an array:
    # [Category(index=-1, score=0.6321459412574768, display_name='', category_name='Thumb_Up')] <list>
    
    if not results.gestures or not results.hand_landmarks:
        return image, None
    
    for i, landmarks in enumerate(results.hand_landmarks):
        hand = {
            "side": results.handedness[i][0].category_name,
            "gesture": results.gestures[i][0],
            "landmarks": landmarks
        }
        x = int((hand["landmarks_origin"][0]*w) - 30)
        y = int((hand["landmarks_origin"][1]*h) - 30)
        cv.putText(img=image, text=hand["gesture"], org=(x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(120, 23, 190), lineType = cv.LINE_AA)
```

## Como eu fiz no meu código:

### Tenho uma classe para a mão e o detector dela
```python
class Hand():
    
    def __init__(self, side: str, gesture, landmarks: list):
        self.side = side
        self.sound_file_path = None
        self.landmarks = landmarks
        self.gesture = gesture.category_name
        self.landmarks_origin = (min(land.x for land in self.landmarks), min(land.y for land in self.landmarks))
        self.landmarks_end = (max(land.x for land in self.landmarks), max(land.y for land in self.landmarks))
        
    def getSoundFilePath(self):
        return self.sound_file_path
    
    def __repr__(self):
        landmarks_count = len(self.landmarks)
        
        return (
            f"Hand(\n"
            f"    side='{self.side}',\n"
            f"    gesture='{self.gesture}',\n"
            f"    landmarks_count={landmarks_count},\n"
            f"    sound_file_path='{self.sound_file_path}'\n"
            f"    landmarks={self.landmarks}\n"
            f"    origin={self.landmarks_origin}\n"
            f")"
        )
    
class HandSpeller():
    """
    The Gesture Recognizer
    """
    _MODEL_PATH = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/gesture_recognizer.task"
    
    def __init__(self, model = _MODEL_PATH, running_mode=vision.RunningMode.LIVE_STREAM):
        """_summary_

        Args:
            model (_type_, optional): _description_. Defaults to _MODEL_PATH.
            running_mode (_type_, optional): _description_. Defaults to vision.RunningMode.LIVE_STREAM. 
            Avaliable modes:
                - vision.RunningMode.LIVE_STREAM
                - vision.RunningMode.VIDEO
                - vision.RunningMode.IMAGE
        """
        self.base_options = python.BaseOptions(model_asset_path=model)
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options, running_mode=running_mode, num_hands=2)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
    
    def process_image(self, image, w, h):
        """Process a single image and return the recognition result
        ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert BGR (OpenCV) -> RGB (MediaPipe Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Get the gestures
        results = self.recognizer.recognize(mp_image)
        # Each hand will be represented as within an array:
        # [Category(index=-1, score=0.6321459412574768, display_name='', category_name='Thumb_Up')] <list>
        
        if not results.gestures or not results.hand_landmarks:
            return image, None
        
        #hand = Hand(side=results.handedness[0][0].display_name, pose=results.gestures[0][0])
        #print(hand)
        for i, landmarks in enumerate(results.hand_landmarks):
            hand = Hand(side=results.handedness[i][0].category_name, gesture=results.gestures[i][0], landmarks=landmarks)
            x = int((hand.landmarks_origin[0]*w) - 30)
            y = int((hand.landmarks_origin[1]*h) - 30)
            cv.putText(img=image, text=hand.gesture, org=(x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(120, 23, 190), lineType = cv.LINE_AA)

```

### E no meu loop de processo do frame do OpenCV:
Tenho uma classe para a câmera que tem os métodos de capturar e dentro dele capturar uma image (ou vídeo ao vivo mas ainda não coloquei, tenho prova amanhã né tropinha)
```python
class Camera:
    """
    Main camera handler for motion capture and gesture recognition.
    This class is responsible for:
    - Capturing frames from a video source (webcam or image file)
    - Running MediaPipe Hands on each frame
    - Drawing landmarks, labels, and bounding boxes
    - Recognizing gestures based on conditions stored in the database
    """
    
    _TOLERANCE_THRESHOLD = 0.02  # Small tolerance for landmark comparisons
    _CAPTURE_MODES = ["landmarks", "landmarks_coords", "bounding_box"]

    def __init__(self, name="Just Compose Beta", device=0, capture_mode=None):
        """
        Initialize the Camera

        Args:
            name (str, optional):
                Window title used by OpenCV when displaying the frames.
                Defaults to "Just Compose Beta".
            device (int | str, optional):
                Capture source.
                - If `int`, it is treated as an OpenCV camera index (0 for default webcam).
                - If `str` and the path ends with a compatible extension
                  ('.jpg', '.jpeg', '.png'), it is treated as an image file path.
                Defaults to 0.
            capture_mode (str | None, optional):
                Controls which text overlays are drawn on top of the detected landmarks.
                Supported values:
                    - "landmarks"         → draw only the landmark indices
                    - "landmarks_coords"  → draw landmark indices + normalized coordinates
                    - "bounding_box"      → draw the bounding box around the hand
                    - None                → do not draw any landmark labels
                Defaults to None.
        """

        # MediaPipe structure
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class attributes
        self.dj = DJ()
        self.name = name
        self.device = device
        self.compatible_file_types = ('.jpg', '.jpeg', '.png')
        self.capture_mode = capture_mode
        running_mode = vision.RunningMode.LIVE_STREAM if device == 0 else vision.RunningMode.IMAGE
        self.recognizer = HandSpeller(running_mode=running_mode)    
    
    def capture(self):
        """
        Initializes the motion capture process based on the configured device.
        
        This method acts as a router:
            - If device is a camera index (int), it delegates to real-time video processing.
            - If device is an image file path (str), it delegates to static image processing.
        
        The delegated process blocks the thread until the window is closed 
        or a termination key (ESC, 'q', or 'l') is pressed.
        """
        if self.device == 0:        
            self._process_video_stream()
        elif isinstance(self.device, str) and self.device.endswith(self.compatible_file_types):
            self._process_image()

   def _process_image(self):
        """
        Loads a static image from self.device and performs a single pass of
        hand detection and gesture recognition.
        
        Behavior:
            - Loads the image using OpenCV.
            - Runs MediaPipe Hands in static_image_mode=True once.
            - Draws landmarks and recognized gestures.
            
        The resulting image is displayed until a termination key is pressed 
        (ESC, 'q', 'l', or window close).
        """       
        image = cv.imread(self.device)
        w, h = self._get_frame_dimensions(image)
        self.recognizer.process_image(image, w, h)
        with self.mp_hands.Hands(static_image_mode=True) as hand_detector:
            results = hand_detector.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                self._capture_hands(image, results)
            cv.imshow(self.name, image)
            
            while True:
                key = cv.waitKey(1) 
                # Keyboard LEGACY
                if key in [27, ord("q"), ord("l")]:
                    break
                if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1:
                    break
                
        cv.destroyAllWindows()
```

-----
# PyFluidSynth

## Notas
| Nota musical    | Número MIDI |
| --------------- | ----------- |
| C3 (Dó)         | 48          |
| C4 (Dó central) | 60          |
| D4              | 62          |
| E4              | 64          |
| F4              | 65          |
| G4              | 67          |
| A4 (Lá 440Hz)   | 69          |
| B4              | 71          |
| C5              | 72          |

## Instrumento

No padrão General MIDI (GM):
program vai de 0 a 127
Cada número representa um instrumento
| Instrumento       | GM | program (0-based) |
| ----------------- | -- | ----------------- |
| Piano    | 1  | **0**             |
| Guitarra | 30 | **29**            |
| Baixo | 34 | **33**            |
| Synth      | 81 | **80**            |

### Bateria
Sempre no canal 10 (MIDI) → canal 9 (Python, 0-based)
| Peça           | Nota MIDI |
| -------------- | --------- |
| Kick           | 36        |
| Snare          | 38        |
| Pratos fechado | 42        |
| Pratos aberto  | 46        |

## play_note
def play_note(ch, prog, note, vel=110, dur=0.25, bank=0):
ch → canal
0–15

prog
> 0 → instrumentos normais
> 9 → bateria
> 0   # piano
> 29  # guitarra
> 33  # baixo
> 80  # synth lead

note → nota musical

Exemplos:

60  # C4
62  # D4
64  # E4
72  # C5
36  # kick (drum)

vel → intensidade (força)

0–127

Mais alto = mais forte

vel=80   
vel=120  

dur → duração (segundos)

Quanto tempo a nota fica ligada:

dur=0.15 
dur=0.4  

bank → banco (normalmente 0)

Para bateria GM, usa bank=128

Para instrumentos normais, bank=0