📦 System Overview

**JustCompose is different, it's *spicy***. Gestures, rules, and interactions are **data-driven**, stored in a database, interpreted dynamically at runtime, and completely independent of the camera resolution.

Processing pipeline:

1. Capture video using **OpenCV**
2. Detect hands using **MediaPipe**
3. Extract 21 3D landmarks
4. Process gesture logic (distances and positions conditions between landmarks)
5. Trigger sounds based on recognized gestures

## 📸 How The Capture Works

OpenCV captures frames in a continuous loop:
**frame (OpenCV)** → **detect (MediaPipe)** → **draw & process (MediaPipe + JustCompose methods)** → **next frame**

## The Libraries Behind JustCompose

This project is basically a “real-time pipeline” connecting **computer vision** to **audio synthesis**. Below is how each library fits into that pipeline, what it gives you, and what tradeoffs it brings.

## 🟦 OpenCV (cv2): Real-Time Frames, Timing, and Pixel Space

OpenCV is your **frame clock**. It is responsible for:

* Opening the webcam feed (`cv.VideoCapture`)
* Providing the next frame (`cap.read()`)
* Handling window display (`cv.imshow`)
* Handling loop timing and exit keys (`cv.waitKey`)
* Allowing you to draw text, shapes, overlays, bounding boxes (UI/feedback)

### Why OpenCV matters in this project

MediaPipe will give you landmarks in **normalized space** (0–1), but your UI overlays live in **pixel space**. OpenCV is what connects:

* normalized landmark `(x, y)` → pixel `(px, py)` for drawing
* frame timing → audio timing decisions (cooldowns, debouncing)

### The “frame loop” contract

Real-time systems depend on one golden rule:

> **Never block the frame loop.**
> If you block, you drop frames and recognition becomes unstable.

That’s why in the audio part you avoided `time.sleep()` and used a `Timer` for note-off: it preserves the frame loop performance.

## 🟩 MediaPipe Hands + Gesture Recognizer: Hand Tracking + Gesture Labels

MediaPipe brings two major things:

1. **Hand landmark tracking**
2. **Gesture classification** (from a trained model)

In your code you use both:

* `mp.solutions.hands.Hands()` (classic landmark detector)
* `mediapipe.tasks.python.vision.GestureRecognizer` (gesture model in `.task`)

This is important: **these are two different APIs**.

* The `mp.solutions.hands` API returns `results.multi_hand_landmarks` and `results.multi_handedness`
* The `GestureRecognizer` returns a `GestureRecognizerResult` (gestures + handedness + landmarks in its own structure)

You’re essentially merging the best of both worlds:

* MediaPipe classic pipeline for stable detection + drawing
* GestureRecognizer for “semantic” gesture name output like `Open_Palm`, `Victory`, etc


## ✋ How MediaPipe Returns a Detected Hand

The MediaPipe's hand object is storaged at the `results` variable.
The object has these three attributtes:

* multi_hand_landmarks
* multi_handedness
* multi_hand_world_landmarks
* gesture

### multi_hand_landmarks

| Attribute              | Type                 | Description                      |
| ---------------------- | -------------------- | -------------------------------- |
| `multi_hand_landmarks` | `List[LandmarkList]` | Up to 2 hands, 21 landmarks each |

Each entry in multi_hand_landmarks represents one hand.
Inside each LandmarkList, there are 21 Landmark objects with normalized coordinates

```python
right_hand = multi_hand_landmarks[0]          # First detected hand
index_tip  = right_hand.landmark[8]           # Landmark 8 = index fingertip

print(index_tip)
>>> landmark = {
>>>     "x": 0.564345,
>>>     "y": 0.122322,
>>>     "z": 0.2212242
>>> }
```

#### 🗺️ Landmark Normalized Coordinates

MediaPipe uses **normalized coordinates**, meaning:

* `(0.0, 0.0)` → top-left corner
* `(1.0, 1.0)` → bottom-right corner
* Values between 0 and 1 represent proportions of the image

Example:

* Object A at `(0.50, 0.80)`
* Object B at `(0.40, 0.10)`

➡️ This means **A is to the right and below B**.

![Normalized coords](./docs/image.png)

This is the **most important** attribute. It contains 21 3D normalized landmarks for each detected hand.

### 🫲 `multi_handedness`

This attribute provides the classification (“Left” or “Right”) for every detected hand.
It **matches the order** of `multi_hand_landmarks`.

| Attribute          | Type                       | Description                                      |
| ------------------ | -------------------------- | ------------------------------------------------ |
| `multi_handedness` | `List[ClassificationList]` | Classification for each hand: “Left” or “Right”. |

#### 🔸 Accessing the Hand Label

```python
for hand_landmarks, handedness in zip(
    results.multi_hand_landmarks,
    results.multi_handedness
):
    label = handedness.classification[0].label  # "Left" or "Right"
    
    if label == "Left":
        # Instrument control
        pass
    else:
        # Note/sound triggering
        pass
```

### 🌍 `multi_hand_world_landmarks` Real-World 3D Coordinates

This attribute provides **true 3D coordinates (in meters)** relative to the wrist, independent of camera perspective.

| Attribute                    | Type                 | Description                                                                               |
| ---------------------------- | -------------------- | ----------------------------------------------------------------------------------------- |
| `multi_hand_world_landmarks` | `List[LandmarkList]` | Landmarks in realistic 3D space, unaffected by image scaling or distance from the camera. |


### `gesture` How this was worked with the `Hand` class  

While MediaPipe provides raw lists of coordinates and labels, our system needs a more organized way to handle this data in real-time. The `Hand` class acts as a **High-Level Abstraction Layer** that transforms raw detection results into a functional object for the musical engine.

### Why is this class crucial?

The `Hand` class is the backbone of the **Just Compose** logic for three main reasons:

1. **State Management:** It bundles the side (Left/Right), the semantic gesture (e.g., "Victory"), and the physical landmarks into a single "Source of Truth." This prevents synchronization errors when processing multiple hands.
2. **Spatial Awareness (Bounding Boxes):** The class automatically calculates `landmarks_origin` and `landmarks_end`. This allows the system to know exactly where the hand is in the frame, enabling features like dynamic UI overlays and proximity-based effects.
3. **Semantic Mapping:** It converts MediaPipe's `Category` objects into simple strings. This makes the musical logic (deciding which note to play) much more readable and easier to maintain.

### Conceptual Workflow

1. **Detection:** MediaPipe identifies 21 landmarks.
2. **Extraction:** We extract the label (handedness) and the gesture name.
3. **Encapsulation:** The `Hand` class is instantiated, calculating the hand's boundaries in the 2D space.
4. **Action:** The DJ/Synthesizer queries the `Hand` object to trigger sounds.

```python
# Conceptual usage in our main loop
current_hand = Hand(side="Right", gesture=mp_gesture, landmarks=mp_landmarks)

print(f"Hand: {current_hand.side} | Gesture: {current_hand.gesture}")
# Output: Hand: Right | Gesture: Open_Palm

```

### Key Attributes Breakdown

| Attribute | Logic Source | Importance |
| --- | --- | --- |
| `side` | `multi_handedness` | Determines if the hand controls the **Instrument** (Left) or the **Note** (Right). |
| `gesture` | `GestureRecognizer` | The trigger for specific musical events or commands. |
| `landmarks` | `multi_hand_landmarks` | Used for drawing and precise spatial calculations. |
| `landmarks_origin/end` | Calculated (Min/Max) | Defines the "Hitbox" of the hand for visual feedback. |


## 🎛️ FluidSynth + SoundFont (.sf2): Real Instruments Without Audio Files

This is the part that makes JustCompose feel like a real instrument.

### What FluidSynth does

FluidSynth is a software synthesizer that:

* loads a **SoundFont (.sf2)** with sampled instruments
* accepts **MIDI events** (note on/off, program change)
* produces audio in real time

### Why SoundFonts are great for this project

* You don’t need a folder full of `.mp3` assets
* You can switch instruments instantly using GM programs
* Notes are deterministic: if the same gesture happens, the same sound happens

### How your DJ uses it

In your project:

* **Left hand gesture → GM program (instrument)**
* **Right hand gesture → MIDI note (pitch)**
* `Closed_Fist` acts like a rest/pause and resets the “repeat lock”

Key calls:

* `sfload(sf2_path)` loads the bank
* `program_select(channel, sfid, bank, program)` chooses the instrument
* `noteon(channel, note, velocity)` starts sound
* `noteoff(channel, note)` stops sound

Your use of `Timer(dur, ...)` is a very important architecture choice:

* you keep the video loop non-blocking
* you still schedule note release correctly


## 🟨 Pygame: Audio State + Channel Management

Even though you switched to FluidSynth for real instruments, you still benefit from `pygame.mixer` because it gives you:

* a simple audio init on Windows
* a channel object with `.get_busy()` to prevent overlapping spam

In your `DJ` class, `self.ch.get_busy()` is essentially a “do I already have a note playing?” guard. Combined with cooldown + combo memory, this prevents the system from sounding chaotic in high FPS.

## 📀 Recording Audio: Why You Record “Generated Audio” Instead of System Audio

Windows system audio capture is painful (drivers, loopback devices, channel mismatch). Your project took the smarter route:

> **Record the audio you generate**, at the exact moment you generate it.

In your `_play_note`, when recording is enabled:

* you ask FluidSynth for samples (`get_samples`)
* convert them to `int16`
* write into a `.wav` file

### A Quick Deep Dive in How this Recording Buffer Works

When you trigger a gesture, the code executes a real-time audio "cloning" process. Let's break down the logic behind those four lines of code:

#### Digital Extraction: `self._fs.get_samples(frames_to_generate)`

Instead of recording through a microphone (which would capture background noise), we tap directly into the FluidSynth engine.

* `get_samples` asks the synthesizer: *"Hey, if you were to play this note right now, what would the raw mathematical wave look like?"*
* This returns raw floating-point numbers representing the sound wave.

#### Data Formatting: `np.int16(samples)`

Raw audio from synthesizers usually comes in `float32` (high precision decimals). However, the standard `.wav` format (PCM) expected by most players uses **16-bit Integers**.

* We use **NumPy** to perform a high-speed conversion of every single one of those ~32k samples.
* This makes the file compatible with any media player (VLC, Spotify, etc.) and keeps the file size optimized.

#### Storage: `writeframes(s16.tobytes())`

Finally, we convert the numeric array into raw **Binary Data (bytes)**.

* This is the moment the sound "leaves" the computer's RAM and is permanently etched onto the disk
* By doing this inside `_play_note`, your recording grows dynamically: every gesture adds a new "block" of sound to the file

## 🧱 Architecture: Why Classes Matter Here

### `Camera`

Owns the real-time loop and the frame lifecycle.

### `HandSpeller`

Owns semantic interpretation:

* gesture recognition
* UI overlays (text + icons)
* “hands_by_side” state, decoupled from frame resolution

### `DJ`

Owns audio rules:

* mapping gestures → program/note
* debounce logic (cooldown)
* rest logic
* optional audio recording




