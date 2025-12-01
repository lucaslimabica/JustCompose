# 🎵 JustCompose Work in Progress
![Status](https://img.shields.io/badge/status-WIP-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange)

**Total invested time:** `31h40m`

## Overview

**JustCompose** is a gesture-controlled musical engine powered by **OpenCV** and **MediaPipe Hands**.
It interprets hand landmarks, gestures, angles, distances, and spatial relationships to trigger real-time musical interactions, turning you into a MUSICIAN MAGICIAN.

## 📚 **Table of Contents**
- [📦 System Overview](#-system-overview)
- [📸 How The Capture Works](#-how-the-capture-works)
- [✋ How MediaPipe Returns a Detected Hand](#-how-mediapipe-returns-a-detected-hand)
  - [`multi_hand_landmarks`](#multi_hand_landmarks)
  - [`multi_handedness`](#-multi_handedness)
  - [`multi_hand_world_landmarks`](#-multi_hand_world_landmarks-real-world-3d-coordinates)
- [🎯 Visual Processing Milestones](-#milestones)
  - [Draw Landmarks](#-draw-landmarks)
  - [Draw Bounding Boxes](#-draw-bounding-boxes)
- [🗄️ The Database Structure](#-the-database-structure)
  - [`gesture` Table](#-table-gesture)
  - [`gesture_condition` Table](#-table-gesture_condition)
  - [How Conditions Work](#-how-conditions-work)
  - [Final Combined Data Structure](#-final-combined-data-structure)


## 📦 System Overview

**JustCompose is different, it's _spicy_**. Gestures, rules, and interactions are **data-driven**, stored in a database, interpreted dynamically at runtime, and completely independent of the camera resolution.

Processing pipeline:
1. Capture video using **OpenCV**
2. Detect hands using **MediaPipe**
3. Extract 21 3D landmarks
4. Process gesture logic (distances and positions conditions between landmarks)
5. Trigger sounds based on recognized gestures

## 📸 How The Capture Works

OpenCV captures frames in a continuous loop:
frame (OpenCV) → detect (MediaPipe) → draw & process (MediaPipe and JustCompose Built-in Methods) → next frame

## ✋ How MediaPipe Returns a Detected Hand

The MediaPipe's hand object is storaged at the `results` variable.
The object has these three attributtes:

- multi_hand_landmarks
- multi_handedness
- multi_hand_world_landmarks

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

-----

## Milestones

### 🎯 Draw Landmarks

The `draw_landmarks` step is responsible for rendering the detected hands on the frame using MediaPipe’s landmarks:

- Iterates over each detected hand and its corresponding handedness (`Left` / `Right`).
- Colors are used to visually distinguish hands:
  - **Right hand** → `RGB(235, 137, 52)`
  - **Left hand** → `RGB(235, 52, 113)`
- A **confidence-based color** is also computed for the landmarks (`score_color`), transitioning from yellow to green depending on the detection score.
- Uses `mp_drawing.draw_landmarks(...)` to draw:
  - the 21 hand landmarks,
  - the connections between them,
  - custom styles for points and lines.

On top of that, `draw_landmarks` also calls:

- `draw_landmark_names(...)` → optionally draws the landmark index or coordinates next to each point, depending on the current `capture_mode` (useful for debugging and gesture design).
- `draw_bounding_box(...)` → draws a bounding box around the hand based on key landmarks.

![Landmarks](./docs/image.png)

This milestone provides a **clear visual representation of the hand pose**, making it easier to reason about gestures and interactions.

### 🟩 Draw Bounding Boxes

The `draw_bounding_box` step computes and draws a rough bounding box around the detected hand:

- Converts the normalized landmark coordinates (from MediaPipe) into pixel coordinates using the current frame size.
- Uses the most left, right, top and bottom landmarks as reference points:
- Builds a rectangle from these landmarks and adds a small padding (`± 20px`) to avoid a tight crop.
- Draws the final box with `cv.rectangle(...)` in green.

This milestone is the basis for:

- **Region-of-interest processing**
- **Gesture-based UI elements**
- Future features like cropping the hand area, tracking, or triggering effects when the hand enters a certain region.

![Landmarks with Bounding Box](./docs/landmarks_with_bb.png)

---

## 🗄️ The Database Structure
This section describes the database schema used to store, configure, and recognize custom hand gestures for motion capture and real-time interaction.
The system is built on **two relational tables**:

* `gestures` → High-level metadata
* `gesture_conditions` → Booleans landmark-based constraints

This structure allows gestures to be **fully data-driven**, editable without modifying code, and easily expandable.

### 🧩 Table: `gesture`

The `gesture` table stores descriptive and functional information about each gesture.

| Field         | Type    | Description                                                | Example                      |
| ------------- | ------- | ---------------------------------------------------------- | ---------------------------- |
| `id`          | Integer | Primary key, unique identifier for the gesture             | `1`                          |
| `name`        | String  | Human-readable name of the gesture                         | `'Number One'`               |
| `description` | Text    | Explanation of what the gesture represents or how it looks | `'Index finger pointing up'` |
| `sound_file`  | String  | Relative path to the audio triggered on recognition        | `'./assets/boing.mp3'`       |

### Example record

```python
{
  'id': 1,
  'name': 'Number One',
  'description': 'Index finger pointing up gesture',
  'sound_file': './assets/boing.mp3'
}
```

### 🧠 Table: `gesture_condition`

The `gesture_conditions` table defines the **mathematical rules** that must hold true for a gesture to be considered recognized.
Each row corresponds to **one Boolean condition** derived from MediaPipe’s 21 normalized hand landmarks.

| Field        | Type    | Description                                                          | Example         |
| ------------ | ------- | -------------------------------------------------------------------- | --------------- |
| `id`         | Integer | Primary key for the condition                                        | `1`             |
| `gesture_id` | Integer | Foreign key → links to the `gestures` table                          | `1`             |
| `landmark_a` | Integer | MediaPipe landmark ID (0–20), first operand                          | `8` (Index tip) |
| `operator`   | String  | Comparison operator (`<`, `>`, `<=`, `>=`, `==`)                     | `'<'`           |
| `landmark_b` | Integer | MediaPipe landmark ID used as the second operand                     | `6` (Index PIP) |
| `axis`       | String  | Axis to compare: `'x'`, `'y'`, `'z'`                                 | `'y'`           |
| `hand_side`  | String  | Which hand the condition applies to: `'left'`, `'right'`, or `'any'` | `'any'`         |

#### Example rows (gesture “Number One”)

```
gesture_id, a, op, b, axis, hand_side, description
1,8,<,6,y,any,Index tip (8) is above PIP (6)
1,7,<,6,y,any,Index DIP (7) is above PIP (6)
1,6,<,5,y,any,Index PIP (6) is above MCP (5)
1,4,>,6,y,any,Thumb tip (4) is below PIP (6)
1,12,>,9,y,any,Middle finger folded
1,16,>,13,y,any,Ring finger folded
1,20,>,17,y,any,Pinky finger folded
1,4,>,6,x,right,Right hand → thumb right of index base
1,4,<,6,x,left,Left hand → thumb left of index base
```

### 🧮 How Conditions Work

A condition such as:

```
8.y < 6.y
```

Means:

> The Y-coordinate of landmark 8 (index fingertip) must be **less than** the Y-coordinate of landmark 6 (index PIP joint).
> Since Y decreases upward in MediaPipe, this confirms the **index finger is extended upwards**.

This representation allows gestures to be defined entirely through **mathematical relationships** in 3D space.

## 🔗 Final Combined Data Structure

When loaded from the database and connected inside the gesture recognizer, all conditions belonging to a gesture are grouped under the same ID:

```python
{
  1: {
    'name': 'Number One',
    'description': 'Index finger pointing up gesture',
    'sound': './assets/boing.mp3',
    'conditions': [
        {'a': 8, 'op': '<', 'b': 6, 'axis': 'y', 'side': 'any'},
        {'a': 7, 'op': '<', 'b': 6, 'axis': 'y', 'side': 'any'},
        {'a': 6, 'op': '<', 'b': 5, 'axis': 'y', 'side': 'any'},
        {'a': 4, 'op': '>', 'b': 6, 'axis': 'y', 'side': 'any'},
        {'a': 12, 'op': '>', 9, 'axis': 'y', 'side': 'any'},
        {'a': 16, 'op': '>', 13, 'axis': 'y', 'side': 'any'},
        {'a': 20, 'op': '>', 17, 'axis': 'y', 'side': 'any'},
        {'a': 4, 'op': '>', 'b': 6, 'axis': 'x', 'side': 'right'},
        {'a': 4, 'op': '<', 'b': 6, 'axis': 'x', 'side': 'left'}
    ]
  }
}
```

This unified structure enables the recognizer to:

1. Load gestures dynamically
2. Validate all conditions for each gesture
3. Trigger the associated action immediately

