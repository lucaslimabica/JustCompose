Total invested time: 50m

## How Does JustCompose Use MediaPipe Hands?

This section provided by Google Gemini details the primary attributes of the `results` object returned by the `mp.solutions.hands.Hands.process()` method, which are crucial for gesture recognition and visualization in the Just Compose application.

### 1\. `results.multi_hand_landmarks`

This is the most important attribute, containing the normalized 3D coordinates for all detected hands.

| Attribute | Type | Description | Key for Just Compose |
| :--- | :--- | :--- | :--- |
| **`multi_hand_landmarks`**| `List[LandmarkList]` | A list where each element is a single detected hand. If one hand is found, the list has one element; if two are found, it has two. **This list is `None` if no hands are detected.** | **Coordinate Source.** Used to calculate distances and angles to define gestures (e.g., finger straightness, pinch distance). |

**Accessing Individual Landmark Coordinates (e.g., Index Fingertip - Point 8):**

For each hand in the list, 21 unique landmarks are available. Coordinates are normalized to the image width/height (from $0.0$ to $1.0$).

```python
# Assuming 'hand_landmarks' is one element from the list
index_tip = hand_landmarks.landmark[8] 

x_normalized = index_tip.x
y_normalized = index_tip.y
z_depth = index_tip.z # Relative depth, useful for distance/volume control
```

### 2\. `results.multi_handedness`

This attribute provides the classification (Left or Right) for each hand detected in the frame.

| Attribute | Type | Description | Key for Just Compose |
| :--- | :--- | :--- | :--- |
| **`multi_handedness`**| `List[ClassificationList]` | A list containing the handedness label for each corresponding hand found in `multi_hand_landmarks`. **The order matches the landmarks list.** | **Role Differentiation.** Used to assign specific musical controls (e.g., Left Hand for instrument/volume change; Right Hand for playing notes). |

**Accessing the Label (Left/Right):**

You typically need to iterate through both `multi_hand_landmarks` and `multi_handedness` simultaneously using `zip()`.

```python
for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
    # 'label' will be a string: "Left" or "Right"
    label = handedness.classification[0].label 
    
    # Process gesture based on the hand's role
    if label == "Left":
        # Control volume/instrument
        pass
    else:
        # Play note/sound
        pass
```

### 3\. `results.multi_hand_world_landmarks` (Advanced)

This provides raw 3D coordinates independent of camera position or scale.

| Attribute | Type | Description | Key for Just Compose |
| :--- | :--- | :--- | :--- |
| **`multi_hand_world_landmarks`**| `List[LandmarkList]` | A list of landmarks in **real-world 3D coordinates** (measured in meters, relative to the wrist). These are not affected by the camera's perspective. | **Scale Invariant Gestures.** Useful for advanced logic where gesture size must be consistent regardless of the hand's distance from the camera. |

### ⚠️ Critical Note on Safety Check

The most common error is accessing attributes when no hands are detected. You must always use a conditional check before accessing any data:

```python
if results.multi_hand_landmarks and results.multi_handedness:
    # Safely process and iterate through the data
    pass
```