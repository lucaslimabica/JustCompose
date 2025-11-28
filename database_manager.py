import sqlite3


def create_table_gesture():
    _, cursor = connect_database()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gesture (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sound_file TEXT DEFAULT "./assets/boing.mp3"
        )
    """)
    
def create_table_gesture_condition():
    _, cursor = connect_database()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gesture_condition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture_id INTEGER NOT NULL,
            
            landmark_a INTEGER NOT NULL,
            operator TEXT NOT NULL,
            landmark_b INTEGER NOT NULL,
            axis TEXT NOT NULL,
    
            hand_side TEXT DEFAULT "any",
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (gesture_id) REFERENCES gesture(id) ON DELETE CASCADE
        )
    """)
    
def initialize_database():
    con, _ = connect_database()
    create_table_gesture()
    create_table_gesture_condition()
    con.commit()
    con.close()
    
def connect_database():
    con = sqlite3.connect("justcompese.db")
    cursor = con.cursor()
    return con, cursor

# Functions to interact with the database at the custom modal
def create_gesture(name, description="", sound_file="./assets/boing.mp3") -> int:
    """
    Creates a new gesture in the database
    
    Args:
        name (str): Name of the gesture
        description (str, optional): Description of the gesture. Defaults to ""
        sound_file (str, optional): Path to the sound file associated with the gesture. Defaults to "./assets/boing.mp3"
    
    Returns:
        int: ID of the newly created gesture
    """
    con, cursor = connect_database()
    cursor.execute("""
        INSERT INTO gesture (name, description, sound_file)
        VALUES (?, ?, ?)
    """, (name, description, sound_file))
    con.commit()
    return cursor.lastrowid

def create_gesture_condition(gesture_id, landmark_a, operator, landmark_b, axis, hand_side="any"):
    """
    Creates a new gesture condition in the database
    
    Args:
        gesture_id (int): ID of the gesture this condition belongs to
        landmark_a (int): Index of the first landmark
        operator (str): Comparison operator (e.g., ">", "<", "==")
        landmark_b (int): Index of the second landmark
        axis (str): Axis for comparison ("x" or "y")
        hand_side (str, optional): Side of the hand ("left", "right", "any"). Defaults to "any"
    """
    con, cursor = connect_database()
    cursor.execute("""
        INSERT INTO gesture_condition (gesture_id, landmark_a, operator, landmark_b, axis, hand_side)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (gesture_id, landmark_a, operator, landmark_b, axis, hand_side))
    con.commit()
    return cursor.lastrowid

# Functions to load gestures and their conditions to validate in each frame there is a recognized gesture
def load_all_gestures():
    """
    Loads all gestures and their associated conditions from the database
    
    Returns:
        dict: Dict of all gestures with their conditions as keys
    """
    con, cursor = connect_database()
    query = """
    SELECT g.id, g.name, g.description, g.sound_file,
           c.landmark_a, c.operator, c.landmark_b, c.axis, c.hand_side
    FROM gesture g
    JOIN gesture_condition c ON g.id = c.gesture_id
    ORDER BY g.id;
    """

    rows = cursor.execute(query).fetchall()
    
    gestures = {}

    for row in rows:
        gid = row[0]

        if gid not in gestures:
            gestures[gid] = {
                "name": row[1],
                "description": row[2],
                "sound": row[3],
                "conditions": []
            }

        gestures[gid]["conditions"].append({
            "a": row[4],
            "op": row[5],
            "b": row[6],
            "axis": row[7],
            "side": row[8]
        })

    return gestures




# Example usage
def example():
    initialize_database()

    # Creating a sample gesture and its conditions
    gesture = create_gesture("Number One", "Index finger pointing up gesture")

    # Index finger up condition
    create_gesture_condition(gesture, 8, "<", 6, "y")  # Index tip above index pip (at y-axis the smaller the value, the higher it is)
    create_gesture_condition(gesture, 7, "<", 6, "y")
    create_gesture_condition(gesture, 6, "<", 5, "y")

    # Other fingers down condition
    create_gesture_condition(gesture, 4, ">", 6, "y")
    create_gesture_condition(gesture, 12, ">", 9, "y")
    create_gesture_condition(gesture, 16, ">", 13, "y")
    create_gesture_condition(gesture, 20, ">", 17, "y")

    # Thumb position condition (thumb tip to the side of the hand)
    create_gesture_condition(gesture, 4, ">", 6, "x", hand_side="right")
    create_gesture_condition(gesture, 4, "<", 6, "x", hand_side="left")

