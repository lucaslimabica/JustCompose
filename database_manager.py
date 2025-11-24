import sqlite3

con = sqlite3.connect("justcompese.db")
cursor = con.cursor()

def create_table_gesture():
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
    create_table_gesture()
    create_table_gesture_condition()
    con.commit()
    con.close()

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
    cursor.execute("""
        INSERT INTO gesture_condition (gesture_id, landmark_a, operator, landmark_b, axis, hand_side)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (gesture_id, landmark_a, operator, landmark_b, axis, hand_side))
    con.commit()
    return cursor.lastrowid

initialize_database()