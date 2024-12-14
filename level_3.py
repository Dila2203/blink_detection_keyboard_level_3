import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import numpy as np

class BlinkControlledKeyboard:
    def __init__(self):
        # Initialize Tkinter application
        self.root = tk.Tk()
        self.root.title("Blink-Controlled Keyboard")
        self.root.state("zoomed")

        # Text box for displaying typed text
        self.text_box = tk.Text(self.root, height=10, width=150)
        self.text_box.pack(pady=10)

        # Frame for keyboard and camera
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for the keyboard
        self.keyboard_frame = tk.Frame(self.main_frame)
        self.keyboard_frame.pack(side=tk.RIGHT, padx=20)

        # Frame for camera feed
        self.camera_frame = tk.Label(self.main_frame)
        self.camera_frame.pack(side=tk.LEFT, padx=20)

        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Blink detection parameters
        self.EAR_THRESHOLD = 0.19
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Hover selection tracking
        self.hover_start_time = None
        self.last_position = None
        self.HOVER_THRESHOLD = 2.0  # 2 seconds hover time for selection
        self.key_selected = False
        self.navigation_started = False  # Flag to track if navigation has started
        
        # Current keyboard state
        self.current_keyboard = 'regular'
        self.current_keys = []
        self.current_row = 0
        self.current_col = 0

        # Action cooldown to prevent rapid successive actions
        self.last_action_time = 0
        self.action_cooldown = 0.3

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize keyboard
        self.show_regular_keyboard()

    def calculate_ear(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio (EAR)."""
        def distance(point1, point2):
            return np.sqrt(np.sum((point1 - point2) ** 2))
        
        landmarks = np.array([(lm.x, lm.y) for lm in landmarks])
        left = landmarks[eye_points[0]]
        right = landmarks[eye_points[3]]
        top = landmarks[eye_points[1]]
        bottom = landmarks[eye_points[5]]
        
        vertical_dist = distance(top, bottom)
        horizontal_dist = distance(left, right)
        return vertical_dist / horizontal_dist

    def detect_blink(self, left_ear, right_ear):
        """Simplified blink detection for navigation only."""
        current_time = time.time()
        
        if current_time - self.last_action_time < self.action_cooldown:
            return None

        if left_ear < self.EAR_THRESHOLD and right_ear >= self.EAR_THRESHOLD:
            self.last_action_time = current_time
            self.navigation_started = True
            return "LEFT_WINK"

        if right_ear < self.EAR_THRESHOLD and left_ear >= self.EAR_THRESHOLD:
            self.last_action_time = current_time
            self.navigation_started = True
            return "RIGHT_WINK"

        if left_ear < self.EAR_THRESHOLD and right_ear < self.EAR_THRESHOLD:
            self.last_action_time = current_time
            self.navigation_started = True
            return "SHORT_BLINK"

        return None

    def key_pressed(self, key):
        """Handle key presses."""
        if key == 'Backspace':
            current_text = self.text_box.get("1.0", tk.END)[:-2]
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert(tk.END, current_text)
        elif key == 'Enter':
            self.text_box.insert(tk.END, '\n')
        elif key == 'Space':
            self.text_box.insert(tk.END, ' ')
        elif key == '123':
            self.show_numeric_keyboard()
        elif key == 'ABC':
            self.show_regular_keyboard()
        elif key == 'Emoji':
            self.show_emoji_keyboard()
        elif key == 'Back':  # New key for returning from emoji keyboard
            self.show_regular_keyboard()
        else:
            self.text_box.insert(tk.END, key)

    def create_keyboard(self, keys):
        """Create keyboard buttons."""
        for widget in self.keyboard_frame.winfo_children():
            widget.destroy()
        
        self.current_keys = keys
        for row_index, row in enumerate(keys):
            for col_index, key in enumerate(row):
                button = tk.Button(
                    self.keyboard_frame,
                    text=key,
                    width=8,
                    font=("Helvetica", 16),
                    bg='white'
                )
                button.grid(row=row_index, column=col_index, padx=10, pady=10)

    def highlight_current_key(self):
        """Highlight the current selected key with robust error handling."""
        if not self.navigation_started:
            return

        # Reset all buttons to white first
        for row_index, row in enumerate(self.current_keys):
            for col_index, key in enumerate(row):
                try:
                    button = self.keyboard_frame.grid_slaves(row=row_index, column=col_index)[0]
                    button.configure(bg='white')
                except IndexError:
                    continue

        # Highlight current key in blue with error handling
        try:
            button = self.keyboard_frame.grid_slaves(row=self.current_row, column=self.current_col)[0]
            button.configure(bg='light blue')
        except IndexError:
            # Reset to safe default if current selection is invalid
            self.current_row = 0
            self.current_col = 0
            try:
                button = self.keyboard_frame.grid_slaves(row=self.current_row, column=self.current_col)[0]
                button.configure(bg='light blue')
            except IndexError:
                print("Failed to highlight current key")

    def show_regular_keyboard(self):
        """Show regular keyboard layout."""
        keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['123', 'Emoji', 'Space', 'Backspace', 'Enter']
        ]
        self.create_keyboard(keys)
        self.current_keyboard = 'regular'
        self.current_row = 0
        self.current_col = 0
        self.hover_start_time = None
        self.key_selected = False
        self.navigation_started = False  # Reset navigation flag

    def show_numeric_keyboard(self):
        """Show numeric keyboard layout."""
        keys = [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '9'],
            ['ABC', '0', 'Backspace']
        ]
        self.create_keyboard(keys)
        self.current_keyboard = 'numeric'
        self.current_row = 0
        self.current_col = 0
        self.hover_start_time = None
        self.key_selected = False
        self.navigation_started = False  # Reset navigation flag

    def show_emoji_keyboard(self):
        """Show emoji keyboard layout."""
        keys = [
            ['😀', '😂', '😍'],
            ['😊', '😎', '😢'],
            ['👍', '👏', '❤️'],
            ['Back', '🙏', 'Backspace']
        ]
        self.create_keyboard(keys)
        self.current_keyboard = 'emoji'
        self.current_row = 0
        self.current_col = 0
        self.hover_start_time = None
        self.key_selected = False
        self.navigation_started = False  # Reset navigation flag

    def check_hover_selection(self):
        """Check if current key should be selected based on hover time."""
        if not self.navigation_started:
            return

        current_time = time.time()
        current_position = (self.current_row, self.current_col)
        
        if current_position != self.last_position:
            self.hover_start_time = current_time
            self.last_position = current_position
            self.key_selected = False
        elif self.hover_start_time is not None and not self.key_selected:
            if current_time - self.hover_start_time >= self.HOVER_THRESHOLD:
                if (0 <= self.current_row < len(self.current_keys) and 
                    0 <= self.current_col < len(self.current_keys[self.current_row])):
                    selected_key = self.current_keys[self.current_row][self.current_col]
                    self.key_pressed(selected_key)
                    self.key_selected = True

    def navigate_keyboard(self, action):
        """Navigate keyboard keys with blinks."""
        max_rows = len(self.current_keys)
        old_position = (self.current_row, self.current_col)
        
        if action == "RIGHT_WINK":  # Changed from LEFT_WINK
            self.current_col = (self.current_col + 1) % len(self.current_keys[self.current_row])
            if self.current_col == 0:
                self.current_row = (self.current_row + 1) % max_rows
        
        elif action == "LEFT_WINK":  # Changed from RIGHT_WINK
            self.current_col = (self.current_col - 1 + len(self.current_keys[self.current_row])) % len(self.current_keys[self.current_row])
            if self.current_col == len(self.current_keys[self.current_row]) - 1:
                self.current_row = (self.current_row - 1 + max_rows) % max_rows
                self.current_col = min(self.current_col, len(self.current_keys[self.current_row]) - 1)
        
        elif action == "SHORT_BLINK":
            self.current_row = (self.current_row + 1) % max_rows
            self.current_col = min(self.current_col, len(self.current_keys[self.current_row]) - 1)

        new_position = (self.current_row, self.current_col)
        if new_position != old_position:
            self.key_selected = False
            self.hover_start_time = time.time()

        self.highlight_current_key()

    def update_frame(self):
        """Capture and process video frame."""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_ear = self.calculate_ear(landmarks.landmark, self.LEFT_EYE)
                right_ear = self.calculate_ear(landmarks.landmark, self.RIGHT_EYE)
                
                action = self.detect_blink(left_ear, right_ear)
                if action:
                    self.navigate_keyboard(action)

        self.check_hover_selection()

        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_frame.imgtk = imgtk
        self.camera_frame.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def run(self):
        """Start the application."""
        self.update_frame()
        self.root.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    keyboard = BlinkControlledKeyboard()
    keyboard.run()

if __name__ == "__main__":
    main()