import sqlite3
import tkinter as tk
from tkinter import messagebox
import random
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# DATABASE SETUP
# ==========================================

conn = sqlite3.connect("smart_ration_qlearning.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS qlearning_training (
    episode INTEGER,
    state TEXT,
    action TEXT,
    reward INTEGER,
    q_value REAL
)
""")

conn.commit()

# ==========================================
# STATES AND ACTIONS
# ==========================================

states = ["High", "Medium", "Low"]

actions = [
    "Do Not Send",
    "Send Later",
    "Send Notification"
]

# ==========================================
# Q TABLE INITIALIZATION
# ==========================================

Q_table = np.zeros((len(states), len(actions)))

# ==========================================
# Q LEARNING PARAMETERS
# ==========================================

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2
episodes = 100

# ==========================================
# REWARD SYSTEM
# ==========================================

reward_table = {
    ("High", "Do Not Send"): 10,
    ("High", "Send Later"): 2,
    ("High", "Send Notification"): -10,

    ("Medium", "Do Not Send"): -2,
    ("Medium", "Send Later"): 5,
    ("Medium", "Send Notification"): 3,

    ("Low", "Do Not Send"): -5,
    ("Low", "Send Later"): 4,
    ("Low", "Send Notification"): 10
}

# ==========================================
# STORE GRAPH VALUES
# ==========================================

episode_list = []
reward_list = []

# ==========================================
# Q LEARNING TRAINING
# ==========================================

for episode in range(1, episodes + 1):

    # Random state
    state = random.choice(states)

    state_index = states.index(state)

    # Exploration vs Exploitation
    if random.uniform(0, 1) < epsilon:

        action_index = random.randint(0, 2)

    else:

        action_index = np.argmax(Q_table[state_index])

    action = actions[action_index]

    # Reward
    reward = reward_table[(state, action)]

    # Q Learning Formula
    old_value = Q_table[state_index, action_index]

    next_max = np.max(Q_table[state_index])

    new_value = old_value + learning_rate * (
        reward + discount_factor * next_max - old_value
    )

    Q_table[state_index, action_index] = new_value

    # Store graph data
    episode_list.append(episode)
    reward_list.append(reward)

    # Store in database
    cursor.execute(
        "INSERT INTO qlearning_training VALUES (?, ?, ?, ?, ?)",
        (
            episode,
            state,
            action,
            reward,
            float(new_value)
        )
    )

conn.commit()

# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Smart Ration Shop using Q-Learning")

root.geometry("1200x800")

root.configure(bg="white")

# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="AI Smart Ration Shop Notification System",
    font=("Arial", 24, "bold"),
    bg="white",
    fg="black"
)

title.pack(pady=20)

# ==========================================
# CANVAS
# ==========================================

canvas = tk.Canvas(
    root,
    width=1100,
    height=450,
    bg="lightgray"
)

canvas.pack()

# ==========================================
# PANEL POSITIONS
# ==========================================

positions = {
    "High": (50, 50, 350, 350),
    "Medium": (400, 50, 700, 350),
    "Low": (750, 50, 1050, 350)
}

# ==========================================
# COLORS
# ==========================================

colors = {
    "High": "red",
    "Medium": "orange",
    "Low": "lightgreen"
}

# ==========================================
# ICONS
# ==========================================

icons = {
    "Do Not Send": "❌",
    "Send Later": "⏳",
    "Send Notification": "📢"
}

# ==========================================
# DISPLAY LEARNED POLICY
# ==========================================

for state in states:

    x1, y1, x2, y2 = positions[state]

    color = colors[state]

    state_index = states.index(state)

    best_action_index = np.argmax(Q_table[state_index])

    best_action = actions[best_action_index]

    icon = icons[best_action]

    q_value = round(Q_table[state_index][best_action_index], 2)

    # Rectangle
    canvas.create_rectangle(
        x1,
        y1,
        x2,
        y2,
        fill=color,
        width=4
    )

    # State
    canvas.create_text(
        (x1 + x2) // 2,
        y1 + 40,
        text=f"Crowd Level : {state}",
        font=("Arial", 18, "bold")
    )

    # Action
    canvas.create_text(
        (x1 + x2) // 2,
        y1 + 120,
        text=f"Best Action : {best_action}",
        font=("Arial", 15, "bold")
    )

    # Q Value
    canvas.create_text(
        (x1 + x2) // 2,
        y1 + 190,
        text=f"Q Value : {q_value}",
        font=("Arial", 15, "bold")
    )

    # Icon
    canvas.create_text(
        (x1 + x2) // 2,
        y1 + 260,
        text=icon,
        font=("Arial", 45)
    )

# ==========================================
# SHOW TRAINING LOGS
# ==========================================

def show_logs():

    cursor.execute(
        "SELECT * FROM qlearning_training LIMIT 10"
    )

    rows = cursor.fetchall()

    log_text = ""

    for row in rows:

        log_text += (
            f"Episode : {row[0]}\n"
            f"State : {row[1]}\n"
            f"Action : {row[2]}\n"
            f"Reward : {row[3]}\n"
            f"Q Value : {round(row[4],2)}\n"
            f"-------------------------\n"
        )

    messagebox.showinfo("Q-Learning Logs", log_text)

# ==========================================
# SHOW GRAPH
# ==========================================

def show_graph():

    plt.figure(figsize=(10, 5))

    plt.plot(
        episode_list,
        reward_list,
        marker='o'
    )

    plt.title("Reward vs Episode")

    plt.xlabel("Episodes")

    plt.ylabel("Reward")

    plt.grid(True)

    plt.show()

# ==========================================
# SHOW Q TABLE
# ==========================================

def show_qtable():

    q_text = "\n======= Q TABLE =======\n\n"

    for i in range(len(states)):

        q_text += f"{states[i]}\n"

        for j in range(len(actions)):

            q_text += (
                f"{actions[j]} : "
                f"{round(Q_table[i][j],2)}\n"
            )

        q_text += "\n"

    messagebox.showinfo("Q Table", q_text)

# ==========================================
# BUTTON FRAME
# ==========================================

frame = tk.Frame(root, bg="white")

frame.pack(pady=20)

# ==========================================
# LOG BUTTON
# ==========================================

btn1 = tk.Button(
    frame,
    text="Show Training Logs",
    command=show_logs,
    font=("Arial", 13, "bold"),
    bg="black",
    fg="white",
    padx=15,
    pady=10
)

btn1.grid(row=0, column=0, padx=15)

# ==========================================
# GRAPH BUTTON
# ==========================================

btn2 = tk.Button(
    frame,
    text="Show Reward Graph",
    command=show_graph,
    font=("Arial", 13, "bold"),
    bg="darkgreen",
    fg="white",
    padx=15,
    pady=10
)

btn2.grid(row=0, column=1, padx=15)

# ==========================================
# Q TABLE BUTTON
# ==========================================

btn3 = tk.Button(
    frame,
    text="Show Q Table",
    command=show_qtable,
    font=("Arial", 13, "bold"),
    bg="darkblue",
    fg="white",
    padx=15,
    pady=10
)

btn3.grid(row=0, column=2, padx=15)

# ==========================================
# TERMINAL OUTPUT
# ==========================================

print("\n========= FINAL Q TABLE =========\n")

for i in range(len(states)):

    print(f"\nState : {states[i]}")

    for j in range(len(actions)):

        print(
            f"{actions[j]} : "
            f"{round(Q_table[i][j],2)}"
        )

# ==========================================
# RUN APPLICATION
# ==========================================

root.mainloop()

# ==========================================
# CLOSE DATABASE
# ==========================================

conn.close()