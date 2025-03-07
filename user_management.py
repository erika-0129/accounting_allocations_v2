import sqlite3
import bcrypt

# Create a database for users
def initialize_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Add a new user
def add_user(username, password, role='User'):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Hashed password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                       (username, hashed_password, role))
        conn.commit()
        print(f"User '{username}' added successfully")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists")
    finally:
        conn.close()

# Authenticate a user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT password, role FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        hashed_password, role = result
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            return role
        return None

# Admin: list all users
def list_users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT id, username, role FROM users')
    users = cursor.fetchall()
    conn.close()
    print("User List: ")
    for user in users:
        print(f"ID: {user[0]}, Username: {user[1]}, Role: {user[2]}")

# Admin: Delete User
def delete_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()
    print(f"User '{username}' deleted successfully")

# Main function
if __name__ == "__main__":
    initialize_database()

    while True:
        print("\nUser Management")
        print("1. Add User")
        print("2. Authenticate User")
        print("3. List Users (Admin only)")
        print("4. Delete User (Admin only)")
        print("5. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            username = input("Enter username: ")
            password = input("Enter password: ")
            role = input("Enter role (Admin/User): ").capitalize()
            add_user(username, password, role)
        elif choice == '2':
            username = input("Enter username: ")
            password = input("Enter password: ")
            role = authenticate_user(username, password)
            if role:
                print(f"Login successful! Role: {role}")
            else:
                print("Invalid username or password.")
        elif choice == '3':
            role = input("Enter your role: ").capitalize()
            if role == 'Admin':
                list_users()
            else:
                print("Access denied. Admin only.")
        elif choice == '4':
            role = input("Enter your role: ").capitalize()
            if role == 'Admin':
                username = input("Enter the username to delete: ")
                delete_user(username)
            else:
                print("Access denied. Admin only.")
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")