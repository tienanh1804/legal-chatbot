import sqlite3
import os

# Đường dẫn đến file SQLite
db_path = "ragapp.db"

# Kiểm tra xem file có tồn tại không
if not os.path.exists(db_path):
    print(f"Không tìm thấy file cơ sở dữ liệu: {db_path}")
    exit(1)

# Kết nối đến cơ sở dữ liệu
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Kiểm tra xem cột conversation_id đã tồn tại chưa
    cursor.execute("PRAGMA table_info(query_history)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    
    if "conversation_id" not in column_names:
        print("Thêm cột conversation_id vào bảng query_history...")
        cursor.execute("ALTER TABLE query_history ADD COLUMN conversation_id INTEGER")
        
        # Cập nhật dữ liệu hiện có, đặt conversation_id = id
        cursor.execute("UPDATE query_history SET conversation_id = id WHERE conversation_id IS NULL")
        
        conn.commit()
        print("Đã thêm cột conversation_id thành công!")
    else:
        print("Cột conversation_id đã tồn tại!")
    
except Exception as e:
    conn.rollback()
    print(f"Lỗi: {e}")
finally:
    conn.close()
