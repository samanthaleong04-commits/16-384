import zipfile

def create_submission():
    name = input("Enter your andrew ID: ").strip()
    filename = f"{name}_hw0.zip"
    with zipfile.ZipFile(filename, 'w') as zf:
        zf.write('ex1.py')
        zf.write('ex2.py')
        zf.write('ex3.py')
    print(f"Created submission archive: {filename}")

if __name__ == "__main__":
    create_submission()