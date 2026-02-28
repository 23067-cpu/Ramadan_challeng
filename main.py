"""
SupNum - S3C'1447 Challenge Ramadan
Fichier : main.py
Description : Fichier développé en Python pour la résolution du problème RCPSP.
Tous les algorithmes et logiques internes sont optimisés pour des performances maximales.
"""
"""
main.py — RCPSP Solver Desktop Application
SupNum Coding Challenge S3C'1447

Entry point. Launches the Tkinter GUI application.
Run: python main.py
"""

import sys
import os

def check_dependencies():
    """Check required packages and print helpful messages if missing."""
    missing = []
    optional_missing = []

    required = ['tkinter']
    optional = {
        'matplotlib': 'pip install matplotlib  (for Gantt chart)',
        'openpyxl':   'pip install openpyxl    (for Excel export)',
        'docx':       'pip install python-docx  (for Word export)',
    }

    # Check tkinter
    try:
        import tkinter
    except ImportErreur:
        print("ERROR: tkinter is not available. Please install Python with Tk support.")
        sys.exit(1)

    # Check optional
    for pkg, install_hint in optional.items():
        try:
            __import__(pkg)
        except ImportErreur:
            optional_missing.append(install_hint)

    if optional_missing:
        print("─" * 60)
        print("Optional packages not found (app will still run):")
        for msg in optional_missing:
            print(f"  • {msg}")
        print("─" * 60)


def main():
    print("=" * 60)
    print("  RCPSP Solver — SupNum Coding Challenge S3C'1447")
    print("  Hybrid Genetic Algorithm + Neighborhood Search")
    print("=" * 60)

    check_dependencies()

    # Ensure working directory is the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    from app_gui import RCPSPApp
    app = RCPSPApp()

    # Auto-load the j60.sm folder if it exists
    j60_path = os.path.join(script_dir, 'j60.sm')
    if os.path.isdir(j60_path):
        print(f"Auto-loading j60.sm dataset from: {j60_path}")
        app.after(500, lambda: app._load_instances(j60_path))
        app.dataset_folder = j60_path
        app.folder_var.set("j60.sm/")

    app.mainloop()


if __name__ == '__main__':
    main()
