import os
import glob
import zipfile

def safe_save_path(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}({i}){ext}")
        i += 1
    return candidate

def extract_selected(source_dir, dest_dir, keyword="_4PR_"):
    os.makedirs(dest_dir, exist_ok=True)
    zips = glob.glob(os.path.join(source_dir, "**", "*.zip"), recursive=True)
    k = keyword.lower()

    for z in zips:
        try:
            with zipfile.ZipFile(z, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name_in_zip = info.filename
                    if k in name_in_zip.lower():
                        fname = os.path.basename(name_in_zip)
                        out_path = safe_save_path(dest_dir, fname)
                        with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                        print(f"[OK] {z} -> {out_path}")
        except zipfile.BadZipFile:
            print(f"[SKIP] 손상된 ZIP: {z}")

if __name__ == "__main__":
    extract_selected("1.Training", "train", "_4PR_")
    extract_selected("2.Validation", "test", "_4PR_")
