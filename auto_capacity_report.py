"""
Automated Capacity Report — desktop generator (Tkinter)

Pick the Volume and HC files, click Run — client data (POD, MRR, Lifecycle,
Go Live, Final Service Date) comes live from HubSpot instead of a manual
export, and the full Ideal cascade (Step 1 -> Step 2 -> Step 3) runs
automatically with the app's current default parameters, producing the same
Ideal Excel export the Streamlit app itself generates.

  GUI :  python auto_capacity_report.py

The HubSpot token is asked once and saved locally (next to this script, in
`.hubspot_config.json` — gitignored, never sent anywhere but HubSpot's API).
"""
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Text, StringVar, filedialog, messagebox, simpledialog
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).parent))

SETTINGS_FILE = Path(__file__).with_name(".auto_capacity_report_gui.json")


def _load_settings() -> dict:
    import json
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_settings(d: dict) -> None:
    import json
    SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def run_gui():
    import hubspot_client
    from pipeline_runner import run_pipeline

    settings = _load_settings()

    root = Tk()
    root.title("Automated Capacity Report")
    root.geometry("720x560")
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    ttk.Label(main, text="Automated Capacity Report", font=("Segoe UI", 15, "bold")).pack(anchor="w")
    ttk.Label(
        main,
        text="Pick Volume + HC. Client data (POD/MRR/Lifecycle/dates) is pulled live from "
             "HubSpot — no manual export needed. Runs the full Ideal cascade automatically.",
        wraplength=680, foreground="#555",
    ).pack(anchor="w", pady=(2, 10))

    vol_var = StringVar(value=settings.get("vol", ""))
    hc_var = StringVar(value=settings.get("hc", ""))
    out_var = StringVar(value=settings.get("out", str(Path.home() / "Desktop")))

    ff = ttk.LabelFrame(main, text="Input files", padding=10)
    ff.pack(fill="x")
    ff.columnconfigure(1, weight=1)

    def pick_file(var, title):
        def _cb():
            p = filedialog.askopenfilename(title=title, filetypes=[("Excel files", "*.xlsx *.xls")])
            if p:
                var.set(p)
        return _cb

    def pick_folder(var, title):
        def _cb():
            p = filedialog.askdirectory(title=title)
            if p:
                var.set(p)
        return _cb

    for i, (label, var, cmd) in enumerate([
        ("Volume/AHT file", vol_var, pick_file(vol_var, "Select the Volume/AHT Excel file")),
        ("HC Weekly Report", hc_var, pick_file(hc_var, "Select the HC Weekly Report Excel file")),
    ]):
        ttk.Label(ff, text=label).grid(row=i, column=0, sticky="w", pady=3)
        ttk.Entry(ff, textvariable=var).grid(row=i, column=1, sticky="ew", padx=6)
        ttk.Button(ff, text="Browse…", command=cmd).grid(row=i, column=2)

    ttk.Label(ff, text="Output folder").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Entry(ff, textvariable=out_var).grid(row=2, column=1, sticky="ew", padx=6)
    ttk.Button(ff, text="Browse…", command=pick_folder(out_var, "Select output folder")).grid(row=2, column=2)

    af = ttk.Frame(main)
    af.pack(fill="x", pady=10)
    run_btn = ttk.Button(af, text="▶  Run")
    run_btn.pack(side="left")
    open_btn = ttk.Button(af, text="Open output folder", state="disabled",
                          command=lambda: os.startfile(out_var.get()))
    open_btn.pack(side="left", padx=8)
    status = ttk.Label(af, text="Ready", foreground="#0a7")
    status.pack(side="left", padx=12)

    log_box = Text(main, height=20, wrap="word", bg="#11141a", fg="#dfe3ea", insertbackground="#dfe3ea")
    log_box.pack(fill="both", expand=True, pady=(6, 0))

    def log(msg: str):
        def _write():
            log_box.insert("end", f"{msg}\n")
            log_box.see("end")
        root.after(0, _write)

    def ensure_token() -> str | None:
        tok = hubspot_client.load_token()
        if tok:
            return tok
        tok = simpledialog.askstring(
            "HubSpot token",
            "Paste your HubSpot Private App token (pat-na1-...).\n"
            "It will be saved locally next to this script and never asked again.",
            show="*",
        )
        if tok:
            hubspot_client.save_token(tok)
        return tok or None

    def do_run():
        vol, hc, out = vol_var.get().strip(), hc_var.get().strip(), out_var.get().strip()
        if not vol or not Path(vol).is_file():
            messagebox.showerror("Missing file", "Pick a valid Volume/AHT file first.")
            return
        if not hc or not Path(hc).is_file():
            messagebox.showerror("Missing file", "Pick a valid HC Weekly Report file first.")
            return
        token = ensure_token()
        if not token:
            messagebox.showerror("Missing token", "A HubSpot token is required to continue.")
            return

        _save_settings({"vol": vol, "hc": hc, "out": out})
        run_btn.config(state="disabled")
        open_btn.config(state="disabled")
        status.config(text="Running…", foreground="#e6a700")
        log_box.delete("1.0", "end")
        log(f"Started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        def worker():
            try:
                out_path = run_pipeline(vol, hc, token, out, log=log)
                def _done():
                    status.config(text="Done ✓", foreground="#0a7")
                    run_btn.config(state="normal")
                    open_btn.config(state="normal")
                    messagebox.showinfo("Done", f"Report generated:\n{out_path}")
                root.after(0, _done)
            except Exception as e:
                tb = traceback.format_exc()
                def _fail():
                    log(tb)
                    status.config(text="Failed ✗", foreground="#d33")
                    run_btn.config(state="normal")
                    messagebox.showerror("Failed", str(e))
                root.after(0, _fail)

        threading.Thread(target=worker, daemon=True).start()

    run_btn.config(command=do_run)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
