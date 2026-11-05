"""Display the shell installer's questions; keep its decisions in one place.

fronk-install.sh emits @prompt records and waits for one answer on stdin. The
worker thread reads its output into a queue; only the Tk thread touches widgets.
Ordinary output becomes the log, choices become menus, and paths get pickers.
This UI never installs another computer: the head tells the user when to visit
a node and open the same manager there. See docs/INSTALL.md for the sequence.
"""
from pathlib import Path
import queue
import re
import socket
import subprocess
import sys
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:
    sys.exit("Fronk Setup requires Python 3 with Tk (Linux: install python3-tk), or run the manager with --terminal.")

HERE = Path(__file__).resolve().parent


class Manager:
    def __init__(self, window, args):
        self.win = window
        self.args = args
        self.proc = None
        self.inbox = queue.Queue()
        self.waiting = False
        self.running_app = False
        self.result = 0
        window.title("Fronk Installer Manager")
        window.geometry("740x420")
        window.minsize(600, 380)
        frame = ttk.Frame(window, padding=22)
        self.frame = frame
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="FRONK   /   INSTALLER MANAGER", foreground="#087b78").pack(anchor="w")
        self.title = tk.StringVar(value="Set up your cluster")
        ttk.Label(frame, textvariable=self.title, font=("TkDefaultFont", 19, "bold"), wraplength=620).pack(anchor="w", pady=(8, 2))
        ttk.Label(frame, text=f"This computer  ·  {socket.gethostname().split('.')[0]}").pack(anchor="w")
        self.hint = tk.StringVar(value="Start on the head, then follow its instructions on each node.")
        self.question = tk.StringVar(value="Ready to set up this computer?")
        ttk.Label(frame, textvariable=self.question, font=("TkDefaultFont", 13), wraplength=620).pack(anchor="w", pady=(22, 8))
        self.input = ttk.Frame(frame)
        self.input.pack(fill="x")
        ttk.Label(frame, textvariable=self.hint, wraplength=620, foreground="#555555").pack(anchor="w", pady=(10, 8))
        self.secret = ""
        self.pairing = ttk.Frame(frame)
        self.copy = ttk.Button(self.pairing, text="Copy pairing code", command=self.copy_secret)
        self.copy.pack(side="left")
        self.reveal = ttk.Button(self.pairing, text="Show code", command=self.toggle_secret)
        self.reveal.pack(side="left", padx=5)
        self.secret_text = ttk.Label(self.pairing)
        self.secret_text.pack(side="left")
        self.answer = tk.StringVar()
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(8, 12))
        self.again = ttk.Button(actions, text="Back to manager", command=self.restart)
        self.help = ttk.Button(actions, text="Copy request for a frontier model", command=self.copy_help)
        self.button = ttk.Button(actions, text="Continue", command=self.start)
        self.button.pack(side="right", pady=(12, 0))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.summary = tk.StringVar()
        ttk.Label(frame, textvariable=self.summary, wraplength=620).pack(anchor="w")
        self.demo_lines = []
        self.last_notice = ""
        self.details = ttk.Button(frame, text="Show details", command=self.toggle_details)
        self.details.pack(anchor="w", pady=(10, 0))
        self.log = ScrolledText(frame, height=10, width=80, wrap="word", state="disabled")
        self.details_open = False
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.bind("<Return>", lambda _: self.button.invoke())

    def write(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def toggle_details(self):
        self.show_details(not self.details_open)

    def show_details(self, visible):
        self.details_open = visible
        self.details.configure(text="Hide details" if visible else "Show details")
        if visible:
            self.log.pack(fill="both", expand=True, pady=(8, 0))
        else:
            self.log.pack_forget()
        self.fit_window()

    def fit_window(self):
        self.win.update_idletasks()
        self.win.geometry(f"740x{max(420, self.frame.winfo_reqheight())}")

    def busy(self, active):
        if active:
            self.progress.pack(fill="x", before=self.details, pady=(4, 4))
            self.progress.start(15)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def copy_help(self):
        # last_notice holds the installer's own message, which never carries the pairing code.
        error = self.last_notice.removeprefix("Fronk setup: ") if self.last_notice.startswith("Fronk setup:") else "[paste it from the details]"
        self.win.clipboard_clear()
        self.win.clipboard_append("Read docs/llm/DEBUG.md. Fronk setup stopped on this computer with this error: "
                                  f"\"{error}\". Help me find the cause, one check at a time, and tell me which computer to use. "
                                  "Keep my configuration and verified weights.")
        self.hint.set("Request copied. Paste it into a coding agent opened in your Fronk folder, or attach docs/llm/DEBUG.md to a chat model.")

    def copy_secret(self):
        self.win.clipboard_clear()
        self.win.clipboard_append(self.secret)
        self.copy.configure(text="Copied · copy pairing code again")

    def toggle_secret(self):
        visible = bool(self.secret_text.cget("text"))
        self.secret_text.configure(text="" if visible else self.secret)
        self.reveal.configure(text="Show code" if visible else "Hide code")

    def restart(self):
        # Open the installed-role menu rather than repeating the previous action.
        self.args = self.args[self.args.index("--prefix"):][:2] if "--prefix" in self.args else []
        self.again.pack_forget()
        self.help.pack_forget()
        self.pairing.pack_forget()
        self.secret = ""
        self.secret_text.configure(text="")
        self.reveal.configure(text="Show code")
        self.copy.configure(text="Copy pairing code")
        self.running_app = False
        self.result = 0
        self.start()

    def start(self):
        self.button.configure(state="disabled")
        self.question.set("Reading the local installation…")
        self.hint.set("")
        self.summary.set("")
        self.demo_lines = []
        self.last_notice = ""
        self.busy(True)
        args = [a for a in self.args if a not in ("--gui", "--terminal", "--dialog")]
        self.proc = subprocess.Popen(["bash", str(HERE / "fronk-install.sh"), "--dialog", *args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors="replace", bufsize=1)
        def receive():
            for line in self.proc.stdout:
                self.inbox.put(line.rstrip("\r\n"))
            self.inbox.put(self.proc.wait())
        threading.Thread(target=receive, daemon=True).start()
        self.win.after(50, self.poll)

    def poll(self):
        while not self.inbox.empty():
            item = self.inbox.get_nowait()
            if isinstance(item, int):
                self.busy(False)
                self.result = item
                self.title.set("Setup saved" if item == 0 else "Setup needs attention")
                self.question.set((self.last_notice or "Setup is saved. You can return to the manager at any time.") if item == 0 else
                                  (self.last_notice if self.last_notice.startswith("Fronk setup:") else "The last operation did not finish. See the details below."))
                self.hint.set("")
                if item:
                    self.show_details(True)
                    self.hint.set("Reopening the manager continues from what finished. See docs/TROUBLESHOOTING.md, or ask a frontier model for help.")
                    self.help.pack(side="left", padx=(0, 8))
                if (HERE / "fronk-install.sh").exists():
                    self.again.pack(side="left")
                self.button.configure(text="Close", command=self.close, state="normal")
                self.waiting = False
                self.fit_window()
                return
            if item.startswith("@prompt\t"):
                self.prompt(*item.split("\t", 4)[1:])
            elif item == "@running":
                self.busy(False)
                self.running_app = True
                self.title.set("Fronk is running")
                self.question.set("Close this window to stop the program.")
                self.button.configure(text="Stop and close", command=self.close, state="normal")
            elif item.startswith("@secret\t"):
                self.secret = item.split("\t", 1)[1]
                self.pairing.pack(anchor="w", before=self.details)
                self.write("Use the pairing-code button when setting up each node.")
            elif item.startswith("== "):
                self.title.set(item[3:])
                self.write(item[3:])
            elif item:
                self.write(item)
                if item.startswith(("DEMO estimate:", "DEMO where a pass goes:", "DEMO verdict:")):
                    self.demo_lines.append(item.removeprefix("DEMO "))
                    self.summary.set("\n".join(self.demo_lines[-3:]))
                if item.startswith(("On ", "Return to ", "Node setup pending", "Demo pending", "Removed ", "Fronk setup:")):
                    self.last_notice = item
        self.win.after(60, self.poll)

    def prompt(self, question, default, choices, kind):
        self.busy(False)
        self.waiting = True
        self.question.set(question)
        hint = ""
        if question.startswith("Install this machine"):
            hint = "Start with Head. Tested head: Apple Silicon with Metal. Intel Mac/Linux GPU heads are experimental."
        elif question.startswith("Expert backends:"):
            self.question.set("Use the included backends or add your own?")
            hint = "Keep included unless you have a custom source module."
        elif question.startswith("Select a compute backend"):
            self.title.set("Choose an expert backend")
            self.question.set("Which backend should this computer use?")
            hint = f"Keep {default}, the saved or recommended choice, unless you want to try another backend."
        elif question.startswith("How many node"):
            hint = "Keep this manager open. It will guide you through each node."
        elif question.startswith("Back on the head:"):
            name = re.search(r"verify (.+) now", question)
            if name:
                hint = f"On {name[1]}: open the same manager, choose Node, select its name, paste the pairing code and start. Then return here."
        elif question.startswith("All nodes are connected"):
            hint = "Measure this cluster with synthetic weights before downloading the model."
        elif question.startswith("After the demo"):
            hint = "These are synthetic estimates. Real chat speed also depends on the prompt and context."
        elif question.startswith("Use a file or folder") or question.startswith("Full path of "):
            hint = "Already have the weights? Choose your file or external SSD. Fronk verifies it in place."
        elif kind == "secret":
            hint = "Copy the pairing code from the head. It is kept private on these computers."
        self.hint.set(hint)
        self.answer.set(default)
        if choices:
            field = ttk.Combobox(self.input, values=choices.split("|"), textvariable=self.answer, state="readonly")
        else:
            field = ttk.Entry(self.input, textvariable=self.answer, show="•" if kind == "secret" else "")
        field.pack(side="left", fill="x", expand=True)
        if question.startswith("Full path of "):
            def browse():
                path = filedialog.askopenfilename(title="Choose the model", filetypes=[("GGUF weights", "*.gguf"), ("All files", "*")])
                if path:
                    self.answer.set(path)
            def folder():
                path = filedialog.askdirectory(title="Choose the folder containing the model")
                if path:
                    self.answer.set(path)
            ttk.Button(self.input, text="File…", command=browse).pack(side="left")
            ttk.Button(self.input, text="Folder…", command=folder).pack(side="left")
        elif question.startswith(("Download folder", "Backend module folder")):
            def destination():
                title = "Choose a backend module folder" if question.startswith("Backend module") else "Choose a download folder"
                path = filedialog.askdirectory(title=title)
                if path:
                    self.answer.set(path)
            ttk.Button(self.input, text="Folder…", command=destination).pack(side="left")
        field.focus_set()
        self.button.configure(text="Continue", command=self.submit, state="normal")
        self.fit_window()

    def submit(self):
        if not self.waiting:
            return
        value = self.answer.get()
        self.waiting = False
        self.answer.set("")
        for widget in self.input.winfo_children():
            widget.destroy()
        self.button.configure(state="disabled")
        self.question.set("Working on this computer…")
        self.hint.set("You can follow the technical output in Show details.")
        self.busy(True)
        try:
            self.proc.stdin.write(value + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            pass

    def close(self):
        if self.proc and self.proc.poll() is None:
            if self.running_app:
                if messagebox.askyesno("Stop Fronk?", "Stop the program launched by this window and close it?"):
                    self.proc.terminate()
                    self.proc.wait()
                    self.win.destroy()
                return
            if not self.waiting:
                messagebox.showinfo("Fronk Installer Manager", "A local operation is running. Close this window when it reaches the next prompt.")
                return
            if not messagebox.askyesno("Leave setup?", "End this setup session? You can reopen the manager to resume."):
                return
            self.proc.stdin.close()
            self.proc.wait()
        self.win.destroy()


if __name__ == "__main__":
    try:
        root = tk.Tk()
    except tk.TclError as error:
        sys.exit(f"Cannot open the graphical manager ({error}). Run the installer with --terminal.")
    manager = Manager(root, sys.argv[1:])
    root.mainloop()
    sys.exit(manager.result)
